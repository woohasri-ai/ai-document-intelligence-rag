import unittest

from src.main import (
    build_grounded_prompt,
    create_section_aware_chunks,
    filter_by_distance,
    get_section_name,
    get_topic_aliases,
    prioritize_matching_sections,
    section_match,
)


class TestTopicAliases(unittest.TestCase):

    def test_rag_topic_is_detected(self):
        self.assertIn("rag", get_topic_aliases("Explain RAG"))

    def test_embeddings_are_associated_with_rag(self):
        aliases = get_topic_aliases(
            "Explain the purpose of vector embeddings in RAG"
        )
        self.assertIn("rag", aliases)

    def test_machine_learning_topic_is_detected(self):
        self.assertIn(
            "machine learning",
            get_topic_aliases("Explain machine learning"),
        )

    def test_multiple_topics_are_detected(self):
        aliases = get_topic_aliases("Explain RAG and deep learning")
        self.assertIn("rag", aliases)
        self.assertIn("deep learning", aliases)

    def test_unrelated_question_has_no_topic_aliases(self):
        self.assertEqual(
            get_topic_aliases("What is the capital of Japan?"),
            [],
        )

    def test_phase_number_is_detected(self):
        self.assertIn(
            "rag",
            get_topic_aliases("What should I learn in Phase 6?"),
        )


class TestSectionMatching(unittest.TestCase):

    def test_rag_section_name_is_detected(self):
        self.assertEqual(
            get_section_name(
                "PHASE 6 — RAG Learn embeddings and chunking"
            ),
            "PHASE 6 — RAG",
        )

    def test_rag_section_matches_rag_topic(self):
        self.assertTrue(section_match("PHASE 6 — RAG", ["rag"]))

    def test_deep_learning_does_not_match_rag(self):
        self.assertFalse(
            section_match("PHASE 3 — Deep Learning", ["rag"])
        )

    def test_phase_zero_heading_is_recognized(self):
        self.assertEqual(
            get_section_name("PHASE 0 — Python Foundation"),
            "PHASE 0 — Python Foundation",
        )


class TestDistanceFiltering(unittest.TestCase):

    def setUp(self):
        self.candidates = [
            {"id": "close", "distance": 0.8},
            {"id": "boundary", "distance": 1.65},
            {"id": "far", "distance": 1.9},
        ]

    def test_rejects_candidates_above_threshold(self):
        eligible = filter_by_distance(
            self.candidates,
            max_distance=1.65,
        )
        self.assertEqual(
            [item["id"] for item in eligible],
            ["close", "boundary"],
        )

    def test_accepts_candidate_at_threshold(self):
        eligible = filter_by_distance(
            [{"id": "boundary", "distance": 1.65}],
            max_distance=1.65,
        )
        self.assertEqual(len(eligible), 1)

    def test_rejects_all_candidates_above_threshold(self):
        eligible = filter_by_distance(
            [{"id": "far", "distance": 1.9}],
            max_distance=1.65,
        )
        self.assertEqual(eligible, [])

    def test_empty_candidates_return_empty_result(self):
        self.assertEqual(
            filter_by_distance([], max_distance=1.65),
            [],
        )


class TestTopicPrioritization(unittest.TestCase):

    def setUp(self):
        self.candidates = [
            {
                "id": "rag_chunk",
                "matched_section": True,
                "distance": 1.2,
            },
            {
                "id": "other_chunk",
                "matched_section": False,
                "distance": 0.9,
            },
        ]

    def test_matching_section_is_prioritized(self):
        selected, applied = prioritize_matching_sections(
            self.candidates,
            ["rag"],
        )
        self.assertTrue(applied)
        self.assertEqual(
            [item["id"] for item in selected],
            ["rag_chunk"],
        )
   
    def test_falls_back_when_no_section_matches(self):
        candidates = [
            {
                "id": "rag_chunk",
                "matched_section": False,
                "distance": 1.2,
            },
            {
                "id": "other_chunk",
                "matched_section": False,
                "distance": 0.9,
            },
        ]

        selected, applied = prioritize_matching_sections(
            candidates,
            ["deep learning"],
        )

        self.assertFalse(applied)
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            [item["id"] for item in selected],
            ["rag_chunk", "other_chunk"],
        )

    def test_no_topic_uses_semantic_candidates(self):
        selected, applied = prioritize_matching_sections(
            self.candidates,
            [],
        )
        self.assertFalse(applied)
        self.assertEqual(len(selected), 2)

    def test_empty_candidates_are_handled(self):
        selected, applied = prioritize_matching_sections([], ["rag"])
        self.assertEqual(selected, [])
        self.assertFalse(applied)


class TestGroundedPrompt(unittest.TestCase):

    def setUp(self):
        self.question = "What should I learn in RAG?"
        self.chunk_ids = ["chunk_7"]
        self.contexts = [
            "Learn embeddings, chunking, and retrieval."
        ]
        self.metadata = [
            {
                "source": "Aiml map .pdf",
                "section_name": "PHASE 6 — RAG",
                "source_pages": "10,11,12",
            }
        ]

    def test_prompt_contains_question_and_citation_rules(self):
        prompt = build_grounded_prompt(
            self.question,
            self.chunk_ids,
            self.contexts,
            self.metadata,
        )
        self.assertIn(self.question, prompt)
        self.assertIn("chunk_7", prompt)
        self.assertIn("PDF page(s): 10,11,12", prompt)
        self.assertIn("Never invent page numbers", prompt)

    def test_prompt_rejects_empty_context(self):
        with self.assertRaises(ValueError):
            build_grounded_prompt("Explain RAG", [], [], [])

    def test_prompt_rejects_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            build_grounded_prompt(
                "Explain RAG",
                ["chunk_1"],
                [],
                [],
            )

    def test_prompt_includes_section_name(self):
        prompt = build_grounded_prompt(
            self.question,
            self.chunk_ids,
            self.contexts,
            self.metadata,
        )
        self.assertIn("PHASE 6 — RAG", prompt)


class TestChunkCreation(unittest.TestCase):

    def test_chunks_preserve_page_and_section_metadata(self):
        pages = [
            {
                "page": 1,
                "text": (
                    "PHASE 6 — RAG "
                    + " ".join(["embeddings"] * 30)
                ),
            },
            {
                "page": 2,
                "text": (
                    "PHASE 3 — Deep Learning "
                    + " ".join(["neural"] * 30)
                ),
            },
        ]

        chunks, page_numbers, section_names = (
            create_section_aware_chunks(
                pages,
                chunk_size=20,
                overlap=5,
            )
        )

        self.assertTrue(chunks)
        self.assertEqual(len(chunks), len(page_numbers))
        self.assertEqual(len(chunks), len(section_names))
        self.assertIn("PHASE 6 — RAG", section_names)
        self.assertIn("PHASE 3 — Deep Learning", section_names)

        for pages_for_chunk in page_numbers:
            self.assertTrue(pages_for_chunk)

    def test_empty_pages_raise_error(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "The PDF contains no searchable words.",
        ):
            create_section_aware_chunks(
                [],
                chunk_size=20,
                overlap=5,
            )


if __name__ == "__main__":
    unittest.main()
