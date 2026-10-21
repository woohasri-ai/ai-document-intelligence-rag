
import unittest
from unittest.mock import Mock, patch

from src import gemini_service


class TestGeminiAnswerGeneration(unittest.TestCase):

    def setUp(self):
        self.client = Mock()

        self.model_patch = patch(
            "src.gemini_service.list_text_generation_models",
            return_value=["test-flash"],
        )
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

        self.sleep_patch = patch("src.gemini_service.time.sleep")
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def test_returns_answer_and_model_name(self):
        self.client.models.generate_content.return_value = Mock(
            text="Python is used for programming."
        )

        answer, model = gemini_service.generate_answer_for_ui(
            "What is Python?", client=self.client
        )

        self.assertEqual(answer, "Python is used for programming.")
        self.assertEqual(model, "test-flash")

    def test_strips_whitespace_from_answer(self):
        self.client.models.generate_content.return_value = Mock(
            text="  Grounded answer  \n"
        )

        answer, _ = gemini_service.generate_answer_for_ui(
            "Test question", client=self.client
        )

        self.assertEqual(answer, "Grounded answer")

    def test_retries_after_api_error(self):
        self.client.models.generate_content.side_effect = [
            RuntimeError("Temporary API error"),
            Mock(text="Recovered answer"),
        ]

        with patch("src.gemini_service.MAX_MODEL_ATTEMPTS", 2), patch(
            "src.gemini_service.FALLBACK_MODELS",
            ["test-fallback"],
        ):
            answer, model = gemini_service.generate_answer_for_ui(
                "Test question", client=self.client
            )

        self.assertEqual(answer, "Recovered answer")
        self.assertEqual(model, "test-fallback")
        self.assertEqual(
            self.client.models.generate_content.call_count, 2
        )

    def test_raises_error_when_no_models_are_configured(self):
        with patch(
            "src.gemini_service.list_text_generation_models",
            return_value=[],
        ), patch("src.gemini_service.FALLBACK_MODELS", []):
            with self.assertRaisesRegex(
                RuntimeError,
                "No Gemini text-generation models are configured",
            ):
                gemini_service.generate_answer_for_ui(
                    "Test question", client=self.client
                )

        self.client.models.generate_content.assert_not_called()

    def test_raises_error_after_all_models_fail(self):
        self.client.models.generate_content.side_effect = RuntimeError(
            "API unavailable"
        )

        with patch("src.gemini_service.MAX_MODEL_ATTEMPTS", 1):
            with self.assertRaisesRegex(
                RuntimeError,
                "Gemini could not generate an answer",
            ):
                gemini_service.generate_answer_for_ui(
                    "Test question", client=self.client
                )


if __name__ == "__main__":
    unittest.main()
