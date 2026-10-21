
import os
import re
import time
import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "documents" / "Aiml map .pdf"
CHROMA_PATH = BASE_DIR / "chroma_db"
ENV_PATH = BASE_DIR / ".env"

COLLECTION_NAME = "ai_document_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 30

TOP_K = 5
CANDIDATE_K = 20
MAX_RETRIEVAL_DISTANCE = 1.65
SECTION_MATCH_BONUS = 0.25

MAX_MODEL_ATTEMPTS = 4
MODEL_RETRY_DELAY = 1

FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-latest",
    "gemini-2.5-flash",
]

PHASE_HEADINGS = {
    0: "PHASE 0 — Python Foundation",
    1: "PHASE 1 — Data + SQL",
    2: "PHASE 2 — Machine Learning",
    3: "PHASE 3 — Deep Learning",
    4: "PHASE 4 — LLM Engineering",
    5: "PHASE 5 — Prompt Engineering",
    6: "PHASE 6 — RAG",
    7: "PHASE 7 — AI Agents",
    8: "PHASE 8 — Evaluation",
    9: "PHASE 9 — Backend",
    10: "PHASE 10 — Databases",
    11: "PHASE 11 — Docker",
    12: "PHASE 12 — Deployment",
}

PHASE_TOPICS = {
    0: "python foundation",
    1: "data + sql",
    2: "machine learning",
    3: "deep learning",
    4: "llm engineering",
    5: "prompt engineering",
    6: "rag",
    7: "ai agents",
    8: "evaluation",
    9: "backend",
    10: "databases",
    11: "docker",
    12: "deployment",
}

SECTION_HEADING_PATTERN = (
    r"\bPHASE\s+\d+\s*[—-]"
    r"|\bPROFESSIONAL\s+SUMMARY\b\s*:?"
    r"|\bSOFT\s+SKILLS\b\s*:?"
    r"|\bTECHNICAL\s+SKILLS\b\s*:?"
    r"|(?<!SOFT )(?<!TECHNICAL )\bSKILLS\b\s*(?::|↓)"
    r"|\bEDUCATION\b\s*:?"
    r"|\bPROJECTS\b\s*:?"
    r"|\bINTERNSHIPS\s*&?\s*CERTIFICATIONS\b\s*:?"
    r"|\bCERTIFICATIONS\b\s*:?"
    r"|\bLANGUAGES\b\s*:?"
)


# ============================================================
# 2. PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_path):
    """Extract searchable text while preserving one-based PDF pages."""
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages = []

    print(f"\nPDF opened successfully: {pdf_path.name}")
    print(f"Pages: {len(reader.pages)}")

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:
            print(f"Could not extract page {page_number}: {error}")
            text = ""

        pages.append({
            "page": page_number,
            "text": " ".join(text.split()),
        })

    if not any(page["text"] for page in pages):
        raise RuntimeError(
            "No searchable text was extracted. The PDF may need OCR."
        )

    return pages


# ============================================================
# 3. SECTION DETECTION
# ============================================================

def split_into_sections(full_text):
    """Split text at recognized headings."""
    sections = re.split(
        rf"(?={SECTION_HEADING_PATTERN})",
        full_text,
        flags=re.IGNORECASE,
    )

    return [section.strip() for section in sections if section.strip()]


def get_section_name(section):
    """Return the heading associated with a text segment."""
    text = section.strip()

    phase_match = re.match(
        r"(?i)^PHASE\s+(\d+)\s*[—-]",
        text,
    )

    if phase_match:
        number = int(phase_match.group(1))
        return PHASE_HEADINGS.get(number, f"PHASE {number}")

    headings = [
        (r"(?i)^PROFESSIONAL\s+SUMMARY\b", "Professional Summary"),
        (r"(?i)^SOFT\s+SKILLS\b", "Soft Skills"),
        (r"(?i)^TECHNICAL\s+SKILLS\b", "Technical Skills"),
        (r"(?i)^SKILLS\b", "Skills"),
        (r"(?i)^EDUCATION\b", "Education"),
        (r"(?i)^PROJECTS\b", "Projects"),
        (
            r"(?i)^INTERNSHIPS\s*&?\s*CERTIFICATIONS\b",
            "Internships & Certifications",
        ),
        (r"(?i)^CERTIFICATIONS\b", "Certifications"),
        (r"(?i)^LANGUAGES\b", "Languages"),
    ]

    for pattern, name in headings:
        if re.match(pattern, text):
            return name

    first_line = text.splitlines()[0].strip() if text else ""

    if re.search(r"(?i)\bAI\s*/?\s*ML\s+ENGINEER\b", first_line):
        return "Contact Information"

    if re.search(
        r"(?i)(?:\+?\d[\d\s()-]{8,}\d|[\w.+-]+@[\w.-]+\.\w+)",
        first_line,
    ):
        return "Contact Information"

    return "General"


# ============================================================
# 4. SECTION-AWARE CHUNKING
# ============================================================

def create_section_aware_chunks(
    pages,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """Create overlapping chunks and retain their source PDF pages."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    all_words = []
    word_pages = []

    for page in pages:
        page_words = page["text"].split()
        all_words.extend(page_words)
        word_pages.extend([page["page"]] * len(page_words))

    if not all_words:
        raise RuntimeError("The PDF contains no searchable words.")

    sections = split_into_sections(" ".join(all_words))

    chunks = []
    chunk_page_numbers = []
    chunk_section_names = []

    global_word_offset = 0
    step = chunk_size - overlap

    for section in sections:
        section_words = section.split()
        section_name = get_section_name(section)

        for start in range(0, len(section_words), step):
            words = section_words[start:start + chunk_size]

            if not words:
                continue

            word_start = global_word_offset + start
            word_end = word_start + len(words)

            source_pages = sorted(
                set(word_pages[word_start:word_end])
            )

            chunks.append(" ".join(words))
            chunk_page_numbers.append(source_pages)
            chunk_section_names.append(section_name)

        global_word_offset += len(section_words)

    print(f"\nExtracted words: {len(all_words)}")
    print(f"Detected sections: {len(sections)}")
    print(f"Created chunks: {len(chunks)}")
    print("\n--- Detected Sections ---")

    for index, section in enumerate(sections, start=1):
        print(f"{index}. {get_section_name(section)}")
        print(f"   {section[:150]}...")

    return chunks, chunk_page_numbers, chunk_section_names


# ============================================================
# 5. CHROMADB COLLECTION AND INDEXING
# ============================================================

def get_or_create_collection():
    """Open the persistent collection, creating it if necessary."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "l2"},
        )

    return client, collection


def rebuild_vector_collection(
    chunks,
    chunk_page_numbers,
    chunk_section_names,
    embedding_model,
    pdf_path,
):
    """
    Index one PDF without deleting records belonging to other PDFs.

    IDs include a filename-derived prefix so different PDFs can both
    contain a chunk_0 without overwriting each other.
    """
    if not (
        len(chunks)
        == len(chunk_page_numbers)
        == len(chunk_section_names)
    ):
        raise ValueError("Chunk data lists must have equal lengths.")

    if not chunks:
        raise ValueError("Cannot index an empty list of chunks.")

    pdf_path = Path(pdf_path)
    source_name = pdf_path.name

    source_prefix = hashlib.sha1(
        source_name.encode("utf-8")
    ).hexdigest()[:10]

    print(f"\nGenerating embeddings for {source_name}...")

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True,
    )

    _, collection = get_or_create_collection()

    # Delete only the old records for this PDF, not every indexed PDF.
    try:
        existing = collection.get(
            where={"source": source_name},
            include=["metadatas"],
        )
        existing_ids = existing.get("ids", [])

        if existing_ids:
            collection.delete(ids=existing_ids)
            print(
                f"Removed {len(existing_ids)} old chunks "
                f"for {source_name}."
            )
    except Exception as error:
        raise RuntimeError(
            f"Could not replace existing records for {source_name}: {error}"
        ) from error

    ids = [
        f"{source_prefix}_chunk_{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "source": source_name,
            "source_pages": ",".join(map(str, pages)),
            "section_name": section,
        }
        for pages, section in zip(
            chunk_page_numbers,
            chunk_section_names,
        )
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} chunks for {source_name}.")
    print(f"Total chunks in shared collection: {collection.count()}")

    return collection, embeddings


# ============================================================
# 6. TOPIC AND PROJECT DETECTION
# ============================================================

def is_broad_resume_question(question):
    """Identify questions asking for a whole-resume overview."""
    q = question.lower()

    return bool(
        re.search(
            r"\b(?:comprehensive|complete|full|entire|overall)\b.{0,40}\bresume\b"
            r"|\bresume\s+(?:summary|overview|breakdown)\b"
            r"|\ball\s+(?:the\s+)?resume\s+sections\b",
            q,
        )
    )



def get_requested_source(question, available_sources):
    """Identify a specific PDF when the question names one document."""
    q = question.lower()

    resume_terms = (
        "my resume",
        "my cv",
        "in the resume",
        "in my cv",
        "from the resume",
        "from my resume",
    )

    roadmap_terms = (
        "the roadmap",
        "my roadmap",
        "in the roadmap",
        "from the roadmap",
        "ai/ml roadmap",
        "ai/ml roadmap",
    )

    requested_type = None

    if any(term in q for term in resume_terms):
        requested_type = "resume"
    elif any(term in q for term in roadmap_terms):
        requested_type = "roadmap"

    if requested_type is None:
        return None

    for source in available_sources:
        filename = source.lower()

        if requested_type == "resume" and (
            "resume" in filename or "cv" in filename
        ):
            return source

        if requested_type == "roadmap" and (
            "roadmap" in filename or "aiml map" in filename
        ):
            return source

    return None


def is_cross_document_question(question):
    """Identify questions that need evidence from multiple documents."""
    q = question.lower()

    explicit_comparison = re.search(
        r"\b(compare|comparison|contrast|both documents|across documents|"
        r"cross[- ]document|which of those|"       r"resume and (?:the )?roadmap|roadmap and (?:my )?resume|"
        r"resume versus roadmap|roadmap versus resume)\b",
        q,
    )

    mentions_both = (
        re.search(r"\bresume\b", q)
        and re.search(r"\broadmap\b", q)
    )

    return bool(explicit_comparison or mentions_both)


def get_topic_aliases(question):
    """Detect resume sections, project questions, and roadmap topics."""
    q = question.lower()
    aliases = []

    topic_rules = [
        (r"\bprojects?\b", "projects"),
        (
            r"\bnlp[-\s]+based\s+intelligent\s+text\s+"
            r"classification\s+system\b"
            r"|\bintelligent\s+text\s+classification\s+system\b"
            r"|\btext\s+classification\s+system\b",
            "projects",
        ),
        (
            r"\bai[-\s]+powered\s+document\s+intelligence\b"
            r"|\bdocument\s+intelligence\s+(?:and|&)\s*question\s+answering\b"
            r"|\bquestion\s+answering\s+system\b",
            "projects",
        ),
        (
            r"\bend[-\s]+to[-\s]+end\s+(?:machine\s+learning|ml)\s+prediction\s+system\b"
            r"|\bprediction\s+system\b",
            "projects",
        ),
        (
            r"\btext\s+classification\b|\bnlp\s+(?:text\s+)?classification\s+model\b"
            r"|\bclassification\s+model\b|\baccuracy\s+percentage\b"
            r"|\baccuracy\s+(?:of|for)\s+(?:my\s+)?"
            r"(?:nlp|text\s+classification|classification\s+model)\b",
            "projects",
        ),
        (
            r"\btechnical\s+skills?\b|\btech(?:nical)?\s+stack\b"
            r"|\btechnologies\b|\btech\s+stack\b",
            "technical skills",
        ),
        (r"\beducation(?:al)?\b|\bcollege\b|\bdegree\b", "education"),
        (r"\bcertifications?\b|\bcertificates?\b", "certifications"),
        (r"\binternships?\b", "internships"),
        (r"\bsoft\s+skills?\b", "soft skills"),
        (
            r"\bprofessional\s+summary\b|\bcareer\s+(?:summary|objective)\b",
            "professional summary",
        ),
        (r"\blanguages?\b", "languages"),
        (
            r"\bcontact\s+information\b|\bemail address\b|\bphone number\b",
            "contact information",
        ),
        (r"\bmachine learning\b|\bml\b", "machine learning"),
        (r"\bai agents?\b|\bagents?\b|\bagentic\b", "ai agents"),
        (r"\brag\b|retrieval[- ]augmented", "rag"),
        (
            r"\bvector databases?\b|\bvector db\b|vector storage|similarity search",
            "vector database",
        ),
        (
            r"\bembeddings?\b|\bchunking\b|\breranking\b|"
            r"context construction|citation/grounding",
            "rag",
        ),
        (r"\bdeploy(?:ment)?\b|\bcloud\b|\bhosting\b", "deployment"),
        (r"\bpython\b|programming foundation", "python foundation"),
        (r"\bsql\b|postgresql", "data + sql"),
        (r"deep learning|neural networks?|\bpytorch\b", "deep learning"),
        (r"\bnlp\b|natural language processing", "deep learning"),
        (r"\bllm\b|large language models?", "llm engineering"),
        (r"prompt engineering|\bprompting\b", "prompt engineering"),
        (r"\bevaluation\b|\bevaluate\b|\bmetrics\b", "evaluation"),
        (r"\bfastapi\b|\bbackend\b|\bapi\b", "backend"),
        (r"\bdocker\b|\bcontainers?\b", "docker"),
        (r"\bdatabases?\b|\bpostgresql\b", "databases"),
    ]

    for pattern, topic in topic_rules:
        if re.search(pattern, q) and topic not in aliases:
            aliases.append(topic)

    phase_match = re.search(r"\bphase\s+(\d+)\b", q)

    if phase_match:
        number = int(phase_match.group(1))
        topic = PHASE_TOPICS.get(number)

        if topic and topic not in aliases:
            aliases.append(topic)

    if is_broad_resume_question(question):
        for topic in (
            "education",
            "technical skills",
            "projects",
            "certifications",
            "internships",
            "soft skills",
            "professional summary",
            "languages",
            "contact information",
        ):
            if topic not in aliases:
                aliases.append(topic)

    return aliases


# ============================================================
# 7. SECTION MATCHING
# ============================================================

def section_match(section_name, aliases):
    """Match section metadata against detected topic aliases."""
    heading = re.sub(
        r"[^a-z0-9+ ]+",
        " ",
        (section_name or "").lower(),
    )
    heading = " ".join(heading.split())

    resume_sections = {
        "projects": {"projects"},
        "technical skills": {"technical skills", "skills"},
        "education": {"education"},
        "certifications": {
            "certifications",
            "internships certifications",
        },
        "internships": {
            "internships",
            "internships certifications",
        },
        "soft skills": {"soft skills"},
        "professional summary": {"professional summary"},
        "languages": {"languages"},
        "contact information": {"contact information"},
    }

    roadmap_sections = {
        "python foundation": {"python foundation", "python"},
        "data + sql": {"data + sql", "sql"},
        "machine learning": {"machine learning"},
        "deep learning": {"deep learning"},
        "llm engineering": {"llm engineering", "llm"},
        "prompt engineering": {"prompt engineering"},
        "rag": {"rag"},
        "ai agents": {"ai agents"},
        "evaluation": {"evaluation"},
        "backend": {"backend"},
        "databases": {"databases", "database"},
        "vector database": {"databases", "database"},
        "docker": {"docker"},
        "deployment": {"deployment"},
    }

    for alias in aliases:
        if alias in resume_sections:
            if heading in resume_sections[alias]:
                return True
            continue

        terms = roadmap_sections.get(alias)

        if terms and any(term in heading for term in terms):
            return True

    return False


def get_target_sections(aliases):
    """Map aliases to exact section metadata values."""
    mapping = {
        "projects": ["Projects"],
        "technical skills": ["Technical Skills", "Skills"],
        "education": ["Education"],
        "certifications": [
            "Certifications",
            "Internships & Certifications",
        ],
        "internships": ["Internships & Certifications"],
        "soft skills": ["Soft Skills"],
        "professional summary": ["Professional Summary"],
        "languages": ["Languages"],
        "contact information": ["Contact Information"],
        "python foundation": [PHASE_HEADINGS[0]],
        "data + sql": [PHASE_HEADINGS[1]],
        "machine learning": [PHASE_HEADINGS[2]],
        "deep learning": [PHASE_HEADINGS[3]],
        "llm engineering": [PHASE_HEADINGS[4]],
        "prompt engineering": [PHASE_HEADINGS[5]],
        "rag": [PHASE_HEADINGS[6]],
        "ai agents": [PHASE_HEADINGS[7]],
        "evaluation": [PHASE_HEADINGS[8]],
        "backend": [PHASE_HEADINGS[9]],
        "databases": [PHASE_HEADINGS[10]],
        "vector database": [PHASE_HEADINGS[10]],
        "docker": [PHASE_HEADINGS[11]],
        "deployment": [PHASE_HEADINGS[12]],
    }

    sections = []

    for alias in aliases:
        for section in mapping.get(alias, []):
            if section not in sections:
                sections.append(section)

    return sections


# ============================================================
# 8. FILTERING AND PRIORITIZATION
# ============================================================

def filter_by_distance(
    candidates,
    max_distance=MAX_RETRIEVAL_DISTANCE,
):
    """Keep candidates within the configured distance threshold."""
    return [
        item for item in candidates
        if item["distance"] <= max_distance
    ]


def prioritize_matching_sections(eligible, aliases, question=""):
    """
    Prefer requested sections while preserving cross-document coverage.
    Project questions focus on Projects unless another section is requested.
    """
    if not aliases:
        return eligible, False

    matching = [
        item for item in eligible
        if item.get("matched_section", False)
    ]

    if not matching:
        return eligible, False

    q = question.lower()

    asks_for_other_section = bool(
        re.search(
            r"\b(?:technical\s+skills?|skills?|technologies|technology|"
            r"tech(?:nical)?\s+stack|education|college|degree|"
            r"certifications?|certificates?|internships?|soft\s+skills?|"
            r"professional\s+summary|career\s+summary|languages?|"
            r"contact\s+information|email\s+address|phone\s+number)\b",
            q,
        )
    )

    asks_for_projects = (
        "projects" in aliases
        or bool(re.search(r"\b(?:project|projects)\b", q))
    )

    if asks_for_projects and not asks_for_other_section:
        project_chunks = [
            item for item in matching
            if item.get("metadata", {})
            .get("section_name", "")
            .strip()
            .lower() == "projects"
        ]

        if project_chunks:
            return project_chunks, True

    return matching, True


# ============================================================
# 9. RETRIEVAL
# ============================================================

def retrieve_chunks(
    collection,
    embedding_model,
    question,
    total_chunks,
):
    """
    Retrieve relevant chunks and prioritize source coverage for comparisons.

    Return format is compatible with src/app.py:
    ids, texts, distances, metadatas.
    """
    if total_chunks <= 0:
        print("The collection contains no searchable chunks.")
        return [], [], [], []

    query_embedding = embedding_model.encode(
        [question]
    ).tolist()

    
    aliases = get_topic_aliases(question)
    target_sections = get_target_sections(aliases)
    cross_document = is_cross_document_question(question)

    print("\n--- Retrieval Diagnostics ---")
    print(f"Question: {question}")
    print(f"Detected aliases: {aliases}")
    print(f"Target sections: {target_sections}")
    print(f"Cross-document comparison: {cross_document}")

    candidates_by_id = {}

    def add_query_results(results):
        """Merge nested Chroma query results by unique chunk ID."""
        if not results:
            return

        ids_groups = results.get("ids") or []
        if not ids_groups or not ids_groups[0]:
            return

        ids = ids_groups[0]
        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        for chunk_id, document, distance, metadata in zip(
            ids,
            documents,
            distances,
            metadatas,
        ):
            metadata = metadata or {}
            distance = float(distance)

            candidate = {
                "id": chunk_id,
                "text": document or "",
                "distance": distance,
                "metadata": metadata,
            }

            previous = candidates_by_id.get(chunk_id)

            if previous is None or distance < previous["distance"]:
                candidates_by_id[chunk_id] = candidate

    candidate_count = max(1, min(CANDIDATE_K, total_chunks))

    # A. Global semantic retrieval.
    add_query_results(
        collection.query(
            query_embeddings=query_embedding,
            n_results=candidate_count,
            include=["documents", "distances", "metadatas"],
        )
    )

    # B. For comparisons, independently retrieve from each indexed source.
    # This prevents a larger PDF from crowding the other PDF out.
    
    source_names = set()

    try:
        all_metadata = collection.get(include=["metadatas"])

        for metadata in all_metadata.get("metadatas") or []:
            if metadata and metadata.get("source"):
                source_names.add(metadata["source"])

    except Exception as error:
        print(f"Could not enumerate PDF sources: {error}")

    requested_source = get_requested_source(
        question,
        source_names,
    )

    print(f"Requested source: {requested_source}")

    if cross_document:
        try:
            all_metadata = collection.get(include=["metadatas"])

            for metadata in all_metadata.get("metadatas") or []:
                if metadata and metadata.get("source"):
                    source_names.add(metadata["source"])

        except Exception as error:
            print(f"Could not enumerate PDF sources: {error}")

        print(f"Indexed sources: {sorted(source_names)}")

        if len(source_names) > 1:
            print("\n--- Per-Source Retrieval ---")

            for source_name in sorted(source_names):
                try:
                    source_records = collection.get(
                        where={"source": source_name},
                        include=["metadatas"],
                    )
                    source_count = len(source_records.get("ids", []))

                    if not source_count:
                        continue

                    results = collection.query(
                        query_embeddings=query_embedding,
                        n_results=min(candidate_count, source_count),
                        where={"source": source_name},
                        include=["documents", "distances", "metadatas"],
                    )

                    before = set(candidates_by_id)
                    add_query_results(results)
                    added = len(set(candidates_by_id) - before)

                    print(
                        f"Source: {source_name} | "
                        f"Chunks: {source_count} | "
                        f"New candidates: {added}"
                    )

                except Exception as error:
                    print(
                        f"Source query failed for {source_name}: "
                        f"{type(error).__name__}: {error}"
                    )

    # C. Query every requested section explicitly.
    print("\n--- Targeted Section Retrieval ---")

    for section_name in target_sections:
        try:
            section_records = collection.get(
                where={"section_name": section_name},
                include=["metadatas"],
            )

            # ChromaDB count(where=...) is not portable across versions.
            section_count = len(section_records.get("ids", []))

            if section_count == 0:
                print(
                    f"Section: {section_name} | "
                    "No chunks found with this exact section name."
                )
                continue

            results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(candidate_count, section_count),
                where={"section_name": section_name},
                include=["documents", "distances", "metadatas"],
            )

            before = set(candidates_by_id)
            add_query_results(results)
            added = len(set(candidates_by_id) - before)

            print(
                f"Section: {section_name} | New candidates: {added}"
            )

        except Exception as error:
            print(
                f"Section query failed for '{section_name}': "
                f"{type(error).__name__}: {error}"
            )

    # D. Score and annotate every unique candidate.
    ranked = []

    for item in candidates_by_id.values():
        section_name = item["metadata"].get("section_name", "")
        item["matched_section"] = section_match(section_name, aliases)
        item["rerank_score"] = (
            item["distance"]
            - (
                SECTION_MATCH_BONUS
                if item["matched_section"]
                else 0.0
            )
        )
        ranked.append(item)

    if not ranked:
        print("No chunks were retrieved.")
        return [], [], [], []

    
    eligible = filter_by_distance(ranked)

    # Keep single-document questions restricted to the requested PDF.
    # Cross-document comparisons must still use evidence from both PDFs.
    if requested_source and not cross_document:
        eligible = [
            item for item in eligible
            if item["metadata"].get("source") == requested_source
        ]

        print(f"Filtering results to source: {requested_source}")
        print(f"Chunks remaining after source filter: {len(eligible)}")

    print("\n--- Retrieval Summary ---")

    print(f"Unique candidates examined: {len(ranked)}")
    print(f"Chunks within threshold: {len(eligible)}")
    print(f"Maximum allowed distance: {MAX_RETRIEVAL_DISTANCE:.4f}")

    if not eligible:
        print("No chunks passed the relevance threshold.")
        return [], [], [], []

    # E. Apply section matching.
    if cross_document:
        topic_filter_applied = False
        print(
            "Cross-document comparison: preserving candidates "
            "from all document sources."
        )
    else:
       eligible, topic_filter_applied = prioritize_matching_sections(
          eligible,
          aliases,
          question,
        )

    if topic_filter_applied:
        print(
            "Matching-section prioritization enabled. "
            f"Matching chunks: {len(eligible)}"
        )
    elif aliases:
        print("No matching section passed the threshold; using semantic fallback.")
    else:
        print("Using semantic relevance without section filtering.")

    eligible.sort(
        key=lambda item: (
            item["rerank_score"],
            item["distance"],
        )
    )

    # F. Select final context, guaranteeing source diversity for comparisons.
    selected = []
    selected_ids = set()

    if cross_document and len(source_names) > 1:
        for source_name in sorted(source_names):
            source_candidates = [
                item for item in eligible
                if item["metadata"].get("source") == source_name
            ]

            if source_candidates:
                best = source_candidates[0]

                if best["id"] not in selected_ids:
                    selected.append(best)
                    selected_ids.add(best["id"])

        # Fill the remaining slots with the best-ranked distinct chunks.
        for item in eligible:
            if len(selected) >= TOP_K:
                break

            if item["id"] not in selected_ids:
                selected.append(item)
                selected_ids.add(item["id"])

        # If the number of sources exceeds TOP_K, preserve the best overall.
        selected.sort(
            key=lambda item: (
                item["rerank_score"],
                item["distance"],
            )
        )
        selected = selected[:TOP_K]

    elif is_broad_resume_question(question):
        selection_limit = min(10, max(TOP_K, len(target_sections)))
        seen_sections = set()

        for item in eligible:
            section = (
                item["metadata"].get("section_name", "unknown")
                .strip()
                .lower()
            )

            if section not in seen_sections:
                selected.append(item)
                selected_ids.add(item["id"])
                seen_sections.add(section)

            if len(selected) >= selection_limit:
                break

        if len(selected) < selection_limit:
            for item in eligible:
                if item["id"] not in selected_ids:
                    selected.append(item)
                    selected_ids.add(item["id"])

                if len(selected) >= selection_limit:
                    break

    else:
        selected = eligible[:TOP_K]

    # For ordinary questions, avoid sending multiple chunks
    # from the same section when other sections are available.
    if not cross_document and not is_broad_resume_question(question):
        deduplicated = []
        seen_sections = set()

        for item in selected:
            section = (
                item["metadata"].get("section_name", "unknown")
                .strip()
                .lower()
            )

            if section not in seen_sections:
                deduplicated.append(item)
                seen_sections.add(section)

        selected = deduplicated
    print(f"\nChunks passed to Gemini: {len(selected)}")

    for item in selected:
        metadata = item["metadata"]

        print(
            f"\nChunk: {item['id']} | "
            f"Source: {metadata.get('source', 'unknown')} | "
            f"Distance: {item['distance']:.4f} | "
            f"Rerank: {item['rerank_score']:.4f} | "
            f"Section: {metadata.get('section_name', 'unknown')} | "
            f"Pages: {metadata.get('source_pages', 'unknown')}"
        )
        print(f"Preview: {item['text'][:350]}...")

    return (
        [item["id"] for item in selected],
        [item["text"] for item in selected],
        [item["distance"] for item in selected],
        [item["metadata"] for item in selected],
    )


# ============================================================
# 10. GROUNDED PROMPT
# ============================================================

def build_grounded_prompt(
    question,
    retrieved_ids,
    retrieved_texts,
    retrieved_metadatas,
):
    """Build a grounded prompt with source filenames and PDF pages."""
    if not (
        len(retrieved_ids)
        == len(retrieved_texts)
        == len(retrieved_metadatas)
    ):
        raise ValueError("Retrieved data lists must have equal lengths.")

    if not retrieved_texts:
        raise ValueError("Cannot build a prompt without retrieved context.")

    context_parts = []

    for chunk_id, chunk_text, metadata in zip(
        retrieved_ids,
        retrieved_texts,
        retrieved_metadatas,
    ):
        metadata = metadata or {}

        context_parts.append(
            f"Source chunk: {chunk_id}\n"
            f"PDF: {metadata.get('source', 'unknown')}\n"
            f"Section: {metadata.get('section_name', 'unknown')}\n"
            f"PDF page(s): {metadata.get('source_pages', 'unknown')}\n"
            f"Text:\n{chunk_text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a careful document question-answering assistant.

Answer the user's question using only the supplied document context.

GROUNDING RULES:
- Treat the supplied context as the only source of facts.
- Never invent facts, technologies, project names, metrics, or citations.
- If the context does not support an answer, say so clearly.
- If only part of the question is answerable, answer that part and
  explain what information is missing.
- Distinguish an explicit statement from a reasonable inference.
- Do not assume that two projects are the same just because their topics
  are similar.
- When comparing documents, identify which PDF supports each claim.
- Distinguish technologies explicitly associated with a project from
  technologies merely listed elsewhere.
- Do not claim that a model achieved an accuracy percentage unless
  the percentage is explicitly stated in the context.
- If a requested document is not represented in the context, disclose
  that limitation instead of pretending to have searched it.

COMPARISON RULES:
- For questions comparing a resume and roadmap, inspect evidence from
  both sources when both are present in the context.
- Separate projects explicitly mentioned in the resume from projects
  suggested or listed in the roadmap.
- Identify overlapping project areas only when the supplied text supports
  the overlap.
- Do not claim that a roadmap project was completed for the resume unless
  the resume supports that conclusion.
- If the documents differ, describe the difference rather than forcing
  a match.

CITATION RULES:
- Cite every important document-based claim.
- Use the PDF filename and one-based PDF page numbers provided in metadata.
- Format citations like [Aiml map .pdf, PDF page 17] or
  [WoohasrResume.pdf, PDF pages 2, 3].
- Use the exact source filename provided in the context.
- Never invent filenames or page numbers.
- Never invent page numbers.
- Never invent filenames.

WRITING RULES:
- Answer the actual question directly.
- Use concise bullets or a table when useful.
- Preserve project names and document terminology.
- Do not repeat the same point unnecessarily.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    return prompt.strip()


# ============================================================
# 11. GEMINI MODEL DISCOVERY
# ============================================================

def list_text_generation_models(gemini_client):
    """Find API-listed Flash models that support text generation."""
    print("\n--- Checking Available Gemini Models ---")

    try:
        available_models = list(gemini_client.models.list())
    except Exception as error:
        print(
            "Could not list models: "
            f"{type(error).__name__}: {error}"
        )
        print("Configured fallback models will be tried.")
        return []

    candidates = []

    for model in available_models:
        model_name = getattr(model, "name", None)
        actions = getattr(model, "supported_actions", None) or []

        if not model_name or "generateContent" not in actions:
            continue

        normalized = model_name.removeprefix("models/")

        if (
            "flash" in normalized.lower()
            and normalized not in candidates
        ):
            candidates.append(normalized)

    ordered = []

    for preferred in FALLBACK_MODELS:
        if preferred in candidates and preferred not in ordered:
            ordered.append(preferred)

    for candidate in sorted(candidates):
        if candidate not in ordered:
            ordered.append(candidate)

    for model_name in ordered:
        print(f"  - {model_name}")

    if not ordered:
        print(
            "No API-listed Flash models found; "
            "using configured fallback models."
        )

    return ordered


# ============================================================
# 12. GEMINI ANSWER GENERATION
# ============================================================

def generate_answer(gemini_client, prompt):
    """Try available models and configured fallbacks."""
    available_models = list_text_generation_models(gemini_client)

    models_to_try = []

    for model_name in available_models + FALLBACK_MODELS:
        if model_name not in models_to_try:
            models_to_try.append(model_name)

    models_to_try = models_to_try[:MAX_MODEL_ATTEMPTS]

    if not models_to_try:
        print("No candidate Gemini models are configured.")
        return False

    last_error = None

    for index, model_name in enumerate(models_to_try):
        try:
            print(f"\nTrying Gemini model: {model_name}")

            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1200,
                ),
            )

            answer = getattr(response, "text", None)

            if answer and answer.strip():
                print(f"\n--- Answer from {model_name} ---")
                print(answer.strip())
                return True

            last_error = RuntimeError(
                f"{model_name} returned no text."
            )
            print(last_error)

        except Exception as error:
            last_error = error
            print(
                f"Request failed for {model_name}: "
                f"{type(error).__name__}: {error}"
            )

        if index < len(models_to_try) - 1:
            time.sleep(MODEL_RETRY_DELAY)

    print("\nGemini could not generate an answer.")
    print(
        "Review the errors above for model access, quota, "
        "or service issues."
    )
    print("Keep your API key private.")

    if last_error:
        print(
            f"Last error: {type(last_error).__name__}: {last_error}"
        )

    return False


# ============================================================
# 13. STANDALONE MAIN PROGRAM
# ============================================================

def main():
    """
    Standalone command-line entry point.

    This entry point indexes the configured roadmap PDF. The Streamlit
    application must index uploaded PDFs separately to enable comparisons
    across the resume and roadmap.
    """
    load_dotenv(ENV_PATH)

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("ERROR: GEMINI_API_KEY was not found in your .env file.")
        print("Keep your API key private.")
        return

    if not PDF_PATH.is_file():
        print(f"ERROR: PDF file not found: {PDF_PATH}")
        print("Check the documents folder and exact PDF filename.")
        return

    try:
        gemini_client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=60000),
        )

        pages = extract_pdf_pages(PDF_PATH)

        chunks, page_numbers, section_names = (
            create_section_aware_chunks(pages)
        )

        print("\nLoading embedding model...")

        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        collection, embeddings = rebuild_vector_collection(
            chunks,
            page_numbers,
            section_names,
            embedding_model,
            PDF_PATH,
        )

        print(f"\nEmbedding count: {len(embeddings)}")
        print(
            f"Chunk size: {CHUNK_SIZE}; "
            f"overlap: {CHUNK_OVERLAP}"
        )
        print(
            f"Retrieval: up to {CANDIDATE_K} candidates; "
            f"up to {TOP_K} chunks passed to Gemini"
        )

        while True:
            question = input(
                "\nAsk a question about your indexed PDFs "
                "(or type 'exit' to quit): "
            ).strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            if not question:
                print("Please enter a question.")
                continue

            (
                retrieved_ids,
                retrieved_texts,
                distances,
                metadatas,
            ) = retrieve_chunks(
                collection,
                embedding_model,
                question,
                collection.count(),
            )

            if not retrieved_texts:
                print(
                    "\nNo answer was generated because no relevant "
                    "document context passed the retrieval filter."
                )
                continue

            prompt = build_grounded_prompt(
                question,
                retrieved_ids,
                retrieved_texts,
                metadatas,
            )

            print("\nGenerating an answer from the retrieved context...")
            generate_answer(gemini_client, prompt)

    except KeyboardInterrupt:
        print("\nProgram cancelled by user.")

    except Exception as error:
        print(
            f"\nERROR: The project could not complete: "
            f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    main()
