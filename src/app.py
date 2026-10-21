
import hashlib
from io import BytesIO

import chromadb
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from src.main import (
    BASE_DIR,
    EMBEDDING_MODEL_NAME,
    create_section_aware_chunks,
    retrieve_chunks,
    build_grounded_prompt,
)
from src.gemini_service import generate_answer_for_ui


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CHROMA_PATH = BASE_DIR / "uploaded_chroma_db"
COLLECTION_NAME = "documind_multi_pdf_v1"


# ============================================================
# DARK BLUE AND BLACK THEME — NO ANIMATIONS
# ============================================================

st.markdown(
    """
    <style>
        #MainMenu, footer, header {
            visibility: hidden;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        .stApp {
            background: #080d18;
            color: #f1f5f9;
        }

        .block-container {
            max-width: 1050px;
            padding: 2rem 2rem 3rem;
        }

        h1, h2, h3, p, label {
            color: #f1f5f9;
        }

        .brand {
            font-size: 1.15rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 2.5rem;
        }

        .brand span {
            color: #60a5fa;
        }

        .hero {
            background: #101a2d;
            border: 1px solid #233653;
            border-radius: 16px;
            padding: 2.3rem;
            margin-bottom: 2rem;
        }

        .eyebrow {
            color: #60a5fa;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 2px;
        }

        .hero-title {
            color: #f8fafc;
            font-size: 2.5rem;
            font-weight: 750;
            line-height: 1.2;
            margin: 0.8rem 0;
        }

        .hero-copy {
            color: #b6c4d8;
            font-size: 1rem;
            line-height: 1.8;
            max-width: 680px;
        }

        .section-title {
            color: #f1f5f9;
            font-size: 1.25rem;
            font-weight: 650;
            margin: 1.8rem 0 0.4rem;
        }

        .muted {
            color: #9aabc2;
            font-size: 0.9rem;
        }

        [data-testid="stFileUploader"] {
            background: #101a2d;
            border: 1px solid #2a3d59;
            border-radius: 12px;
            padding: 1rem;
        }

        [data-testid="stFileUploader"] section {
            background: #101a2d;
        }

        [data-testid="stFileUploader"] button {
            background: #1d4ed8;
            color: white;
            border: none;
            border-radius: 7px;
        }

        .stTextInput input,
        .stTextArea textarea {
            background: #0d1626;
            color: #f8fafc;
            border: 1px solid #30435f;
            border-radius: 9px;
        }

        .stButton button,
        .stFormSubmitButton button {
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.55rem 1.2rem;
            font-weight: 600;
        }

        .stButton button:hover,
        .stFormSubmitButton button:hover {
            background: #1d4ed8;
            color: white;
            border: none;
        }

        [data-testid="stMetric"] {
            background: #101a2d;
            border: 1px solid #263954;
            border-radius: 10px;
            padding: 1rem;
        }

        [data-testid="stMetricLabel"] {
            color: #a8bad1;
        }

        [data-testid="stMetricValue"] {
            color: #f8fafc;
        }

        [data-testid="stAlert"] {
            background: #101a2d;
            color: #e2e8f0;
        }

        [data-testid="stExpander"] {
            background: #101a2d;
            border: 1px solid #263954;
            border-radius: 9px;
        }

        hr {
            border-color: #263954;
        }

        .footer {
            text-align: center;
            color: #8293aa;
            font-size: 0.8rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #263954;
        }

        @media (max-width: 650px) {
            .block-container {
                padding: 1rem 1rem 2rem;
            }

            .hero {
                padding: 1.3rem;
            }

            .hero-title {
                font-size: 1.9rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD RESOURCES
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_uploaded_pdf(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())

        pages.append({
            "page": page_number,
            "text": text,
        })

    if not any(page["text"] for page in pages):
        raise ValueError(
            "No searchable text was found in this PDF. "
            "The file may contain scanned pages."
        )

    return pages


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="brand">DocuMind <span>AI</span></div>

    <div class="hero">
        <div class="eyebrow">AI-POWERED DOCUMENT INTELLIGENCE</div>
        <div class="hero-title">
            Your documents.<br>Clear answers.
        </div>
        <div class="hero-copy">
            Upload multiple PDFs, ask questions in natural language,
            and explore answers grounded in your documents.
            Find the information you need without reading
            every page manually.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MULTIPLE DOCUMENT UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">01 &nbsp; Upload your documents</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="muted">Choose one or more PDFs with selectable text.</p>',
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDFs to ask questions across your documents.",
)


if not uploaded_files:
    st.info("Upload one or more PDFs to start exploring your documents.")

    st.markdown(
        """
        <div class="footer">
            DocuMind AI · Document intelligence powered by AI
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# READ AND PREPARE ALL UPLOADED DOCUMENTS
# ============================================================

all_chunks = []
all_metadatas = []
document_details = []
total_pages = 0

# A signature changes when the selected files or their contents change.
document_signature_parts = []

try:
    with st.spinner("Reading and preparing your PDFs..."):

        for uploaded_file in uploaded_files:
            pdf_bytes = uploaded_file.getvalue()

            file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]

            document_signature_parts.append(
                f"{uploaded_file.name}:{file_hash}"
            )

            pages = extract_uploaded_pdf(pdf_bytes)

            chunks, chunk_pages, chunk_sections = (
                create_section_aware_chunks(pages)
            )

            if not chunks:
                st.warning(
                    f"No usable text chunks were created for "
                    f"{uploaded_file.name}. This file will be skipped."
                )
                continue

            document_details.append({
                "name": uploaded_file.name,
                "hash": file_hash,
                "size": len(pdf_bytes),
                "pages": len(pages),
                "chunks": len(chunks),
            })

            total_pages += len(pages)

            for index, chunk in enumerate(chunks):
                all_chunks.append(chunk)

                # Each chunk retains its original document and page.
                all_metadatas.append({
                    "source": uploaded_file.name,
                    "file_hash": file_hash,
                    "source_pages": ",".join(
                        str(page) for page in chunk_pages[index]
                    ),
                    "section_name": (
                        chunk_sections[index] or "Document"
                    ),
                })

except Exception as error:
    st.error("We couldn't process one of your PDFs.")

    with st.expander("Technical details"):
        st.code(f"{type(error).__name__}: {error}")

    st.stop()


if not all_chunks:
    st.error(
        "No searchable text chunks were created. "
        "Please try another PDF with selectable text."
    )
    st.stop()


document_signature = hashlib.sha256(
    "|".join(document_signature_parts).encode("utf-8")
).hexdigest()


# ============================================================
# INDEX ALL DOCUMENTS IN ONE COLLECTION
# ============================================================

try:
    with st.spinner("Preparing your documents for search..."):

        embedding_model = load_embedding_model()
        client = load_chroma_client()

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "l2"},
        )

        # Rebuild the collection only when the selected document set changes.
        if (
            st.session_state.get("indexed_document_signature")
            != document_signature
        ):
            embeddings = embedding_model.encode(
                all_chunks,
                show_progress_bar=False,
            )

            existing_records = collection.get()

            if existing_records["ids"]:
                collection.delete(ids=existing_records["ids"])

            chunk_ids = [
                f"{metadata['file_hash']}_{index}"
                for index, metadata in enumerate(all_metadatas)
            ]

            collection.upsert(
                ids=chunk_ids,
                documents=all_chunks,
                embeddings=embeddings.tolist(),
                metadatas=all_metadatas,
            )

            st.session_state["indexed_document_signature"] = (
                document_signature
            )

            # Avoid displaying an answer from the previous document set.
            st.session_state.pop("last_answer", None)
            st.session_state.pop("last_question", None)
            st.session_state.pop("last_evidence", None)
            st.session_state.pop("last_model", None)

except Exception as error:
    st.error("We couldn't index your documents.")

    with st.expander("Technical details"):
        st.code(f"{type(error).__name__}: {error}")

    st.stop()


# ============================================================
# DOCUMENT SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">02 &nbsp; Your documents</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("PDF files", len(document_details))

with col2:
    st.metric("Total pages", total_pages)

with col3:
    st.metric("Text chunks", len(all_chunks))

with col4:
    total_size = sum(
        document["size"] for document in document_details
    )
    st.metric("Combined size", f"{total_size / 1024:.1f} KB")


st.success(
    f"Ready to explore {len(document_details)} document(s)."
)

with st.expander("View uploaded documents"):
    for document in document_details:
        st.markdown(f"**{document['name']}**")
        st.caption(
            f"{document['pages']} page(s) · "
            f"{document['chunks']} text chunks · "
            f"{document['size'] / 1024:.1f} KB"
        )


# ============================================================
# ASK QUESTIONS ACROSS ALL DOCUMENTS
# ============================================================

st.markdown(
    '<div class="section-title">03 &nbsp; Ask your documents</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="muted">'
    'Ask a question about one document or information across your PDFs.'
    '</p>',
    unsafe_allow_html=True,
)

with st.form("question_form"):
    question = st.text_input(
        "Your question",
        placeholder=(
            "e.g. Compare the key information across my documents."
        ),
    )

    ask_button = st.form_submit_button(
        "Ask Question",
        use_container_width=True,
    )


if ask_button:
    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        # Do not leave an old answer visible if this request fails.
        st.session_state.pop("last_answer", None)
        st.session_state.pop("last_evidence", None)
        st.session_state.pop("last_model", None)

        try:
            with st.spinner("Finding relevant passages..."):

                ids, texts, distances, metadatas = retrieve_chunks(
                    collection,
                    embedding_model,
                    question.strip(),
                    collection.count(),
                )

            if not texts:
                st.warning(
                    "I couldn't find sufficiently relevant passages "
                    "in your uploaded documents. Try asking differently."
                )

            else:
                prompt = build_grounded_prompt(
                    question.strip(),
                    ids,
                    texts,
                    metadatas,
                )

                with st.spinner("Generating your answer..."):
                    answer, model_name = generate_answer_for_ui(prompt)

                st.session_state["last_question"] = question.strip()
                st.session_state["last_answer"] = answer
                st.session_state["last_evidence"] = list(
                    zip(texts, metadatas, distances)
                )
                st.session_state["last_model"] = model_name

        except Exception as error:
            st.error("Unable to generate an answer.")

            with st.expander("Technical details"):
                st.code(f"{type(error).__name__}: {error}")


# ============================================================
# DISPLAY ANSWER AND SUPPORTING EVIDENCE
# ============================================================

if st.session_state.get("last_answer"):
    st.markdown("---")

    st.markdown(
        '<div class="section-title">Answer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(st.session_state["last_answer"])

    st.caption(
        f"Generated using "
        f"{st.session_state.get('last_model', 'Gemini')}"
    )

    evidence = st.session_state.get("last_evidence", [])

    if evidence:
        with st.expander("View supporting document passages"):

            for index, (text, metadata, distance) in enumerate(
                evidence,
                start=1,
            ):
                section = metadata.get("section_name", "Document")
                source = metadata.get("source", "Unknown document")
                pages_text = metadata.get("source_pages", "Unknown")

                st.markdown(
                    f"**Passage {index} · {section}**"
                )

                st.caption(
                    f"Source: {source} · Page(s): {pages_text}"
                )

                st.write(text)
                st.divider()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>DocuMind AI</strong><br>
        Your documents, understood.
    </div>
    """,
    unsafe_allow_html=True,
)
