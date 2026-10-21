# DocuMind AI — AI-Powered Document Intelligence & Question Answering

DocuMind AI is an AI-powered document question-answering application that uses **Retrieval-Augmented Generation (RAG)** to answer questions from uploaded PDF documents. It retrieves relevant document passages, generates answers using Google Gemini, and displays supporting passages with source page numbers to help users verify the answers.

## Features

- **Multiple PDF uploads:** Upload and query multiple PDF documents in one session.
- **Document text extraction:** Extract searchable text from PDF files using `pypdf`.
- **Section-aware chunking:** Split document content into manageable chunks while preserving section context.
- **Semantic retrieval:** Use sentence embeddings and ChromaDB vector search to retrieve relevant passages.
- **AI-powered answers:** Generate context-grounded responses using the Gemini API.
- **Source citations:** Display supporting passages and PDF page numbers alongside answers.
- **Reduced unsupported answers:** Provide retrieved document context to the language model so answers can be grounded in the supplied documents.
- **Interactive interface:** Use a Streamlit web interface to upload documents and ask questions.
- **Automated tests:** Test core RAG functionality and Gemini service integration.

## Technology Stack

- **Python 3.12**
- **Streamlit** — interactive web application
- **pypdf** — PDF text extraction
- **Sentence Transformers** — text embeddings
- **ChromaDB** — vector storage and retrieval
- **Google Gemini API** — answer generation
- **Python unittest** — automated testing
- **Git and GitHub** — version control

## How It Works

1. **Upload:** The user uploads one or more PDF documents.
2. **Extract:** The application extracts text and page information from the PDFs.
3. **Chunk:** Extracted text is divided into section-aware chunks.
4. **Embed and index:** Text chunks are converted into embeddings and stored in ChromaDB.
5. **Retrieve:** The application searches for relevant passages based on the user's question.
6. **Generate:** Retrieved passages are supplied as context to Gemini.
7. **Display:** The application presents the answer along with supporting document passages and page references.

## Project Structure

```text
ai-document-intelligence-rag/
├── documents/              # Local PDF documents (not committed)
├── src/
│   ├── __init__.py
│   ├── app.py               # Streamlit application
│   ├── main.py              # RAG, chunking and retrieval logic
│   ├── gemini_service.py    # Gemini API integration
│   ├── test_rag.py          # RAG tests
│   └── test_gemini_service.py
├── uploaded_chroma_db/      # Local vector database (not committed)
├── .env                     # Private API key (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.12 or a compatible Python version
- Git
- A Google Gemini API key

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/woohasri-ai/ai-document-intelligence-rag.git
cd ai-document-intelligence-rag
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root and add:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own API key. Never commit your `.env` file or share your API key publicly.

### 5. Run the application

**Windows PowerShell:**

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m streamlit run src/app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501` or another available port.

## Running the Tests

Run the automated RAG and Gemini service tests:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m unittest src.test_rag src.test_gemini_service -v
```

The project's RAG and mocked Gemini service test suites have previously passed **29 tests**. Run the command above to verify the current checkout.

## Example Questions

After uploading relevant PDFs, try questions such as:

- What technical skills are mentioned in the resume?
- What evaluation metrics should I use to assess an AI chatbot?
- What technologies are covered in Phase 9 of the roadmap?
- Compare information discussed across the uploaded documents.

Answers depend on the content of the uploaded documents and the quality of retrieved passages.

## Current Limitations

- PDF files must contain extractable text; scanned documents may require OCR.
- Answer quality depends on document quality, chunking, retrieval relevance, and the language model.
- Gemini API access requires a valid API key and may incur usage costs.
- Local vector data is stored on the machine and is excluded from version control.
- The application is currently designed as a local Streamlit application; public deployment and production hardening are separate steps.

## Future Improvements

- Add OCR support for scanned PDFs.
- Improve retrieval ranking and evaluation.
- Add document-level filtering and richer citation navigation.
- Add a systematic RAG evaluation dataset and performance metrics.
- Deploy the application to a hosted environment.
- Add authentication and production-grade security where needed.

## Author

**Wooha Sri**

GitHub: [woohasri-ai](https://github.com/woohasri-ai)

---

*DocuMind AI — Your documents, understood.*
