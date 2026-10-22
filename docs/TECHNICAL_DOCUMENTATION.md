# DocuMind AI — Technical Documentation

## 1. Project Overview

DocuMind AI is an AI-powered document question-answering application that allows users to upload multiple PDF documents and ask questions about their contents.

It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and provide answers grounded in the retrieved text. The application also provides source citations and evidence passages to help users verify the answers.

## 2. Problem Statement

Reading and searching through lengthy documents manually can be time-consuming. General-purpose AI models may also produce answers that are not supported by a particular document.

DocuMind AI addresses this problem by retrieving relevant passages from uploaded PDFs and using those passages as context for answer generation.

## 3. Key Features

- Upload and process multiple PDF documents.
- Extract text from PDFs while preserving page information.
- Split extracted text into smaller, section-aware chunks.
- Generate semantic embeddings using Sentence Transformers.
- Store and retrieve document chunks using ChromaDB.
- Generate context-grounded answers using the Google Gemini API.
- Display source citations and supporting evidence passages.
- Support questions involving information across uploaded documents.
- Handle API errors and model fallback scenarios.
- Run automated tests through GitHub Actions.

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and document processing |
| Streamlit | Interactive web application |
| PyPDF | PDF text extraction |
| Sentence Transformers | Semantic embedding generation |
| all-MiniLM-L6-v2 | Embedding model used in the document-processing pipeline |
| ChromaDB | Vector storage and similarity retrieval |
| Google Gemini API | Context-grounded answer generation |
| pytest | Automated testing |
| Git and GitHub | Version control and project hosting |
| GitHub Actions | Automated test workflow |

## 5. System Architecture

The application follows a Retrieval-Augmented Generation (RAG) workflow.

### Document ingestion

1. The user uploads one or more PDF documents.
2. The application extracts text and tracks page numbers.
3. The extracted text is divided into smaller chunks.
4. Each chunk is associated with metadata such as its source filename, page information, and section name.
5. Sentence Transformers converts the text into numerical embeddings.
6. ChromaDB stores the document chunks, embeddings, and metadata for retrieval.

### Question answering

1. The user enters a question in the Streamlit interface.
2. The retrieval component searches for document chunks relevant to the question.
3. The application prepares a prompt containing the question and retrieved context.
4. The Google Gemini API generates an answer using the supplied context.
5. The application presents the answer alongside source citations and evidence passages.

## 6. Retrieval-Augmented Generation (RAG)

RAG combines information retrieval with language-model generation.

Instead of relying only on information learned during model training, the application retrieves relevant passages from the user's documents and supplies them as context to the language model.

The workflow can be summarized as:

**PDFs → Text extraction → Chunking → Embeddings → Vector retrieval → Context-grounded generation → Answer with evidence**

### Why use RAG?

- It connects answers to the contents of the uploaded documents.
- It allows users to query their own information.
- It provides a way to inspect the sources supporting an answer.
- It reduces reliance on the language model's general knowledge for document-specific questions.

RAG does not guarantee that every generated answer is correct, so citations and evidence should still be checked.

## 7. Semantic Embeddings and Vector Search

An embedding represents text as a numerical vector that captures aspects of its meaning.

Sentence Transformers generates embeddings for document chunks. ChromaDB uses vector similarity to identify chunks that are semantically relevant to a question.

This approach can retrieve related passages even when the wording of the question differs from the wording in the document.

## 8. Chunking and Metadata

Large documents are divided into smaller text chunks because retrieving an entire document for every question would be inefficient and could provide too much irrelevant context.

Section-aware chunking helps preserve useful document structure. Chunk overlap helps retain context across neighboring chunks.

Metadata associates retrieved passages with their source documents, page information, and section names. This supports evidence display and source attribution.

## 9. Grounded Answers and Citations

The application supplies retrieved document passages to the language model as context.

The answer-generation instructions emphasize using the available evidence and avoiding invented filenames or page numbers.

Source citations help users trace an answer to the relevant document and page. Evidence passages provide additional context for checking whether the answer is supported by the retrieved text.

Citations improve traceability, but users should still verify that the cited passages support the generated claims.

## 10. Codebase Structure

| File | Responsibility |
|---|---|
| `src/app.py` | Streamlit interface, PDF uploads, document indexing, retrieval workflow, and answer presentation |
| `src/main.py` | Core PDF processing, chunking, embedding, vector-store, retrieval, and prompt-related functionality |
| `src/gemini_service.py` | Gemini API integration, answer generation, model fallback, and retry handling |
| `src/test_gemini.py` | Tests for Gemini-related functionality |
| `src/test_gemini_service.py` | Tests for the Gemini service |
| `src/test_rag.py` | Tests for retrieval-augmented generation functionality |
| `src/__init__.py` | Python package initialization |

## 11. Error Handling and Reliability

The project includes handling for relevant document-processing and API failure scenarios.

The Gemini service validates the input prompt, checks API configuration, attempts supported model alternatives when necessary, and includes retry handling.

Automated tests help detect regressions when the code changes.

## 12. Testing and Continuous Integration

The project uses automated tests and GitHub Actions.

The latest completed workflow run passed all **54 tests**. This verifies the behavior covered by the current test suite; it does not guarantee that every possible scenario is free of defects.

## 13. Running the Project Locally

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies listed in the project's dependency file.
4. Configure the `GEMINI_API_KEY` environment variable as required by the application.
5. Start the Streamlit application using the project's documented launch command.

Consult the main README for the exact setup commands and required configuration. Keep API keys private and never commit them to GitHub.

## 14. Current Limitations

- PDFs must contain extractable text. Scanned documents may require an OCR step.
- Answer quality depends on the quality of PDF extraction, chunking, retrieval, and the language model's response.
- Incorrect or incomplete retrieval can lead to incomplete answers.
- Generated answers and citations require verification.
- API availability, quotas, and network connectivity can affect answer generation.

## 15. Potential Future Improvements

- Add OCR support for scanned PDFs.
- Improve retrieval using hybrid keyword and semantic search.
- Add retrieval evaluation metrics and a curated question-answer test set.
- Improve handling of tables, images, and complex PDF layouts.
- Add more detailed latency, retrieval-quality, and error monitoring.
- Expand test coverage for difficult documents and edge cases.

## 16. Learning Outcomes

This project demonstrates practical experience with:

- Python application development.
- PDF processing and text extraction.
- Natural language processing and text embeddings.
- Vector databases and semantic retrieval.
- Retrieval-Augmented Generation (RAG).
- Integration with a large language model API.
- Source attribution and evidence-grounded responses.
- Automated testing and continuous integration.
- Building and deploying an interactive AI application.

## 17. Project Links

- **GitHub Repository:** https://github.com/woohasri-ai/ai-document-intelligence-rag
- **Live Application:** https://ai-document-intelligence-rag-ha25okq7mgpfei5lz7rsf5.streamlit.app/
