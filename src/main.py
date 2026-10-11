
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb


def split_into_chunks(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def main():
    pdf_path = "documents/Aiml map .pdf"

    reader = PdfReader(pdf_path)
    print("DEBUG: PDF opened")
    
    print(f"Number of pages: {len(reader.pages)}")
    all_text = []
    
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        all_text.append(text)
        
        print(f"Processed page {page_number}")
    
    full_text = "\n".join(all_text)
    print(f"\nTotal words in PDF: {len(full_text.split())}")
    
    sample_text = "Python is easy to learn. SQL manages data. AI answers questions."

    chunks = split_into_chunks(full_text, chunk_size=500)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection(
    name="ai_document_collection"
    )
    embeddings = model.encode(chunks)
    print(f"DEBUG: Created {len(chunks)} chunks")
    print("DEBUG: Generating embeddings...")
    
    print("DEBUG: Embeddings generated")
    
    collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
    )

    print(f"Stored {collection.count()} chunks in ChromaDB.")
    print("Expected chunks:", len(chunks))
    question = "Which programming and data skills should I learn first?"
    
    results = collection.query(
    query_embeddings=model.encode([question]).tolist(),
    n_results=1
    )

    print("\n--- ChromaDB Search Results ---")
    print(f"Question: {question}")
    print(f"Matched text: {results['documents'][0][0]}")
    print(f"Chunk ID: {results['ids'][0][0]}")
    print(f"Distance: {results['distances'][0][0]:.4f}")
    print("\n--- Embedding Results ---")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Vector dimensions: {embeddings.shape[1]}")

    print("\n--- Chunking Results ---")
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk preview: {chunks[0][:200]}")

if __name__ == "__main__":
    main()
