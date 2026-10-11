
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

def split_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def main():
    pdf_path = "documents/Aiml map .pdf"

    reader = PdfReader(pdf_path)

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

    embeddings = model.encode(chunks)
    
    question = "What skills should I learn to become an AI Engineer?"

    question_embedding = model.encode([question])

    similarities = model.similarity(question_embedding, embeddings)

    best_match_index = similarities.argmax().item()

    print("\n--- Semantic Search Results ---")
    print(f"Question: {question}")
    print(f"Best matching chunk: {best_match_index + 1}")
    print(f"Similarity score: {similarities[0][best_match_index].item():.4f}")
    print(f"Matched text: {chunks[best_match_index][:300]}")
    print("\n--- Embedding Results ---")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Vector dimensions: {embeddings.shape[1]}")

    print("\n--- Chunking Results ---")
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk preview: {chunks[0][:200]}")

if __name__ == "__main__":
    main()
