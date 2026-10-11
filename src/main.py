
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

    print("\n--- Chunking Results ---")
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk preview: {chunks[0][:200]}")

if __name__ == "__main__":
    main()
