
from pypdf import PdfReader


def main():
    pdf_path = "documents/Aiml map .pdf"

    reader = PdfReader(pdf_path)

    print(f"Number of pages: {len(reader.pages)}")

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        print(f"\n--- Page {page_number} ---")
        print(text)


if __name__ == "__main__":
    main()
