
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# 1. Define where our project and documents live
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

# 2. Load a local embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME)

# 3. Set up the local vector database
client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name="cybersecurity_documents"
)


def load_documents():
    """Extract text from every PDF in the data folder."""
    documents = []

    for pdf_path in DATA_DIR.glob("*.pdf"):
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number,
                })

    return documents


def chunk_text(text, chunk_size=800, overlap=120):
    """Split text into overlapping character-based chunks."""
    text = " ".join(text.split())
    chunks = []

    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def store_chunks(documents):
    """Embed document chunks and save them in ChromaDB."""
    ids = []
    texts = []
    metadatas = []

    for document in documents:
        chunks = chunk_text(document["text"])

        for chunk_number, chunk in enumerate(chunks):
            ids.append(
                f'{document["source"]}-page-{document["page"]}-chunk-{chunk_number}'
            )
            texts.append(chunk)
            metadatas.append({
                "source": document["source"],
                "page": document["page"],
            })

    if not texts:
        print("No text chunks found. Check your PDF files.")
        return

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    # Replace previously indexed copies of these IDs
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Stored {len(texts)} chunks from {len(documents)} PDF pages.")


if __name__ == "__main__":
    documents = load_documents()

    if not documents:
        print("No readable PDF text found in the data folder.")
    else:
        store_chunks(documents)
