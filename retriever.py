
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_collection(
    name="cybersecurity_documents"
)


def retrieve_documents(question, n_results=5):
    """Retrieve the most relevant document chunks for a question."""

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_documents = []

    for i, document in enumerate(results["documents"][0]):
        retrieved_documents.append({
            "text": document,
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i],
        })

    return retrieved_documents
