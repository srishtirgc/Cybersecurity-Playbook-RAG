
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

# Load the same embedding model used during ingestion
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Open the existing vector database
client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_collection(
    name="cybersecurity_documents"
)

# Ask a test question
question = (
    "What containment and recovery actions should be taken "
    "for compromised accounts and credentials, including "
    "password resets and revoking privileged access?"
)
# Convert the question into an embedding
query_embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

# Retrieve the 3 most relevant chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

print(f"\nQuestion: {question}\n")
print("Retrieved passages:\n")

for i, document in enumerate(results["documents"][0], start=1):
    metadata = results["metadatas"][0][i - 1]
    distance = results["distances"][0][i - 1]

    print(f"--- Result {i} ---")
    print(f"Distance: {distance:.4f}")
    print(f"Source: {metadata['source']}")
    print(f"Page: {metadata['page']}")
    print(document)
    print()
