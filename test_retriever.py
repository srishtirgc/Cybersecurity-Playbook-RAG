
from retriever import retrieve_documents


question = (
    "What containment and recovery actions should be taken "
    "for compromised accounts and credentials, including "
    "password resets and revoking privileged access?"
)

documents = retrieve_documents(question, n_results=5)

print(f"\nQuestion: {question}\n")

for i, document in enumerate(documents, start=1):
    print(f"--- Result {i} ---")
    print(f"Distance: {document['distance']:.4f}")
    print(f"Source: {document['source']}")
    print(f"Page: {document['page']}")
    print(document["text"])
    print()
