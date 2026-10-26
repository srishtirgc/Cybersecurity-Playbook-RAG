
import ollama

from retriever import retrieve_documents


def generate_answer(question):
    """
    Retrieve relevant cybersecurity passages and generate
    an answer using the local Llama 3.2 model.
    """

    # Step 1: Retrieve relevant passages from ChromaDB.
    documents = retrieve_documents(question, n_results=5)

    if not documents:
        return "I could not find relevant information in the documents."

    # Step 2: Build the context and track source references.
    context_parts = []
    sources = []

    for document in documents:
        source = document["source"]
        page = document["page"]
        passage = document["text"]

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Passage: {passage}"
        )

        reference = (source, page)

        if reference not in sources:
            sources.append(reference)

    context = "\n\n".join(context_parts)

    # Step 3: Prepare instructions and retrieved context.
  
    
    prompt = f"""
You are a cybersecurity assistant answering questions using a CISA
incident and vulnerability response playbook.

QUESTION:
{question}

SOURCE PASSAGES:
{context}

INSTRUCTIONS:
1. Answer using only information explicitly supported by the source passages.
2. Separate containment, eradication, and recovery actions when relevant.
3. Include every relevant action supported by the passages.
4. Do not invent actions or introduce recommendations from general knowledge.
5. Do not claim that an action appears in the document unless the
   provided passages support that claim.
6. If the passages do not contain enough information, clearly state
   what information is missing.
7. Avoid contradictions. If you identify a relevant action, do not
   subsequently claim that no such action is mentioned.
8. Write a clear, practical answer using headings and bullet points.
9. Do not invent page numbers, quotations, or citations. Source
   references will be provided separately by the application.

10. Preserve the incident-response phases and checklist labels used in
    the source. Do not move an action into a different phase unless
    the source explicitly supports that classification.

11. Do not add MITRE ATT&CK identifiers, technique names, or other
    external classifications unless they appear in the provided
    passages and are relevant to the question.

12. For each recommended action, use only evidence from the supplied
    passages. If a statement comes from general cybersecurity
    knowledge rather than the source, do not present it as a
    recommendation from the playbook.

13. Do not invent page numbers or checklist references. If the source
    location of a claim is unclear, omit the reference rather than
    guessing.

14. Prioritize directly relevant passages. Do not include unrelated
    checklist items merely because they appear in the retrieved context.

15. Before stating that information is missing, check all supplied
    passages for that information.

16. Never list a recommendation as both supported and unsupported.
    If the source explicitly states an action, treat it as supported.

17. Do not add a "Missing Information" section unless the question
    asks for information that the supplied passages do not address.

18. Preserve the source's checklist numbering and incident-response
    phase labels. Do not infer a different phase classification.

19. Distinguish between:
    - Explicitly stated in the source
    - Reasonably inferred from the source
    - Not addressed by the supplied passages

    Do not present an inference as an explicit source statement.


Answer:
"""



    # Step 4: Ask the local Llama 3.2 model to generate an answer.
    response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}],
    keep_alive="30m",
)

    answer = response["message"]["content"]

    # Step 5: Append the retrieved source references.
    answer += "\n\nSources retrieved:\n"

    for source, page in sources:
        answer += f"- {source}, page {page}\n"

    return answer


if __name__ == "__main__":
    question = (
        "What containment and recovery actions should be taken "
        "for compromised accounts and credentials?"
    )

    print(f"\nQuestion: {question}\n")

    try:
        answer = generate_answer(question)
        print("Answer:\n")
        print(answer)

    except Exception as error:
        print(f"An error occurred: {error}")
        print(
            "\nCheck that Ollama is running, the llama3.2 model "
            "is installed, and the Python dependencies are available."
        )
