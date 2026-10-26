
import streamlit as st
from rag import generate_answer

st.set_page_config(
    page_title="Cybersecurity RAG Assistant",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Cybersecurity RAG Assistant")
st.caption(
    "Ask questions about the CISA Federal Incident and "
    "Vulnerability Response Playbooks."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept a new question
question = st.chat_input(
    "Ask a cybersecurity question..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching the playbook and generating an answer..."
        ):
            try:
                answer = generate_answer(question)
            except Exception as error:
                answer = (
                    f"An error occurred: {error}\n\n"
                    "Check that Ollama is running, the model "
                    "is installed, and the ChromaDB collection exists."
                )

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
