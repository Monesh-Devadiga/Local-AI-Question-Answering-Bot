import os
import streamlit as st
from rag_engine import get_rag_chain, AVAILABLE_MODELS
from ingest import CHROMA_DIR, ingest_uploaded_file
          
st.set_page_config(page_title="Local AI Q&A Bot", page_icon="🤖", layout="wide")
st.title("Local AI Q&A Bot")
st.caption("Powered by Ollama + LangChain + ChromaDB ")
st.markdown("It can answer questions from a set of Upload Documents (PDF/text documents).", unsafe_allow_html=True)
  
model_name = st.selectbox("Select Ollama Model", AVAILABLE_MODELS, index=0)

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF or text files here",
        type=["pdf", "txt", "md", "csv", "log"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uf in uploaded_files:
            with st.spinner(f"Indexing {uf.name}..."):
                try:
                    n = ingest_uploaded_file(uf.name, uf.read())
                    st.success(f"{uf.name}: {n} chunks added")
                except Exception as e:
                    st.error(f"{uf.name}: {e}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something about your documents...")

if user_input and user_input.strip():
    if not os.path.isdir(CHROMA_DIR):
        st.error("Please upload a document first using the sidebar.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    rag_invoke = get_rag_chain(model_name=model_name)

    past_msgs = [
        (m["content"], "")
        for m in st.session_state.chat_history[:-1]
        if m["role"] == "user"
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_invoke(user_input, past_msgs)
            st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})

if st.session_state.chat_history:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
