import os

import streamlit as st

from src.loader import save_uploaded_pdf, load_pdf_documents
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore
from src.retriever import get_retriever

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot")
st.caption("Upload a PDF and retrieve the most relevant context for your question.")


@st.cache_resource(show_spinner=False)
def create_retriever_from_pdf(file_bytes: bytes, file_name: str):
    class UploadedFileWrapper:
        def __init__(self, name: str, data: bytes):
            self.name = name
            self._data = data

        def getvalue(self):
            return self._data

    wrapped_file = UploadedFileWrapper(file_name, file_bytes)
    temp_pdf_path = save_uploaded_pdf(wrapped_file)

    try:
        documents = load_pdf_documents(temp_pdf_path)
        chunks = split_documents(documents)
        embeddings = get_embeddings()
        vectorstore = build_vectorstore(chunks, embeddings)
        retriever = get_retriever(vectorstore)
        return retriever, len(documents), len(chunks)
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a PDF to begin.")
else:
    with st.spinner("Processing PDF and building vector index..."):
        retriever, page_count, chunk_count = create_retriever_from_pdf(
            uploaded_file.getvalue(),
            uploaded_file.name,
        )

    col1, col2 = st.columns(2)
    col1.metric("Pages loaded", page_count)
    col2.metric("Chunks created", chunk_count)

    question = st.text_input("Ask a question about the PDF")

    if question:
        with st.spinner("Retrieving relevant context..."):
            docs = retriever.invoke(question)

        st.subheader("Top retrieved context")
        for i, doc in enumerate(docs, start=1):
            with st.expander(f"Chunk {i}"):
                st.write(doc.page_content)
                if doc.metadata:
                    st.caption(f"Metadata: {doc.metadata}")
