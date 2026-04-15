from pathlib import Path
import tempfile

from langchain_community.document_loaders import PyPDFLoader


def save_uploaded_pdf(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def load_pdf_documents(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()
