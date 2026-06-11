from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os


class PDFProcessor:
    def __init__(self, chunk_size: int = 3000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_from_bytes(self, file_bytes: bytes, filename: str) -> List[Any]:
        """Save uploaded bytes to a temp file and load with PyMuPDF."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            loader = PyMuPDFLoader(tmp_path)
            documents = loader.load()
            print(f"[INFO] Loaded {len(documents)} pages from '{filename}'.")
            return documents
        finally:
            os.remove(tmp_path)

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into chunks for summarization."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} pages into {len(chunks)} chunks.")
        return chunks
