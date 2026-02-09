import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_DIR = "pdf_data/files"

def load_all_pdfs() -> List[Document]:
    """
    Load all PDFs from PDF_DIR and return LangChain Documents
    """
    all_docs: List[Document] = []

    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # Add filename to metadata
            for doc in docs:
                doc.metadata["source_pdf"] = filename

            all_docs.extend(docs)

    return all_docs

def load_and_chunk_pdfs(chunk_size=500, chunk_overlap=100):
    docs = load_all_pdfs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(docs)
