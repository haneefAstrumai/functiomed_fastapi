import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CLEAN_DIR = "data/clean_text"

def get_all_text_with_metadata() -> List[Document]:
    documents: List[Document] = []

    for filename in os.listdir(CLEAN_DIR):
        if filename.endswith(".txt"):
            page_name = filename.replace(".txt", "")
            file_path = os.path.join(CLEAN_DIR, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"page_name": page_name}
                    )
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)
