from web_data.web_data import get_all_text_with_metadata
from pdf_data.pdf_data import load_and_chunk_pdfs
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder


# For local
# VECTOR_DB_PATH = "data/faiss_index"

# For deployment on rendor
import os
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/faiss_index")


def load_all_chunks():
    web_docs = get_all_text_with_metadata()
    pdf_docs = load_and_chunk_pdfs()
    return web_docs + pdf_docs


def build_or_load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-mpnet-base-v2",  #
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    try:
        # Load existing FAISS index
        vector_store = FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return vector_store

    except Exception:
        # Build new index
        docs = load_all_chunks()
        vector_store = FAISS.from_documents(
            documents=docs,
            embedding=embedding_model
        )
        vector_store.save_local(VECTOR_DB_PATH)
        return vector_store


def load_reranker():
    # Load CrossEncoder reranker
    return CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
                        


def retrieve(query: str, top_n: int = 5):
    """
    Hybrid retrieval: FAISS + BM25 + CrossEncoder reranking
    """
    try:
        # Load vector store and all chunks
        vector_store = build_or_load_vectorstore()
        all_chunks = load_all_chunks()

        # 1️⃣ FAISS retrieval
        faiss_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_n * 2}  # retrieve extra for reranking
        )
        faiss_docs = faiss_retriever.invoke(query)

        # 2️⃣ BM25 retrieval
        bm25 = BM25Retriever.from_documents(all_chunks, bm25_variant="plus")
        bm25.k = top_n * 2
        bm25_docs = bm25.invoke(query)

        # 3️⃣ Combine and deduplicate
        combined_docs = []
        seen = set()
        for doc in faiss_docs + bm25_docs:
            if doc.page_content not in seen:
                combined_docs.append(doc)
                seen.add(doc.page_content)

        if not combined_docs:
            return []

        # 4️⃣ Rerank using CrossEncoder
        reranker = load_reranker()
        pairs = [[query, doc.page_content] for doc in combined_docs]
        scores = reranker.predict(pairs)

        ranked_docs = sorted(
            zip(combined_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top N after reranking
        return [doc for doc, score in ranked_docs[:top_n]]

    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        return []
