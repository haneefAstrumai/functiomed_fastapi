from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
import os, re, asyncio
from web_data.web_data import get_all_text_with_metadata
from pdf_data.pdf_data import load_and_chunk_pdfs
from pydantic import BaseModel
from typing import List, Dict
from embedding.embedding import build_or_load_vectorstore, retrieve
from chating.chating import ask_llm
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env into os.environ


app = FastAPI(title="Functiomed RAG Scraper")

# -------------------------
# CORS Middleware Configuration
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Fixed!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Directories
# -------------------------
RAW_DIR = "data/raw_html"
CLEAN_DIR = "data/clean_text"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# -------------------------
# Config
# -------------------------
BASE_URL = "https://www.functiomed.ch/angebot/osteophatie-etiopathie"
visited = set()
to_visit = {BASE_URL: 0}

MAX_PAGES =60
MAX_DEPTH =20

# -------------------------
# Helper functions
# -------------------------
def normalize_url(url):
    parsed = urlparse(url)
    url = urlunparse(parsed._replace(fragment=""))
    if url.endswith("/") and url != BASE_URL+"/":
        url = url[:-1]
    return url.lower()

def is_internal_link(url):
    return urlparse(url).netloc == urlparse(BASE_URL).netloc

def is_valid_page(url):
    bad_ext = (".pdf", ".jpg", ".png", ".jpeg", ".svg", ".zip")
    if url.lower().endswith(bad_ext):
        return False
    if "undefined" in url:
        return False
    return True

def skip_dynamic_pages(url):
    m = re.search(r"news/page/(\d+)", url)
    if m and int(m.group(1)) > 20:
        return True
    return False

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_html(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    main = soup.find("main")
    content = main.get_text(" ") if main else soup.get_text(" ")
    return clean_text(content)

# -------------------------
# Endpoints
# -------------------------


# @app.get("/scrape")
# async def scrape_site():
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=True)
#         page = await browser.new_page()
#         scraped_count = 0

#         while to_visit and len(visited) < MAX_PAGES:
#             url, depth = to_visit.popitem()
#             url = normalize_url(url)
#             if url in visited or depth > MAX_DEPTH:
#                 continue
#             visited.add(url)

#             if skip_dynamic_pages(url):
#                 continue

#             print(f"Scraping ({scraped_count+1}): {url}")

#             try:
#                 await page.goto(url, timeout=30000)
#                 await page.wait_for_load_state("networkidle")
#                 html = await page.content()

#                 # Save HTML
#                 filename = url.replace("https://", "").replace("/", "_") + ".html"
#                 with open(os.path.join(RAW_DIR, filename), "w", encoding="utf-8") as f:
#                     f.write(html)

#                 # Extract text
#                 text = extract_text_from_html(html)
#                 out_file = filename.replace(".html", ".txt")
#                 with open(os.path.join(CLEAN_DIR, out_file), "w", encoding="utf-8") as f:
#                     f.write(text)

#                 scraped_count += 1

#                 # Collect links
#                 links = await page.eval_on_selector_all(
#                     "a[href]", "elements => elements.map(el => el.href)"
#                 )
#                 for link in links:
#                     link = normalize_url(link)
#                     if is_internal_link(link) and is_valid_page(link) and link not in visited:
#                         to_visit[link] = depth + 1

#             except Exception as e:
#                 print(f"Failed {url}: {e}")

#         await browser.close()
#         print(f"Scraping completed. Total pages: {scraped_count}")

#     return {"status": "completed", "pages_scraped": scraped_count}

@app.get("/all_text")
def all_text():
    docs = get_all_text_with_metadata()

    return {
        "total_chunks": len(docs),
        "documents": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
    }

@app.get("/pdfs")
def get_pdf_chunks():
    docs = load_and_chunk_pdfs()

    return {
        "total_chunks": len(docs),
        "documents": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
    }
#----------------------------------------------//////////////--------------------------------------
# Load FAISS index once on startup
vector_store = None

# For local
# @app.on_event("startup")
# def startup_event():
#     global vector_store
#     vector_store = build_or_load_vectorstore()

# For deployment
@app.on_event("startup")
def startup_event():
    global vector_store
    try:
        vector_store = build_or_load_vectorstore()
        print("Vector store loaded")
    except Exception as e:
        print("Vector store load failed:", e)

# -------------------------
# Request model
# -------------------------
class QueryRequest(BaseModel):
    query: str
    k: int = 5  # number of chunks to retrieve after reranking


# -------------------------
# Ingest endpoint
# -------------------------
@app.post("/ingest")
def ingest_data():
    """
    Build or reload FAISS index (ingest all PDFs + web chunks)
    """
    global vector_store
    vector_store = build_or_load_vectorstore()
    return {"message": "Vector store built or loaded successfully"}


# -------------------------
# Ask endpoint with hybrid + reranking
# -------------------------
@app.post("/retrieve")
def retrieve_text(request: QueryRequest):
    """
    Query using hybrid retrieval (FAISS + BM25) with CrossEncoder reranking
    """
    global vector_store
    if vector_store is None:
        vector_store = build_or_load_vectorstore()

    # Use your hybrid + reranking retrieve function
    results = retrieve(request.query, top_n=request.k)

    return {
        "query": request.query,
        "results": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in results
        ]
    }

# Chat endpoint
class ChatQueryRequest(BaseModel):
    query: str
@app.post("/chat")
def chat(request: ChatQueryRequest):
    """
    Ask the AI chatbot a question using the RAG + LLM pipeline.
    """
    user_query = request.query
    answer = ask_llm(user_query)
    print(f"query: {user_query}")
    print(f"Answer: {answer}")
    return {
        "query": user_query,
        "answer": answer
    }