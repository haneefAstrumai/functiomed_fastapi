"""
Configuration file for Functiomed RAG System
Centralizes all magic numbers and settings
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────
API_TITLE = "Functiomed RAG API"
API_VERSION = "2.0.0"
API_DESCRIPTION = "Production RAG chatbot with source attribution"

# ─────────────────────────────────────────────
# CORS Configuration
# ─────────────────────────────────────────────
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8080",
    # Add production domains here:
    # "https://functiomed.ch",
    # "https://www.functiomed.ch",
]

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
RAW_DIR = "data/raw_html"
CLEAN_DIR = "data/clean_text"
VECTOR_DB_PATH = "data/faiss_index"
PDF_DIR = "pdf_data/files"
PDF_FILE_PREFIX = "pdf__"

# ─────────────────────────────────────────────
# Chunking Configuration
# ─────────────────────────────────────────────
CHUNK_SIZE = 700  # Optimal for medical content (was 400 - too small!)
CHUNK_OVERLAP = 150  # 20-25% of chunk size is recommended
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ─────────────────────────────────────────────
# Retrieval Configuration
# ─────────────────────────────────────────────
# How many documents to retrieve before reranking
CANDIDATE_MULTIPLIER = 3  # Fetch 3x more candidates than target

# How many final documents to return
DEFAULT_TOP_N = 6  # Balance between context and speed

# CrossEncoder reranking
RERANK_BATCH_SIZE = 16  # Process in batches to avoid hanging
MAX_CONTENT_LENGTH_FOR_RERANKING = 512  # Truncate long docs

# Relevance threshold (boosted scores must be >= this)
RELEVANCE_THRESHOLD = -2.5

# ─────────────────────────────────────────────
# Query Intent Boosting
# ─────────────────────────────────────────────
WEB_BOOST_INFORMATION = 3.0  # Strong boost for info queries
WEB_BOOST_GENERAL = 1.5      # Moderate boost for general queries
WEB_BOOST_FORM = 0.0         # Neutral for form queries

# ─────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "qwen/qwen3-32b"
LLM_TEMPERATURE = 0  # Deterministic responses
LLM_MAX_TOKENS = 2000  # Reasonable limit (was 5000 - too high!)
LLM_TIMEOUT = 30  # seconds
LLM_MAX_RETRIES = 2

# How many chunks to send to LLM (must align with DEFAULT_TOP_N)
LLM_CONTEXT_CHUNKS = 12  # Up from 6 to give more context

# ─────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DEVICE = "cpu"  # Change to "cuda" if GPU available

# ─────────────────────────────────────────────
# CrossEncoder Model
# ─────────────────────────────────────────────
RERANKER_MODEL_PRIMARY = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_MODEL_FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not found in environment variables")

# Validate chunk settings
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(f"❌ CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be < CHUNK_SIZE ({CHUNK_SIZE})")

# Validate retrieval settings
if LLM_CONTEXT_CHUNKS < DEFAULT_TOP_N:
    import warnings
    warnings.warn(
        f"⚠️ LLM_CONTEXT_CHUNKS ({LLM_CONTEXT_CHUNKS}) < DEFAULT_TOP_N ({DEFAULT_TOP_N}). "
        f"Some retrieved docs won't be sent to LLM."
    )

print("✅ Configuration loaded successfully")