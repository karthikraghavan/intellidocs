"""
config.py
─────────
Single source of truth for all runtime settings.
Values are read from environment variables (or a .env file).

To adapt this chatbot to a different document:
  1.  Put your new PDF in data/
  2.  Update DOCUMENT_PATH and DOCUMENT_LABEL in .env
  3.  Delete chroma_db/ so the vector store is rebuilt on the next run
  4.  Run:  python ingestion.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Document source ─────────────────────────────────────────────────────────
DOCUMENT_PATH: str = os.getenv("DOCUMENT_PATH", "data/annual-report-2024-2025.pdf")
DOCUMENT_LABEL: str = os.getenv("DOCUMENT_LABEL", "Annual Report")

# ── Vector store ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "annual_report")

# ── Structured data (SQLite) ──────────────────────────────────────────────────
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "structured_db/data.db")

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "400"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "5"))

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

# ── Feature switches ──────────────────────────────────────────────────────────
# Set ENABLE_WEB_SEARCH=true in .env (and provide TAVILY_API_KEY) to activate.
ENABLE_WEB_SEARCH: bool = False

# ── Web crawling ──────────────────────────────────────────────────────────────
# How many link-levels deep to follow when ingesting a URL via the admin page.
# depth=1 → only the given page; depth=2 → that page + all pages it links to, etc.
# Set WEB_CRAWL_DEPTH=1 in .env to disable crawling (original behaviour).
WEB_CRAWL_DEPTH: int = int(os.getenv("WEB_CRAWL_DEPTH", "3"))

# ── Keys ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")  # optional

# ── Validation ────────────────────────────────────────────────────────────────
def validate():
    """Call once at startup to catch missing required config early."""
    import logging
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example → .env and add your key."
        )
    if DOCUMENT_PATH and not os.path.exists(DOCUMENT_PATH):
        logging.getLogger(__name__).warning(
            "DOCUMENT_PATH '%s' not found. "
            "Use the admin page (/admin) to upload documents.",
            DOCUMENT_PATH,
        )
