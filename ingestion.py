"""
ingestion.py
────────────
One-time (or on-demand) pipeline:

    PDF  →  text chunks  →  OpenAI embeddings  →  ChromaDB (persisted on disk)

Run this before starting the chatbot:

    python ingestion.py

Re-run whenever you swap the source document (after deleting chroma_db/).
The script is idempotent: if the collection already has documents it skips
re-ingestion unless you pass --force.

Swapping documents
──────────────────
    1. Drop your new PDF in data/
    2. Update DOCUMENT_PATH (and optionally DOCUMENT_LABEL) in .env
    3. Delete chroma_db/  (rm -rf chroma_db)
    4. python ingestion.py
"""

import argparse
import logging
import os
import shutil
import sys
import chromadb

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Global ChromaDB Client ───────────────────────────────────────────────────
_chroma_client = None

def get_chroma_client():
    global _chroma_client
    if _chroma_client is not None:
        try:
            _chroma_client.heartbeat()
        except Exception:
            _chroma_client = None
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return _chroma_client

# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_pdf(path: str):
    """
    Load a PDF with PyPDFLoader (text-extractable PDFs only).
    Each page becomes a separate Document with metadata:
      {source: path, page: N}
    """
    log.info("Loading PDF: %s", path)
    loader = PyPDFLoader(path)
    docs = loader.load()
    log.info("  → %d pages loaded", len(docs))
    return docs


def _split_documents(docs, chunk_size: int, chunk_overlap: int):
    """
    Split raw page documents into smaller, overlapping chunks so that:
      - Each chunk fits comfortably in the LLM context window
      - Overlap preserves sentence continuity across chunk boundaries

    RecursiveCharacterTextSplitter tries to split on paragraph breaks,
    then sentences, then words — preserving semantic units wherever possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    log.info(
        "  → %d chunks (size=%d, overlap=%d)",
        len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def _build_vectorstore(chunks, collection_name: str, persist_dir: str):
    """
    Embed every chunk with OpenAI text-embedding-3-small and store in ChromaDB.

    ChromaDB is chosen because:
      • Runs 100% locally — no external service required
      • Persists to disk (survives restarts)
      • Fast cosine-similarity search
      • Free for any document size

    The collection is named after the document type so you can maintain
    multiple collections side-by-side.
    """
    log.info("Building vector store → %s/%s", persist_dir, collection_name)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=config.OPENAI_API_KEY,
    )

    # Chroma().from_documents() embeds + persists in one call
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        client=get_chroma_client(),
    )
    log.info(
        "  → Vector store ready  (%d vectors stored)",
        vectorstore._collection.count(),
    )
    return vectorstore


# ── Admin API helper ─────────────────────────────────────────────────────────

def ingest_documents(docs, label: str = "uploaded") -> int:
    """
    Embed and add pre-loaded LangChain Document objects to the existing
    ChromaDB collection (additive — does NOT wipe existing vectors).
    Returns the number of chunks added.

    Used by the admin upload API in main.py.
    """
    chunks = _split_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=config.OPENAI_API_KEY,
    )
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        client=get_chroma_client(),
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks)
    log.info("Added %d chunks from '%s' to vector store.", len(chunks), label)
    return len(chunks)


def list_sources() -> list:
    """Return a list of distinct sources currently in ChromaDB with chunk counts."""
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=config.OPENAI_API_KEY,
        )
        vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            client=get_chroma_client(),
            embedding_function=embeddings,
        )
        results = vectorstore._collection.get(include=["metadatas"])
        counts: dict = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": k, "chunks": v} for k, v in sorted(counts.items())]
    except Exception:
        return []


def clear_all_sources() -> int:
    """Delete every document from ChromaDB and all structured tables from SQLite.
    Returns the total number of items deleted."""
    total = 0

    # Clear ChromaDB
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=config.OPENAI_API_KEY,
        )
        vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            client=get_chroma_client(),
            embedding_function=embeddings,
        )
        ids = vectorstore._collection.get()["ids"]
        if ids:
            vectorstore._collection.delete(ids=ids)
        total += len(ids)
    except Exception:
        pass

    # Clear SQLite structured tables
    try:
        from structured_db import clear_all_structured_tables
        total += clear_all_structured_tables()
    except Exception:
        pass

    return total


# ── Structured data ingestion ─────────────────────────────────────────────────

def ingest_excel_to_sqlite(file_path: str, source_label: str) -> dict:
    """
    Read all sheets from an Excel file and ingest each sheet as a SQLite table.

    Args:
        file_path:    Path to the .xlsx / .xls file.
        source_label: Human-readable label (usually the original filename).

    Returns:
        dict with keys 'tables' (list of {sheet, table, rows}) and 'errors'.
    """
    import pandas as pd
    from structured_db import ingest_dataframe

    results = []
    errors  = []

    try:
        xl = pd.ExcelFile(file_path)
    except Exception as e:
        return {"tables": [], "errors": [f"Failed to open Excel file: {e}"]}

    for sheet_name in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sheet_name=sheet_name)
            if df.empty:
                log.info("Sheet '%s' is empty — skipping.", sheet_name)
                continue

            # Use "<source_label>_<sheet_name>" as the table name
            table_label = f"{source_label}_{sheet_name}"
            row_count   = ingest_dataframe(df, table_name=table_label, source_label=source_label)
            results.append({
                "sheet": sheet_name,
                "table": table_label,
                "rows":  row_count,
            })
            log.info("Ingested sheet '%s' → %d rows into SQLite.", sheet_name, row_count)
        except Exception as e:
            errors.append(f"Sheet '{sheet_name}': {e}")
            log.exception("Failed to ingest sheet '%s'", sheet_name)

    return {"tables": results, "errors": errors}


# ── Entry point ───────────────────────────────────────────────────────────────

def ingest(force: bool = False):
    config.validate()

    if os.path.exists(config.CHROMA_PERSIST_DIR) and not force:
        # Quick check: if the collection already exists and has documents, skip.
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=config.OPENAI_API_KEY,
        )
        existing = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            client=get_chroma_client(),
            embedding_function=embeddings,
        )
        count = existing._collection.count()
        if count > 0:
            log.info(
                "Vector store already populated (%d vectors). "
                "Use --force to re-ingest.",
                count,
            )
            return existing

    if force and os.path.exists(config.CHROMA_PERSIST_DIR):
        log.info("--force: removing existing vector store")
        shutil.rmtree(config.CHROMA_PERSIST_DIR)

    log.info("Path: %s", config.DOCUMENT_PATH)
    docs   = _load_pdf(config.DOCUMENT_PATH)
    chunks = _split_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    return _build_vectorstore(
        chunks,
        config.CHROMA_COLLECTION_NAME,
        config.CHROMA_PERSIST_DIR,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF into vector store")
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing vector store and re-ingest from scratch",
    )
    args = parser.parse_args()

    try:
        ingest(force=args.force)
        log.info("Ingestion complete. You can now run the chatbot.")
    except (EnvironmentError, FileNotFoundError) as e:
        log.error("%s", e)
        sys.exit(1)
