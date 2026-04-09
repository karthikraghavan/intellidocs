"""
main.py
────────
FastAPI server — the runtime entry point.

Endpoints
─────────
POST /chat              — submit a question, get an answer + trace metadata
GET  /health            — liveness check (also reports vector store status)
DELETE /session         — clear chat history for a given session_id
GET  /document-info     — returns document label and vector store stats
GET  /admin             — admin UI for uploading sources
POST /admin/upload      — ingest PDF, DOCX files or URLs into ChromaDB
GET  /admin/sources     — list all ingested sources
DELETE /admin/sources   — clear all sources from ChromaDB

Run with:
    uvicorn main:app --reload --port 8000
"""

import os
import shutil
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

import config

config.validate()  # fail fast if env is misconfigured

from graph.graph import graph                        # noqa: E402
from graph.nodes.retrieve import invalidate_retriever  # noqa: E402
from ingestion import ingest_documents, list_sources, clear_all_sources, get_chroma_client, ingest_excel_to_sqlite  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic RAG Chatbot",
    description=f"Document Q&A powered by LangGraph — {config.DOCUMENT_LABEL}",
    version="1.0.0",
)


# ── In-memory session store ───────────────────────────────────────────────────
_sessions: dict[str, list] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    steps: list[str]
    hallucination_free: bool
    answer_useful: bool
    session_id: str
    sources: list[str]
    sql_queries_used: list[str]   # SQL queries executed to produce this answer (empty for text-only questions)

class HealthResponse(BaseModel):
    status: str
    document: str
    vector_store_ready: bool


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Run the Agentic RAG pipeline for one question."""
    session_id   = request.session_id
    chat_history = _sessions.get(session_id, [])

    initial_state = {
        "question":         request.question,
        "chat_history":     chat_history,
        "steps":            [],
        "_gen_retries":     0,
        "sql_queries_used": [],
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        log.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail=str(e))

    answer = final_state.get("generation", "No answer generated.")

    chat_history.append({"role": "user",      "content": request.question})
    chat_history.append({"role": "assistant", "content": answer})
    _sessions[session_id] = chat_history

    sources = []
    for doc in final_state.get("documents", []):
        page = doc.metadata.get("page")
        src  = doc.metadata.get("source", "")
        if page is not None:
            sources.append(f"p.{page + 1}")
        elif src.startswith("http"):
            sources.append(src)
        elif src:
            sources.append(Path(src).name)
    sources = list(dict.fromkeys(sources))[:5]

    sql_queries = final_state.get("sql_queries_used", [])
    if sql_queries:
        log.info(
            "SQL queries used in session '%s':\n%s",
            session_id,
            "\n".join(f"  [{i+1}] {q}" for i, q in enumerate(sql_queries)),
        )

    return ChatResponse(
        answer             = answer,
        steps              = final_state.get("steps", []),
        hallucination_free = final_state.get("hallucination") != "yes",
        answer_useful      = final_state.get("answer_useful") == "yes",
        session_id         = session_id,
        sources            = sources,
        sql_queries_used   = sql_queries,
    )


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    """Serve the admin upload UI."""
    admin_html = Path(__file__).parent / "static" / "admin.html"
    return HTMLResponse(content=admin_html.read_text())


@app.post("/admin/upload")
async def admin_upload(
    files: list[UploadFile] = File(default=[]),
    urls: str = Form(default=""),
):
    """Ingest uploaded PDF/DOCX files and/or URLs into ChromaDB."""
    from langchain_community.document_loaders import WebBaseLoader

    results = []
    errors  = []

    # ── Files ─────────────────────────────────────────────────────────────────
    for upload in files:
        if not upload.filename:
            continue
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in {".pdf", ".docx", ".xlsx", ".xls"}:
            errors.append(f"{upload.filename}: unsupported type (PDF, DOCX, XLSX, XLS only)")
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload.file, tmp)
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                from ingestion import _load_pdf
                loaded_docs = _load_pdf(tmp_path)
                count = ingest_documents(loaded_docs, label=upload.filename)
                results.append({"source": upload.filename, "chunks_added": count})

            elif suffix == ".docx":
                from langchain_community.document_loaders import Docx2txtLoader
                loaded_docs = Docx2txtLoader(tmp_path).load()
                count = ingest_documents(loaded_docs, label=upload.filename)
                results.append({"source": upload.filename, "chunks_added": count})

            elif suffix in {".xlsx", ".xls"}:
                # ── Dual ingestion: structured → SQLite AND text → ChromaDB ──
                # 1. Structured path: each sheet becomes a queryable SQLite table
                structured = ingest_excel_to_sqlite(tmp_path, source_label=upload.filename)
                for tbl in structured["tables"]:
                    results.append({
                        "source":       f"{upload.filename} / sheet: {tbl['sheet']}",
                        "chunks_added": 0,
                        "rows_ingested": tbl["rows"],
                        "sqlite_table":  tbl["table"],
                    })
                errors.extend(structured["errors"])

                # 2. Text path: embed each row as a Document for semantic search.
                # Uses pandas (already required for SQLite ingestion) — no extra deps.
                try:
                    import pandas as pd
                    from langchain_core.documents import Document as LCDocument
                    xl_file   = pd.ExcelFile(tmp_path)
                    text_docs = []
                    for sheet in xl_file.sheet_names:
                        df = pd.read_excel(xl_file, sheet_name=sheet)
                        if df.empty:
                            continue
                        for idx, row in df.iterrows():
                            content = "\n".join(
                                f"{col}: {val}"
                                for col, val in row.items()
                                if val is not None and str(val).strip() not in ("", "nan", "NaT")
                            )
                            if content.strip():
                                text_docs.append(LCDocument(
                                    page_content=content,
                                    metadata={
                                        "source": upload.filename,
                                        "sheet":  sheet,
                                        "row":    idx,
                                    },
                                ))
                    if text_docs:
                        count = ingest_documents(text_docs, label=upload.filename)
                        results.append({
                            "source":      f"{upload.filename} (text chunks)",
                            "chunks_added": count,
                        })
                except Exception as tex_err:
                    log.warning("Excel text embedding skipped: %s", tex_err)

        except Exception as e:
            errors.append(f"{upload.filename}: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── URLs ──────────────────────────────────────────────────────────────────
    url_list = [u.strip() for u in urls.replace(",", "\n").splitlines() if u.strip()]
    for url in url_list:
        try:
            loader = WebBaseLoader(url)
            docs   = loader.load()
            count  = ingest_documents(docs, label=url)
            results.append({"source": url, "chunks_added": count})
        except Exception as e:
            errors.append(f"{url}: {e}")

    if results:
        invalidate_retriever()  # rebuild retriever on next /chat request

    return {"ingested": results, "errors": errors}


@app.get("/admin/sources")
def admin_sources():
    """List all distinct sources currently ingested in ChromaDB."""
    return {"sources": list_sources()}


@app.delete("/admin/sources")
def admin_clear_sources():
    """Delete all documents from ChromaDB and all structured tables from SQLite."""
    deleted = clear_all_sources()
    invalidate_retriever()
    return {"deleted": deleted}


@app.get("/admin/suggested-questions")
async def suggested_questions():
    """Generate relevant questions based on actual content chunks from ChromaDB."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_chroma import Chroma

    # Check there's anything ingested
    sources = list_sources()
    if not sources:
        return {"questions": []}

    # Pull a sample of real text chunks from ChromaDB
    emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=config.OPENAI_API_KEY)
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        client=get_chroma_client(),
        embedding_function=emb,
    )
    total = vectorstore._collection.count()
    # Sample up to 20 chunks spread across the collection
    sample_size = min(20, total)
    raw = vectorstore._collection.get(limit=sample_size, include=["documents"])
    sample_text = "\n\n---\n\n".join(raw["documents"])

    #log.info("Generating suggested questions from %d sampled chunks (total=%d)", sample_size, total)

    llm    = ChatOpenAI(model=config.LLM_MODEL, temperature=0.4, openai_api_key=config.OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Based on the document excerpts below, generate 8 specific, "
         "diverse questions that a user might ask about this content. "
         "Make questions concrete and directly answerable from the material shown. "
         "Return only the questions, one per line, no numbering or bullets."),
        ("human", "Document excerpts:\n\n{content}"),
    ])
    result    = await (prompt | llm).ainvoke({"content": sample_text})
    questions = [q.strip() for q in result.content.strip().splitlines() if q.strip()]
    return {"questions": questions[:8]}


# ── Misc ──────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        emb   = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=config.OPENAI_API_KEY)
        vs    = Chroma(collection_name=config.CHROMA_COLLECTION_NAME,
                       client=get_chroma_client(),
                       embedding_function=emb)
        ready = vs._collection.count() > 0
    except Exception:
        ready = False
    return HealthResponse(status="ok", document=config.DOCUMENT_LABEL, vector_store_ready=ready)


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.get("/document-info")
def document_info():
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        emb   = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=config.OPENAI_API_KEY)
        vs    = Chroma(collection_name=config.CHROMA_COLLECTION_NAME,
                       client=get_chroma_client(),
                       embedding_function=emb)
        count = vs._collection.count()
    except Exception:
        count = 0

    return {
        "document_label":     config.DOCUMENT_LABEL,
        "vector_count":       count,
        "collection":         config.CHROMA_COLLECTION_NAME,
        "llm_model":          config.LLM_MODEL,
        "chunk_size":         config.CHUNK_SIZE,
        "retriever_k":        config.RETRIEVER_K,
        "web_search_enabled": bool(config.TAVILY_API_KEY),
    }


# ── Static UI ─────────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_ui():
        return FileResponse(str(STATIC_DIR / "index.html"))
