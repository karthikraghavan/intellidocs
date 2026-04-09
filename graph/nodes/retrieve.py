"""
graph/nodes/retrieve.py
────────────────────────
Node: retrieve

Queries the ChromaDB vector store with the user's question and populates
state["documents"] with the top-k most semantically similar chunks.

Design notes
────────────
• Similarity search uses cosine distance on OpenAI embeddings.
• k is read from config (default 5) — increase for broader coverage,
  decrease for more focused context (and lower token cost).
• The retriever is cached in a module-level variable; call
  invalidate_retriever() after adding new documents so the next request
  rebuilds the retriever against the updated ChromaDB index.
• Metadata (source page number) is preserved on every Document so the
  generation node can cite page references in its answer.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import config
from graph.state import GraphState


_retriever = None   # invalidatable singleton — NOT lru_cache


def get_retriever():
    """
    Return the cached retriever, building it lazily on first call.
    Call invalidate_retriever() to force a rebuild after uploads.
    """
    global _retriever
    if _retriever is None:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=config.OPENAI_API_KEY,
        )
        vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
        _retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.RETRIEVER_K},
        )
    return _retriever


def invalidate_retriever():
    """Force the retriever to be rebuilt on the next request."""
    global _retriever
    _retriever = None


def retrieve(state: GraphState) -> GraphState:
    """
    LangGraph node — retrieve relevant chunks from the vector store.

    Input  state keys: question
    Output state keys: documents, steps
    """
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)

    steps = state.get("steps", [])
    steps.append("retrieve")

    return {"documents": documents, "steps": steps}
