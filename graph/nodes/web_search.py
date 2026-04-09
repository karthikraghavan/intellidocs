"""
graph/nodes/web_search.py
──────────────────────────
Node: web_search

Fallback node that fires when the local vector store doesn't have enough
relevant content.  Uses Tavily Search API to retrieve up-to-date web results
and appends them to state["documents"] so the generate node can use them
alongside (or instead of) the vector-store chunks.

Design notes
────────────
• Tavily is chosen over raw Google because it returns cleaned, structured
  snippets rather than raw HTML — no scraping needed.
• If TAVILY_API_KEY is not set the node gracefully skips (warns in logs)
  so the chatbot degrades gracefully to vector-store-only mode.
• Web results are wrapped in LangChain Documents with source metadata set
  to the URL — this lets the generation node cite web sources the same way
  it cites PDF page numbers.
• We limit to 3 web results (num_results=3) to keep context concise.
  Increase to 5 for richer coverage at the cost of more tokens.
"""

from __future__ import annotations
import logging

from langchain_core.documents import Document

import config
from graph.state import GraphState

log = logging.getLogger(__name__)


def web_search(state: GraphState) -> GraphState:
    """
    LangGraph node — supplement context with live web search results.

    Input  state keys: question, documents
    Output state keys: documents (extended with web results), steps
    """
    if not config.TAVILY_API_KEY:
        log.warning(
            "TAVILY_API_KEY not set — skipping web search. "
            "Add it to .env to enable the Adaptive RAG fallback."
        )
        steps = state.get("steps", [])
        steps.append("web_search_skipped")
        return {"steps": steps}

    # Lazy import so Tavily is optional at install time
    from tavily import TavilyClient  # type: ignore

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    question = state["question"]

    response = client.search(query=question, max_results=3)

    web_docs = [
        Document(
            page_content=r.get("content", ""),
            metadata={"source": r.get("url", "web"), "title": r.get("title", "")},
        )
        for r in response.get("results", [])
    ]

    # Merge with any surviving vector-store docs
    existing = state.get("documents", [])
    merged   = existing + web_docs

    steps = state.get("steps", [])
    steps.append("web_search")

    return {"documents": merged, "steps": steps}
