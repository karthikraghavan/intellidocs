"""
graph/graph.py
───────────────
Assembles the Agentic RAG LangGraph workflow.

Graph topology
──────────────

                         START
                           │
                           ▼
                    route_question
                    /      |      \\
           sql_query  retrieve  web_search (disabled)
               │          │
               │     grade_documents
                \\        /
                 ▼      ▼
                generate  ──(retry)──┐
                    │                │
                    ▼                │
                   END ──────────────┘

Three routing paths:
  • sql_query   → runs SQL against the structured SQLite store → generate
  • vectorstore → retrieve from ChromaDB → grade → generate
  • web_search  → Tavily search → generate  (requires ENABLE_WEB_SEARCH=true in .env)

Retry cap
─────────
generate loops back to itself up to MAX_GENERATION_RETRIES times when the
self-evaluation detects hallucination or an unhelpful answer.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

import config
from graph.state import GraphState
from graph.nodes.retrieve import retrieve
from graph.nodes.grade_documents import grade_documents
from graph.nodes.generate import generate
from graph.nodes.sql_query import sql_query
from graph.nodes.web_search import web_search
from graph.nodes.router import (
    route_question,
    decide_after_routing,
    decide_after_generation,
)

MAX_GENERATION_RETRIES = 2


# ── Retry-capped generate wrapper ─────────────────────────────────────────────

def _generate_with_cap(state: GraphState) -> GraphState:
    """Wrap generate with a retry counter to prevent infinite loops."""
    retries = state.get("_gen_retries", 0)
    if retries >= MAX_GENERATION_RETRIES:
        steps = state.get("steps", [])
        steps.append("generate_gave_up")
        return {
            "generation": (
                "I was unable to produce a grounded answer after several attempts. "
                "The uploaded sources may not contain enough information for this question. "
                "Please try rephrasing, or upload additional relevant documents via the admin page."
            ),
            "hallucination": "no",
            "answer_useful": "yes",
            "steps": steps,
            "_gen_retries": retries + 1,
        }
    result = generate(state)
    result["_gen_retries"] = retries + 1
    return result


def _decide_after_generation_capped(state: GraphState) -> str:
    retries = state.get("_gen_retries", 0)
    if retries >= MAX_GENERATION_RETRIES:
        return END
    return decide_after_generation(state)


def _decide_after_grading(state: GraphState) -> str:
    """After grade_documents: go to web_search if enabled and flagged, else generate."""
    if config.ENABLE_WEB_SEARCH and state.get("web_search"):
        return "web_search"
    return "generate"


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node("route_question",  route_question)
    builder.add_node("retrieve",        retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("sql_query",       sql_query)
    builder.add_node("generate",        _generate_with_cap)
    if config.ENABLE_WEB_SEARCH:
        builder.add_node("web_search", web_search)

    # Entry point
    builder.set_entry_point("route_question")

    # Routing edges from route_question
    if config.ENABLE_WEB_SEARCH:
        web_search_target = "web_search"
    else:
        web_search_target = "retrieve"  # fall back when disabled

    builder.add_conditional_edges(
        "route_question",
        decide_after_routing,
        {
            "sql_query":  "sql_query",
            "retrieve":   "retrieve",
            "web_search": web_search_target,
        },
    )

    # Vectorstore path: retrieve → grade → (web_search | generate)
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        _decide_after_grading,
        {"web_search": "web_search", "generate": "generate"}
        if config.ENABLE_WEB_SEARCH
        else {"generate": "generate"},
    )

    if config.ENABLE_WEB_SEARCH:
        builder.add_edge("web_search", "generate")

    # SQL path: sql_query → retrieve → grade_documents → generate
    # Hybrid approach: always follow SQL with a vectorstore retrieval so that:
    #   • PDF text data (employees, revenue, etc.) supplements SQL results
    #   • If SQL finds nothing (wrong table, empty result), vector docs take over
    #   • Excel text columns are also retrievable via semantic search
    builder.add_edge("sql_query", "retrieve")

    # Generate retry loop
    builder.add_conditional_edges(
        "generate",
        _decide_after_generation_capped,
        {"generate": "generate", END: END},
    )

    return builder.compile()


graph = build_graph()
