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
  • web_search  → (disabled; falls back to vectorstore)

Retry cap
─────────
generate loops back to itself up to MAX_GENERATION_RETRIES times when the
self-evaluation detects hallucination or an unhelpful answer.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.nodes.retrieve import retrieve
from graph.nodes.grade_documents import grade_documents
from graph.nodes.generate import generate
from graph.nodes.sql_query import sql_query
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
    """After grade_documents: always go to generate (web search disabled)."""
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

    # Entry point
    builder.set_entry_point("route_question")

    # Routing edges from route_question
    builder.add_conditional_edges(
        "route_question",
        decide_after_routing,
        {
            "sql_query": "sql_query",
            "retrieve":  "retrieve",
            # web_search is disabled — fall back to retrieve
            "web_search": "retrieve",
        },
    )

    # Vectorstore path: retrieve → grade → generate
    builder.add_edge("retrieve", "grade_documents")
    builder.add_edge("grade_documents", "generate")

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
