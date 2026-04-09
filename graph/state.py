"""
graph/state.py
──────────────
Defines GraphState — the single shared data structure that flows through
every node in the LangGraph pipeline.

Think of it as the "working memory" of one conversation turn:
  • Every node receives the full state and returns a partial update.
  • LangGraph merges updates automatically.
  • Adding new fields here (and handling them in the relevant nodes) is
    all that's needed to extend the agent's capabilities.

Field guide
───────────
question       Raw question from the user (never mutated after entry).
documents      List of LangChain Document objects retrieved from the vector
               store (or web search). Replaced in-place after grade_documents
               filters out irrelevant ones.
generation     Final answer string produced by the generate node.
web_search     Flag set by grade_documents when too many retrieved chunks are
               irrelevant — triggers the web_search node instead of generate.
hallucination  "yes" | "no" — set by generate after self-evaluation.
answer_useful  "yes" | "no" — set by generate after evaluating whether the
               answer actually addresses the question.
steps          Ordered list of node names visited this turn (for debugging
               and the UI trace panel).
chat_history   Accumulated list of {role, content} dicts for multi-turn
               conversation memory.  Passed into the generation prompt so the
               LLM can refer to prior exchanges.
"""

from __future__ import annotations
from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    # ── Core RAG fields ──────────────────────────────────────────────────────
    question: str
    documents: List[Document]
    generation: str

    # ── Routing / self-evaluation flags ──────────────────────────────────────
    web_search: bool        # True  → fall back to Tavily web search
    route_type: str         # "vectorstore" | "sql_query" | "web_search"
    hallucination: str      # "yes" | "no"
    answer_useful: str      # "yes" | "no"

    # ── Structured data (SQL) ─────────────────────────────────────────────────
    sql_result: str               # Formatted SQL query results (empty string when no SQL was run)
    sql_queries_used: List[str]   # All SQL queries executed this turn

    # ── Observability ────────────────────────────────────────────────────────
    steps: List[str]        # e.g. ["route", "retrieve", "grade", "generate"]

    # ── Multi-turn memory ────────────────────────────────────────────────────
    chat_history: List[dict]  # [{"role": "user"|"assistant", "content": "..."}]

    # ── Internal counters ────────────────────────────────────────────────────
    _gen_retries: int         # incremented by _generate_with_cap to cap loops
