"""
graph/nodes/router.py
──────────────────────
Node: route_question  (entry-point router)
Edge functions: decide_after_routing, decide_after_grading, decide_after_generation

Routing topology
────────────────
                 ┌──────────────┐
    question ──▶ │ route_question│
                 └──────┬───────┘
           sql_query ◀──┼──▶ vectorstore ──▶ retrieve ──▶ grade_documents
                        │
                        └──▶ web_search

route_question classifies the question into one of three paths:
  • "sql_query"   → structured tabular data question (aggregation, filtering, etc.)
  • "vectorstore" → text-based document Q&A from ChromaDB
  • "web_search"  → live / general knowledge outside the uploaded documents

decide_after_grading
─────────────────────
After grade_documents runs, this edge checks state["web_search"]:
  • True  → web_search node
  • False → generate node

decide_after_generation
────────────────────────
After generate runs, checks hallucination and usefulness:
  • hallucination == "no" AND answer_useful == "yes" → END (happy path)
  • hallucination == "yes"                           → re-generate (loop)
  • answer_useful == "no"                            → re-generate (loop)
The loop is capped in graph.py to prevent infinite cycles.
"""

from __future__ import annotations
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from pydantic import BaseModel, Field

import config
from graph.state import GraphState
from structured_db import get_table_schemas


# ── Structured output for routing decision ────────────────────────────────────

class RouteDecision(BaseModel):
    datasource: Literal["vectorstore", "web_search", "sql_query"] = Field(
        description=(
            "Route to 'sql_query' for questions that require aggregation, "
            "filtering, or SQL operations over tabular/structured data (e.g. totals, "
            "averages, counts, date-range filters on rows). "
            "Route to 'vectorstore' for text-based document questions. "
            "Route to 'web_search' for live or general knowledge outside the documents."
        )
    )


ROUTE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query router for a document Q&A system.
The knowledge base contains documents that have been uploaded by the user.
The current document set is labelled: {document_label}

Structured (tabular) tables available in the database:
{structured_tables}

Route to 'sql_query' ONLY when ALL of these are true:
1. Structured tables are listed above (not "none")
2. The question is explicitly about aggregating or filtering rows IN THOSE TABLES
   (e.g. SUM, total, average, count, min, max, grouped by, filtered by date/category)
3. The column names in the question or its intent clearly match a column in the listed tables
   (e.g. "total rent" → matches a table with a Category or Amount column)

Examples that SHOULD go to sql_query (assuming matching tables exist):
- "What is my total rent expense this year?"
- "How much did I spend on food in January?"
- "Show all healthcare transactions over $500"
- "What is the average monthly spending by category?"

Route to 'vectorstore' when:
- The question is about narrative, descriptions, or text in any document (PDF, DOCX, Excel)
- The question asks about numbers mentioned IN A PDF (e.g. employee count, revenue from annual report)
  because those numbers are stored as text, not in structured tables
- The question requires explanation, summarisation, or comparison of document content
- No structured tables are available
- The question involves numbers/figures but they come from PDF text, not spreadsheet rows
- When in doubt — prefer vectorstore

Route to 'web_search' when:
- Information requires live, real-time, or current external data not in any uploaded document
- General knowledge clearly unrelated to the uploaded content""",
    ),
    ("human", "{question}"),
])


_router = None  # lazy singleton


def _build_router():
    global _router
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
    )
    _router = ROUTE_PROMPT | llm.with_structured_output(RouteDecision)


# ── Node ──────────────────────────────────────────────────────────────────────

def route_question(state: GraphState) -> GraphState:
    """
    LangGraph node — classify the question and set the routing intent.
    The actual branching is done by the edge function decide_after_routing().
    """
    global _router
    if _router is None:
        _build_router()

    # Describe available structured tables for the routing prompt
    schemas = get_table_schemas()
    if schemas:
        structured_tables = "\n".join(
            f"  - {tbl}: columns {', '.join(cols)}"
            for tbl, cols in schemas.items()
        )
    else:
        structured_tables = "  (none — no structured tables ingested)"

    decision: RouteDecision = _router.invoke({
        "question":          state["question"],
        "document_label":    config.DOCUMENT_LABEL,
        "structured_tables": structured_tables,
    })

    steps = state.get("steps", [])
    steps.append(f"route→{decision.datasource}")

    return {
        "route_type": decision.datasource,
        "web_search": decision.datasource == "web_search",
        "steps":      steps,
    }


# ── Edge functions (return the name of the next node) ─────────────────────────

def decide_after_routing(state: GraphState) -> str:
    """Edge: after route_question → sql_query | retrieve | web_search."""
    route = state.get("route_type", "vectorstore")
    if route == "sql_query":
        return "sql_query"
    if route == "web_search":
        return "web_search"
    return "retrieve"


def decide_after_grading(state: GraphState) -> str:
    """Edge: after grade_documents → web_search OR generate."""
    return "web_search" if state.get("web_search") else "generate"


def decide_after_generation(state: GraphState) -> str:
    """
    Edge: after generate → END (success) OR generate (retry).

    Loops back to 'generate' if the LLM hallucinated or gave an unhelpful answer.
    graph.py caps retries so this never loops indefinitely.
    """
    if state.get("hallucination") == "yes":
        return "generate"  # retry — answer wasn't grounded
    if state.get("answer_useful") == "no":
        return "generate"  # retry — answer didn't address the question
    return END
