"""
graph/nodes/grade_documents.py
───────────────────────────────
Node: grade_documents

After retrieval, not every chunk may actually be relevant to the question.
This node uses an LLM with structured output to score each document
("yes" / "no") and filter out noise.

If fewer than half the chunks survive grading, web_search is set to True
so the router knows to supplement with a Tavily web search.

Design notes
────────────
• Structured output (Pydantic model) forces the LLM to return a clean
  binary decision rather than free text — no regex parsing needed.
• The grading prompt is deliberately minimal: just question + chunk content.
  Keeping it tight reduces latency and cost.
• Documents are filtered in-place so downstream nodes always work with a
  clean, relevant context window.
• The 50% threshold for triggering web search is a tuneable heuristic.
  Lower it (e.g., any irrelevant doc) for stricter quality; raise it
  (e.g., 80%) to rely on web search less.
"""

from __future__ import annotations
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
from graph.state import GraphState


# ── Structured output schema ──────────────────────────────────────────────────

class GradeDocument(BaseModel):
    """Binary relevance score for a retrieved document chunk."""
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question, 'no' otherwise"
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a relevance grader. Assess whether a retrieved document chunk "
        "contains information useful for answering the user's question. "
        "Be lenient — if the chunk has *any* relevant facts, score it 'yes'. "
        "Score 'no' only if it is completely off-topic.",
    ),
    (
        "human",
        "Question: {question}\n\nDocument chunk:\n{document}",
    ),
])


def _build_grader():
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
    )
    return GRADE_PROMPT | llm.with_structured_output(GradeDocument)


_grader = None  # lazy singleton


def grade_documents(state: GraphState) -> GraphState:
    """
    LangGraph node — filter retrieved documents by relevance.

    Input  state keys: question, documents
    Output state keys: documents (filtered), web_search, steps
    """
    global _grader
    if _grader is None:
        _grader = _build_grader()

    question  = state["question"]
    documents = state.get("documents", [])

    relevant = []
    for doc in documents:
        result: GradeDocument = _grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        if result.binary_score == "yes":
            relevant.append(doc)

    steps = state.get("steps", [])
    steps.append("grade_documents")

    return {
        "documents": relevant,
        "web_search": False,   # external search disabled — answers come only from ingested sources
        "steps": steps,
    }
