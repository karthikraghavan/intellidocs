"""
graph/nodes/generate.py
────────────────────────
Node: generate

The final answer generation node.  It:
  1. Builds a prompt from the question + filtered context + chat history
     (including any SQL results from the sql_query node)
  2. Calls the LLM to generate a grounded answer
  3. Self-evaluates for hallucination (is the answer grounded in context?)
  4. Self-evaluates for usefulness (does the answer address the question?)
  5. Logs any SQL queries used to produce the answer

Both self-evaluation steps use structured output so the results are always
clean binary values that the graph router can act on.

Design notes
────────────
• When state["sql_result"] is populated (SQL path), it is injected as the
  primary context so the LLM synthesises an answer from real query results.
• For the SQL path the hallucination grader is seeded with the SQL result
  rather than raw document chunks, so grounding is checked against the
  actual query output.
• SQL queries logged to state["sql_queries_used"] are surfaced in the API
  response so callers can audit every query issued.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
from graph.state import GraphState

log = logging.getLogger(__name__)


# ── Schemas ───────────────────────────────────────────────────────────────────

class HallucinationGrade(BaseModel):
    binary_score: str = Field(
        description="'yes' if the answer is grounded in the provided context, 'no' if it contains made-up facts"
    )

class UsefulnessGrade(BaseModel):
    binary_score: str = Field(
        description="'yes' if the answer fully addresses the user's question, 'no' if it is evasive or incomplete"
    )


# ── Prompts ───────────────────────────────────────────────────────────────────

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert assistant for the document set: {document_label}.

Answer the user's question using ONLY the provided context.
Rules:
- Be specific and accurate.
- For text-based sources, cite the page number (e.g. [p. 42]) when available.
- For structured/SQL data, present numbers clearly (totals, counts, averages).
  If the context shows SQL results, summarise them in a human-friendly way.
- If the context doesn't contain enough information, say so clearly — do not guess.
- Keep your answer concise but complete.
- Where numbers or figures appear in the context, include them precisely.

Context:
{context}

Prior conversation (for reference):
{chat_history}""",
    ),
    ("human", "{question}"),
])

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a fact-checker. Determine whether the assistant's answer is "
        "fully grounded in the provided context. "
        "Score 'yes' if every factual claim in the answer can be traced to the context. "
        "Score 'no' if any claim is fabricated or cannot be verified from the context.",
    ),
    (
        "human",
        "Context:\n{context}\n\nAssistant answer:\n{generation}",
    ),
])

USEFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a quality assessor. Determine whether the assistant's answer "
        "actually addresses the user's question. Score 'yes' if it does, 'no' if not.",
    ),
    (
        "human",
        "Question: {question}\n\nAssistant answer: {generation}",
    ),
])


# ── LLM chains (built lazily) ─────────────────────────────────────────────────

_gen_chain = None
_hallucination_grader = None
_usefulness_grader = None


def _build_chains():
    global _gen_chain, _hallucination_grader, _usefulness_grader
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        openai_api_key=config.OPENAI_API_KEY,
    )
    _gen_chain            = GENERATION_PROMPT | llm | StrOutputParser()
    _hallucination_grader = HALLUCINATION_PROMPT | llm.with_structured_output(HallucinationGrade)
    _usefulness_grader    = USEFULNESS_PROMPT    | llm.with_structured_output(UsefulnessGrade)


def _format_docs(documents) -> str:
    """Concatenate document chunks into a single context string."""
    parts = []
    for i, doc in enumerate(documents, 1):
        page = doc.metadata.get("page", "?")
        src  = doc.metadata.get("source", "")
        header = f"[Source {i} | p.{page}]" if isinstance(page, int) else f"[Source {i} | {src}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def _format_history(chat_history: list) -> str:
    if not chat_history:
        return "None"
    lines = []
    for msg in chat_history[-6:]:  # last 3 exchanges to keep context concise
        role = msg.get("role", "user").capitalize()
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def generate(state: GraphState) -> GraphState:
    """
    LangGraph node — generate a grounded answer and self-evaluate it.

    Input  state keys: question, documents, sql_result, sql_queries_used, chat_history
    Output state keys: generation, hallucination, answer_useful, steps
    """
    global _gen_chain, _hallucination_grader, _usefulness_grader
    if _gen_chain is None:
        _build_chains()

    question         = state["question"]
    documents        = state.get("documents", [])
    sql_result       = state.get("sql_result", "")
    sql_queries_used = state.get("sql_queries_used", [])
    chat_history     = state.get("chat_history", [])

    # ── Log SQL queries used ──────────────────────────────────────────────────
    if sql_queries_used:
        log.info(
            "SQL queries used for this response:\n%s",
            "\n".join(f"  [{i+1}] {q}" for i, q in enumerate(sql_queries_used)),
        )

    # ── Build context ─────────────────────────────────────────────────────────
    # SQL result takes precedence; document chunks are appended as supplementary.
    context_parts = []
    if sql_result:
        context_parts.append(f"[Structured Data — SQL Query Results]\n{sql_result}")
    if documents:
        context_parts.append(f"[Document Text]\n{_format_docs(documents)}")

    if not context_parts:
        steps = state.get("steps", [])
        steps.append("generate_no_sources")
        return {
            "generation": (
                "I could not find any relevant information in the uploaded sources "
                "to answer your question. Please upload relevant documents via the "
                "admin page (/admin), or try rephrasing your question."
            ),
            "hallucination": "no",
            "answer_useful": "yes",
            "steps": steps,
        }

    context = "\n\n".join(context_parts)
    history = _format_history(chat_history)

    # Step 1 — Generate answer
    generation = _gen_chain.invoke({
        "document_label": config.DOCUMENT_LABEL,
        "context":        context,
        "chat_history":   history,
        "question":       question,
    })

    # Step 2 — Hallucination check
    hall_result: HallucinationGrade = _hallucination_grader.invoke({
        "context":    context,
        "generation": generation,
    })

    # Step 3 — Usefulness check
    use_result: UsefulnessGrade = _usefulness_grader.invoke({
        "question":   question,
        "generation": generation,
    })

    steps = state.get("steps", [])
    steps.append("generate")

    return {
        "generation":      generation,
        "hallucination":   hall_result.binary_score,
        "answer_useful":   use_result.binary_score,
        "sql_queries_used": sql_queries_used,
        "steps":            steps,
    }
