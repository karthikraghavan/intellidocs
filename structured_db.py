"""
structured_db.py
────────────────
SQLite-based structured data store for table data extracted from uploaded files
(Excel sheets, CSV-like tables in PDFs, etc.).

Public API
──────────
ingest_dataframe(df, table_name, source_label) → int
    Persist a pandas DataFrame as a SQLite table.  Returns row count.

list_structured_tables() → list[dict]
    Return metadata for every user table currently stored.

clear_all_structured_tables() → int
    Drop all user tables.  Returns count of tables dropped.

execute_sql(sql) → tuple[list[dict], list[str]]
    Run a read-only SQL query.  Returns (rows, column_names).

get_table_schemas() → dict[str, list[str]]
    Return {table_name: [column_names]} for schema-aware SQL generation.

get_table_sample(table_name, n) → list[dict]
    Return n example rows for context when generating SQL prompts.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

import config

log = logging.getLogger(__name__)


# ── Connection helper ─────────────────────────────────────────────────────────

def get_db_connection() -> sqlite3.Connection:
    """Open (and create if missing) the persistent SQLite database."""
    db_path = config.SQLITE_DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ── Name sanitisation ─────────────────────────────────────────────────────────

def _safe_table_name(name: str) -> str:
    """Convert an arbitrary label into a valid SQLite identifier."""
    # Strip file extension
    name = re.sub(r'\.[a-zA-Z0-9]+$', '', name)
    # Replace non-alphanumeric characters with underscore
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    # Collapse consecutive underscores
    name = re.sub(r'_+', '_', name).strip('_')
    # Must not start with a digit
    if name and name[0].isdigit():
        name = 't_' + name
    return (name or 'table').lower()[:50]


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_dataframe(df, table_name: str, source_label: str) -> int:
    """Persist a pandas DataFrame as a SQLite table.

    Args:
        df:           pandas DataFrame to store.
        table_name:   Desired table name (will be sanitised).
        source_label: Human-readable label (file name, sheet name, etc.).

    Returns:
        Number of rows inserted.
    """
    safe_name = _safe_table_name(table_name)

    conn = get_db_connection()
    try:
        # Metadata registry
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _table_metadata (
                table_name   TEXT PRIMARY KEY,
                source_label TEXT,
                row_count    INTEGER,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Normalise date/datetime columns to ISO strings so SQLite can filter them
        import pandas as pd
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d')

        # Write DataFrame (replace table if it already exists)
        df.to_sql(safe_name, conn, if_exists='replace', index=False)

        row_count = len(df)
        conn.execute(
            "INSERT OR REPLACE INTO _table_metadata "
            "(table_name, source_label, row_count) VALUES (?, ?, ?)",
            (safe_name, source_label, row_count),
        )
        conn.commit()
        log.info(
            "Stored %d rows into SQLite table '%s' (source: %s)",
            row_count, safe_name, source_label,
        )
        return row_count
    finally:
        conn.close()


# ── Listing ───────────────────────────────────────────────────────────────────

def list_structured_tables() -> list[dict]:
    """Return metadata about every user-created structured table."""
    conn = get_db_connection()
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name != '_table_metadata'"
        )
        tables = [row[0] for row in cur.fetchall()]

        meta_exists = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='_table_metadata'"
        ).fetchone()

        result = []
        for t in tables:
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            source_label = t
            if meta_exists:
                meta = conn.execute(
                    "SELECT source_label FROM _table_metadata WHERE table_name=?",
                    (t,),
                ).fetchone()
                if meta:
                    source_label = meta[0]
            result.append({"table": t, "source": source_label, "rows": row_count})
        return result
    except Exception:
        return []
    finally:
        conn.close()


# ── Deletion ──────────────────────────────────────────────────────────────────

def clear_all_structured_tables() -> int:
    """Drop every table (including metadata).  Returns count of tables dropped."""
    conn = get_db_connection()
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        for t in tables:
            conn.execute(f'DROP TABLE IF EXISTS "{t}"')
        conn.commit()
        log.info("Dropped %d structured table(s)", len(tables))
        return len(tables)
    except Exception:
        return 0
    finally:
        conn.close()


# ── Query execution ───────────────────────────────────────────────────────────

def execute_sql(sql: str) -> tuple[list[dict], list[str]]:
    """Execute a read SQL query and return (rows, column_names).

    Raises:
        Exception on SQL error.
    """
    conn = get_db_connection()
    try:
        cur = conn.execute(sql)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
        return rows, columns
    finally:
        conn.close()


# ── Schema introspection ──────────────────────────────────────────────────────

def get_table_schemas() -> dict[str, list[str]]:
    """Return {table_name: [column_names]} for all user tables."""
    conn = get_db_connection()
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name != '_table_metadata'"
        )
        tables = [row[0] for row in cur.fetchall()]
        schemas: dict[str, list[str]] = {}
        for t in tables:
            col_cur = conn.execute(f'PRAGMA table_info("{t}")')
            schemas[t] = [row[1] for row in col_cur.fetchall()]
        return schemas
    except Exception:
        return {}
    finally:
        conn.close()


def get_table_sample(table_name: str, n: int = 3) -> list[dict]:
    """Return n example rows for a table (used in SQL prompt context)."""
    conn = get_db_connection()
    try:
        cur = conn.execute(f'SELECT * FROM "{table_name}" LIMIT {int(n)}')
        columns = [desc[0] for desc in cur.description] if cur.description else []
        return [dict(zip(columns, row)) for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()
