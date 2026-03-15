from __future__ import annotations
from pathlib import Path
import re
import logging
from gemini_client import ask_gemini

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts/sql_prompt.txt"

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    _PROMPT_TEMPLATE = f.read()

SQLITE_RESERVED = {
    "cast", "case", "check", "column", "constraint", "create", "cross",
    "default", "delete", "distinct", "drop", "else", "end", "escape",
    "except", "exists", "explain", "fail", "for", "foreign", "from",
    "full", "glob", "group", "having", "if", "ignore", "immediate",
    "index", "inner", "insert", "instead", "intersect", "into", "is",
    "isnull", "join", "key", "left", "like", "limit", "match", "natural",
    "not", "notnull", "null", "of", "offset", "on", "or", "order",
    "outer", "plan", "pragma", "primary", "query", "raise", "recursive",
    "references", "regexp", "reindex", "release", "rename", "replace",
    "restrict", "right", "rollback", "row", "savepoint", "select", "set",
    "table", "temp", "temporary", "then", "to", "transaction", "trigger",
    "union", "unique", "update", "using", "vacuum", "values", "view",
    "virtual", "when", "where", "with", "without"
}

def _quote_reserved_columns(sql: str, columns: list[str]) -> str:
    for col in columns:
        needs_quoting = (
            col.lower() in SQLITE_RESERVED or
            bool(re.match(r'^\d', col)) or
            bool(re.search(r'\s', col))
        )
        if needs_quoting:
            sql = re.sub(
                rf'(?<!["\'\w]){re.escape(col)}(?!["\'\w])',
                f'"{col}"',
                sql,
                flags=re.IGNORECASE
            )
    return sql

CANNOT_ANSWER_SIGNALS = [
    "CANNOT_ANSWER",
    "cannot_answer",
    "I cannot",
    "I don't know", 
    "not possible",
    "cannot be answered",
    "unable to answer",
]

def _is_truncated_sql(sql: str) -> bool:
    """Detect if SQL was cut off mid-generation."""
    sql = sql.strip().rstrip(";").strip()
    
    if sql.count("(") != sql.count(")"):
        return True
    
    last_token = sql.split()[-1].upper() if sql.split() else ""
    bad_endings = {"SELECT", "FROM", "WHERE", "AND", "OR", "BY", "ON", "SET", "WITH", "SUM", "AVG", "COUNT", "MAX", "MIN", "AS", "CASE", "WHEN", "THEN", "ELSE"}
    if last_token in bad_endings:
        return True
    
    return False


def _clean_sql(raw: str):
    text = raw.strip()
    
    if any(signal.lower() in text.lower() for signal in CANNOT_ANSWER_SIGNALS):
        raise ValueError(
            "I couldn't answer that question with the available data. "
            "Try rephrasing or ask about specific columns in your dataset."
        )
    
    text = text.split(";")[0] + ";"
    
    fenced = re.match(r"^```(?:sql|SQL)?\s*\n?(.*?)```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        inline = re.match(r"^`([^`]+)`$", text)
        if inline:
            text = inline.group(1).strip()
        else:
            sql_start = re.search(
                r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|DROP|ALTER|EXPLAIN)\b",
                text,
                re.IGNORECASE | re.MULTILINE,
            )
            if sql_start and sql_start.start() > 0:
                text = text[sql_start.start():].strip()

    if _is_truncated_sql(text):
        raise ValueError(
            "That query was too complex to generate fully. "
            "Try breaking it into simpler questions — e.g. ask about monthly views first, then filter by region separately."
        )

    return text


def _validate_sql(sql: str):
    sql_keywords = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "DROP", "ALTER", "EXPLAIN")
    tokens = sql.split()
    first_token = tokens[0].upper() if tokens else ""
    
    if first_token not in sql_keywords:
        raise ValueError(
            "I couldn't generate a valid query for that question. "
            "Try rephrasing or ask about specific columns in your dataset."
        )


def generate_sql(data_summary: str, user_query: str, schema: str):

    user_query = user_query.strip()
    schema = schema.strip()

    if not user_query:
        raise ValueError("user_query must not be empty.")
    if not schema:
        raise ValueError("schema must not be empty.")

    columns = []
    for line in schema.split("\n"):
        if line.startswith("-"):
            col = line.split("(")[0].replace("-", "").strip()
            columns.append(col)

    columns_str = ", ".join(columns)

    prompt = _PROMPT_TEMPLATE.format(
        summary = data_summary, schema=schema, user_query=user_query, columns=columns_str
    )

    logger.debug("===== SQL PROMPT SENT TO GEMINI =====")
    logger.debug(prompt)
    logger.debug("=====================================")

    raw_response = ask_gemini(prompt)

    logger.debug("Raw Gemini response:\n%s", raw_response)

    sql = _clean_sql(raw_response)
    _validate_sql(sql)
    
    sql = _quote_reserved_columns(sql, columns)

    logger.debug("Generated SQL:\n%s", sql)
    return sql
