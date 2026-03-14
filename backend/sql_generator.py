from __future__ import annotations
from pathlib import Path
import re
import logging
from gemini_client import ask_gemini
logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts/sql_prompt.txt"

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    _PROMPT_TEMPLATE = f.read()

def _clean_sql(raw: str):
    text = raw.strip()

    fenced = re.match(r"^```(?:sql|SQL)?\s*\n?(.*?)```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
        return text

    inline = re.match(r"^`([^`]+)`$", text)
    if inline:
        text = inline.group(1).strip()
        return text

    sql_start = re.search(
        r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|DROP|ALTER|EXPLAIN)\b",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if sql_start and sql_start.start() > 0:
        text = text[sql_start.start():].strip()

    return text


def _validate_sql(sql: str):

    if sql.strip().upper() == "CANNOT_ANSWER":
        raise ValueError("The question cannot be answered using the dataset schema.")

    sql_keywords = (
        "SELECT", "INSERT", "UPDATE", "DELETE",
        "WITH", "CREATE", "DROP", "ALTER", "EXPLAIN",
    )

    tokens = sql.split()
    first_token = tokens[0].upper() if tokens else ""

    if first_token not in sql_keywords:
        raise ValueError(
            f"Gemini did not return a valid SQL statement. "
            f"Got: {sql[:120]!r}"
        )

def generate_sql(user_query: str, schema: str):
    
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
    schema=schema,
    user_query=user_query
    )

    prompt += f"\n\nAvailable columns: {columns_str}"
    logger.debug("===== SQL PROMPT SENT TO GEMINI =====")
    logger.debug(prompt)
    logger.debug("=====================================")

    raw_response = ask_gemini(prompt)
    
    raw_response = ask_gemini(prompt)
    
    logger.debug("Raw Gemini response:\n%s", raw_response)

    sql = _clean_sql(raw_response)
    _validate_sql(sql)

    logger.debug("Generated SQL:\n%s", sql)
    return sql