from __future__ import annotations
import sqlite3
import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_TABLE = "sales"


def _extract_table_name(sql_query: str):

    match = re.search(r"\bFROM\s+([`\"\[]?(\w+)[`\"\]]?)", sql_query, re.IGNORECASE)
    return match.group(2) if match else None


def _register_dataframe(conn: sqlite3.Connection, df: pd.DataFrame, table_name: str):
    df = df.copy()
    for col in df.columns:
        if (
            pd.api.types.is_datetime64_any_dtype(df[col]) or
            pd.api.types.is_datetime64_dtype(df[col]) or
            str(df[col].dtype) == "datetime64[ns]" or
            hasattr(df[col], 'dt')
        ):
            df.loc[:, col] = df[col].astype(str)
        elif df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if len(sample) > 0 and isinstance(sample.iloc[0], pd.Timestamp):
                df.loc[:, col] = df[col].astype(str)
    df.to_sql(table_name, conn, if_exists="replace", index=False)


def execute_query(
    df: pd.DataFrame,
    sql_query: str,
    *,
    table_name: Optional[str] = None,
):

    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__!r}.")
    if df.empty:
        raise ValueError("df is empty — there is no data to query.")

    sql_query = sql_query.strip()
    if not sql_query:
        raise ValueError("sql_query must not be empty.")

    sql_query = sql_query.rstrip(";")
    
    if "timestamp" in sql_query.lower() and "date" in df.columns:
        sql_query = sql_query.replace("timestamp", "date")
    
    if "limit" not in sql_query.lower():
        sql_query += " LIMIT 1000"

    _WRITE_PATTERN = re.compile(
        r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|REPLACE|TRUNCATE)\b",
        re.IGNORECASE,
    )
    if _WRITE_PATTERN.match(sql_query):
        first_word = sql_query.split()[0].upper()
        raise PermissionError(
            f"Write/DDL operation '{first_word}' is not permitted. "
            "Only SELECT (and WITH / EXPLAIN) queries are allowed."
        )

    resolved_table = table_name or _extract_table_name(sql_query) or _DEFAULT_TABLE
    logger.debug(
        "Registering DataFrame as table '%s' (%d rows).", resolved_table, len(df)
    )

    try:
        with sqlite3.connect(":memory:") as conn:
            _register_dataframe(conn, df, resolved_table)
            _register_dataframe(conn, df, "uploaded_dataset")
            _register_dataframe(conn, df, "data")

            logger.debug("Executing SQL:\n%s", sql_query)
            result = pd.read_sql_query(sql_query, conn)
            result.columns = [
                c.replace("(", "_")
                .replace(")", "")
                .replace("*", "all")
                .replace(" ", "_")
                for c in result.columns
            ]
            result = result.dropna(how="all")
            
            if result.empty:
                raise ValueError(
                    "The query returned no results. "
                    "Try broadening your question or check if the data matches your filter."
                )
            

    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            f"SQL execution failed — OperationalError: {exc}\n"
            f"Query was:\n{sql_query}"
        ) from exc
    except sqlite3.ProgrammingError as exc:
        raise RuntimeError(
            f"SQL execution failed — ProgrammingError: {exc}\n"
            f"Query was:\n{sql_query}"
        ) from exc
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(
            f"SQL execution failed — DatabaseError: {exc}\n" f"Query was:\n{sql_query}"
        ) from exc

    if result.empty:
        logger.warning("Query returned 0 rows.")
    else:
        logger.debug("Query returned %d rows, %d columns.", *result.shape)

    return result