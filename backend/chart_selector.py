from __future__ import annotations
from pathlib import Path
import re
import logging
from gemini_client import ask_gemini

logger = logging.getLogger(__name__)

VALID_CHART_TYPES: frozenset[str] = frozenset({"bar", "line", "pie", "scatter"})

_DEFAULT_CHART = "bar"

PROMPT_PATH = Path(__file__).parent.parent / "prompts/chart_prompt.txt"

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    _PROMPT_TEMPLATE = f.read()

def _parse_chart_type(raw: str):
    
    cleaned = raw.strip().lower()

    if cleaned in VALID_CHART_TYPES:
        return cleaned

    cleaned = re.sub(r"[`*_\-\.\,]", "", cleaned).strip()
    if cleaned in VALID_CHART_TYPES:
        return cleaned

    for chart in VALID_CHART_TYPES:
        if re.search(rf"\b{chart}\b", cleaned):
            logger.debug(
                "Chart type '%s' found by substring scan in response: %r", chart, raw
            )
            return chart

    logger.warning(
        "Could not parse a valid chart type from Gemini response %r. "
        "Falling back to '%s'.",
        raw,
        _DEFAULT_CHART,
    )
    return _DEFAULT_CHART

def choose_chart_type(user_query: str, dataframe_columns: list[str]):

    user_query = user_query.lower()

    if any(word in user_query for word in ["trend", "over time", "monthly", "daily"]):
        return "line"

    if any(word in user_query for word in ["share", "percentage", "split"]):
        return "pie"

    if "vs" in user_query or "correlation" in user_query:
        return "scatter"

    if len(dataframe_columns) <= 2:
        return "bar"

    if any("date" in col.lower() or "time" in col.lower() for col in dataframe_columns):
        return "line"

    columns_str = ", ".join(dataframe_columns)
    prompt = _PROMPT_TEMPLATE.format(user_query=user_query, columns=columns_str)

    raw_response = ask_gemini(prompt)
    return _parse_chart_type(raw_response)

def generate_dashboard_charts(df, user_query):
    charts = []

    columns = list(df.columns)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    main_chart = choose_chart_type(user_query, columns)
    charts.append(main_chart)

    if categorical_cols and numeric_cols:
        charts.append("bar")

    if len(categorical_cols) == 1 and len(numeric_cols) == 1:
        charts.append("pie")

    if len(numeric_cols) >= 2:
        charts.append("scatter")

    return list(dict.fromkeys(charts))[:3]