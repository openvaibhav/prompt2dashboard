from __future__ import annotations
from pathlib import Path
import logging
import re
import pandas as pd
from gemini_client import ask_gemini

logger = logging.getLogger(__name__)

_MAX_ROWS = 50

PROMPT_PATH = Path(__file__).parent.parent / "prompts/insight_prompt.txt"

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    _PROMPT_TEMPLATE = f.read()

def _dataframe_to_csv_snippet(df: pd.DataFrame):
    sample = df.head(_MAX_ROWS).copy()
    
    if sample.shape[1] > 8:
        sample = sample.iloc[:, :8]

    for col in sample.select_dtypes(include="float").columns:
        sample.loc[:, col] = sample[col].round(2)

    return sample.to_csv(index=False)


def _dataframe_stats(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return "No numeric columns."

    lines: list[str] = []
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        lines.append(
            f"{col}: count={len(series)}, "
            f"min={series.min():.2f}, max={series.max():.2f}, "
            f"mean={series.mean():.2f}, sum={series.sum():.2f}"
        )
    return "\n".join(lines) if lines else "No numeric statistics available."


def _parse_bullets(raw: str):
    lines = raw.strip().splitlines()
    bullets: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("•"):
            bullets.append(line)
            continue

        if re.match(r"^[-*]\s+", line):
            bullets.append("• " + re.sub(r"^[-*]\s+", "", line))
            continue

        if re.match(r"^\d+[.)]\s+", line):
            bullets.append("• " + re.sub(r"^\d+[.)]\s+", "", line))
            continue

        if len(line) > 10 and line not in bullets:
            bullets.append("• " + line)

    return bullets if bullets else ["• " + raw.strip()]

def generate_insights(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__!r}.")
    if df.empty:
        raise ValueError("Cannot generate insights for an empty DataFrame.")

    csv_snippet = _dataframe_to_csv_snippet(df)
    stats = _dataframe_stats(df)

    columns = ", ".join(df.columns)

    prompt = _PROMPT_TEMPLATE.format(
        csv_data=csv_snippet,
        stats=stats
    )

    prompt += f"\n\nColumns: {columns}"
    logger.debug(
        "Requesting insights from Gemini for DataFrame (%d rows, %d cols).",
        *df.shape,
    )

    raw_response = ask_gemini(prompt)
    logger.debug("Raw Gemini response:\n%s", raw_response)

    bullets = _parse_bullets(raw_response)

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        max_row = df.loc[df[col].idxmax()]

        auto_insight = f"• Highest {col.replace('_',' ')} observed: {max_row[col]}"
        bullets.insert(0, auto_insight)

    logger.debug("Parsed %d bullet points.", len(bullets))

    return bullets[:5]