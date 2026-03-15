from pathlib import Path
from gemini_client import ask_gemini
from functools import lru_cache
import re

PROMPT_PATH = Path(__file__).parent.parent / "prompts/examples_prompt.txt"

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    _PROMPT_TEMPLATE = f.read()

def _is_id_column(col: str):
    """Returns True if column is likely an identifier, not a metric."""
    col = col.lower()
    id_patterns = [
        r"^id$", r"_id$", r"^id_", r"row_?id", r"^index$",
        r"_key$", r"^key_", r"_code$", r"postal", r"zip",
        r"phone", r"order_?num", r"invoice", r"transaction_?id"
    ]
    return any(re.search(p, col) for p in id_patterns)

def _prettify_col(col: str):
    col = col.strip("_").replace("_", " ").strip()
    for suffix in [" sec", " id", " count", " score", " amt", " num"]:
        if col.endswith(suffix) and len(col) > len(suffix) + 3:
            col = col[: -len(suffix)]
    return col.strip()

def _is_datetime_column(col: str):
    col = col.lower()
    datetime_keywords = ["date", "time", "timestamp", "datetime", "created", "published", "day", "month", "year"]
    return any(k in col for k in datetime_keywords)

def _base_examples(numeric_cols, categorical_cols):
    if not numeric_cols and not categorical_cols:
        return [
            "Show me the data",
            "How many records are there?",
            "Show all columns",
        ]

    real_numeric = [c for c in numeric_cols if not _is_id_column(c)]
    
    datetime_cols = [c for c in categorical_cols if _is_datetime_column(c)]
    real_categorical = [
        c for c in categorical_cols 
        if not _is_id_column(c) and not _is_datetime_column(c)
    ]
    
    if not real_numeric and numeric_cols:
        real_numeric = numeric_cols[:2]
    if not real_categorical and categorical_cols:
        real_categorical = [c for c in categorical_cols if not _is_datetime_column(c)][:2]

    num = [_prettify_col(c) for c in real_numeric]
    cat = [_prettify_col(c) for c in real_categorical]
    dt = [_prettify_col(c) for c in datetime_cols]

    examples = []
    if num:
        examples.append(f"What is the total {num[0]}?")
    if num and cat:
        examples.append(f"Show the total {num[0]} for each {cat[0]}")
    if num and cat:
        examples.append(f"Which {cat[0]} has the highest {num[0]}?")
    if num and cat:
        examples.append(f"Top 10 {cat[0]} by {num[0]}")
    if len(num) >= 2:
        examples.append(f"Compare {num[0]} and {num[1]}")
    if num and cat:
        examples.append(f"Show average {num[0]} by {cat[0]}")
    if cat:
        examples.append(f"How many records are there per {cat[0]}?")
    if len(num) >= 2:
        examples.append(f"What is the relationship between {num[0]} and {num[1]}?")
    if num and dt:
        examples.append(f"How has {num[0]} changed over {dt[0]}?")
    if num and len(cat) >= 2:
        examples.append(f"Show {num[0]} broken down by {cat[1]}")

    return examples[:10]


@lru_cache(maxsize=20)
def generate_examples(data_summary: str, schema: str, numeric_cols: tuple, categorical_cols: tuple):
    base = _base_examples(list(numeric_cols), list(categorical_cols))
    
    try:
        base_text = "\n".join(base)
        prompt = _PROMPT_TEMPLATE.format(summary = data_summary, schema=schema, base_examples=base_text)
        raw = ask_gemini(prompt)
        examples = [
            line.strip("-•0123456789. ").strip().strip("_")
            for line in raw.splitlines()
            if line.strip() and len(line.strip()) > 5
        ]
    except Exception:
        examples = []

    if len(examples) < 10:
        examples += [e for e in base if e not in examples]

    return examples[:10]