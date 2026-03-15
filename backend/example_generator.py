from pathlib import Path
from gemini_client import ask_gemini
from functools import lru_cache

PROMPT_PATH = Path(__file__).parent.parent / "prompts/examples_prompt.txt"

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    _PROMPT_TEMPLATE = f.read()


def _prettify_col(col: str) -> str:
    return col.strip("_").replace("_", " ").strip()

def _base_examples(numeric_cols, categorical_cols):
    # Prettify column names for natural-sounding questions
    num = [_prettify_col(c) for c in numeric_cols]
    cat = [_prettify_col(c) for c in categorical_cols]
    
    examples = []
    if num:
        examples.append(f"Show total {num[0]}")
    if num and cat:
        examples.append(f"Show total {num[0]} by {cat[0]}")
    if num and cat:
        examples.append(f"Which {cat[0]} has the highest {num[0]}")

    if num and cat:
        examples.append(f"Top 10 {cat[0]} by {num[0]}")

    if len(num) >= 2:
        examples.append(f"Compare {num[0]} and {numeric_cols[1]}")

    if num and cat:
        examples.append(f"Show average {num[0]} by {cat[0]}")

    if categorical_cols:
        examples.append(f"Show number of records by {cat[0]}")

    if len(num) >= 2:
        examples.append(f"Show relationship between {num[0]} and {numeric_cols[1]}")

    if numeric_cols:
        examples.append(f"Show distribution of {num[0]}")

    if num and cat:
        examples.append(f"Show {num[0]} by {cat[-1]}")

    return examples[:10]


@lru_cache(maxsize=20)
def generate_examples(schema: str, numeric_cols: tuple, categorical_cols: tuple):
    base = _base_examples(list(numeric_cols), list(categorical_cols))
    
    try:
        base_text = "\n".join(base)
        prompt = _PROMPT_TEMPLATE.format(schema=schema, base_examples=base_text)
        raw = ask_gemini(prompt)
        examples = [
            line.strip("-•0123456789. ").strip().strip("_")
            for line in raw.splitlines()
            if line.strip() and len(line.strip()) > 5
        ]
    except Exception:
        examples = []

    if len(examples) < 4:
        examples += [e for e in base if e not in examples]

    return examples[:10]