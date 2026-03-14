from __future__ import annotations
import logging
import re
from pathlib import Path
from gemini_client import ask_gemini
logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts/followup_prompt.txt"

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    _PROMPT_TEMPLATE = f.read()

def _clean_response(raw: str):
    text = raw.strip()

    preamble_pattern = re.compile(
        r"^(standalone query|rewritten query|here is the rewritten query)\s*[:\-–—]\s*",
        re.IGNORECASE,
    )
    text = preamble_pattern.sub("", text).strip()

    if len(text) >= 2 and text[0] in ('"', "'", "\u201c") and text[-1] in ('"', "'", "\u201d"):
        text = text[1:-1].strip()

    return text

def resolve_followup(previous_query: str, followup_query: str):
    previous_query = (previous_query or "").strip()
    followup_query = (followup_query or "").strip()

    if not followup_query:
        raise ValueError("followup_query must not be empty.")

    if not previous_query:
        logger.debug(
            "No previous query provided; returning follow-up as-is: %r", followup_query
        )
        return followup_query

    prompt = _PROMPT_TEMPLATE.format(
        previous_query=previous_query,
        followup_query=followup_query,
    )

    logger.debug(
        "Resolving follow-up — previous: %r | followup: %r",
        previous_query,
        followup_query,
    )

    raw_response = ask_gemini(prompt)
    logger.debug("Raw Gemini response: %r", raw_response)

    resolved = _clean_response(raw_response)
    
    if len(resolved) > 500:
        logger.warning("Resolved query unusually long. Falling back to follow-up.")
        return followup_query

    if not resolved:
        logger.warning(
            "Gemini returned an empty response for follow-up resolution. "
            "Falling back to raw follow-up query: %r",
            followup_query,
        )
        return followup_query
    
    if resolved.lower() == followup_query.lower():
        logger.debug("Follow-up unchanged after resolution.")

    logger.info("Resolved follow-up: %r → %r", followup_query, resolved)
    return resolved