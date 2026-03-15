from __future__ import annotations
import os
import logging
from functools import lru_cache
import time

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from google.api_core.exceptions import (
    GoogleAPICallError,
    RetryError,
    InvalidArgument,
    PermissionDenied,
    ResourceExhausted,
    ServiceUnavailable,
)
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"

_GENERATION_CONFIG = genai.GenerationConfig(
    temperature=0.2,
    top_p=0.95,
    max_output_tokens=2048
)

_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

@lru_cache(maxsize=1)
def _get_model():
    
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Export it in your shell or add it to a .env file:\n"
            "  export GEMINI_API_KEY='your-key-here'"
        )

    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL).strip()

    genai.configure(api_key=api_key)
    logger.debug("Gemini SDK configured with model '%s'.", model_name)

    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=_GENERATION_CONFIG,
        safety_settings=_SAFETY_SETTINGS,
    )


def ask_gemini(prompt: str):
    prompt = prompt.strip()
    MAX_PROMPT_CHARS = 20000
    
    if "api_key" in prompt.lower():
        raise ValueError("Prompt contains sensitive information.")

    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS]
        
    if not prompt:
        raise ValueError("prompt must not be empty.")

    model = _get_model()

    logger.debug("Sending prompt to Gemini (%d chars).", len(prompt))

    try:
        for attempt in range(3):
            try:
                response: GenerateContentResponse = model.generate_content(prompt)
                break
            except ResourceExhausted:
                if attempt == 2:
                    raise
                logger.warning("Gemini quota hit. Retrying in 2 seconds...")
                time.sleep(2)
    except InvalidArgument as exc:
        raise ValueError(f"Invalid request sent to Gemini API: {exc}") from exc
    except PermissionDenied as exc:
        raise PermissionDenied(
            f"Access denied — check GEMINI_API_KEY and model access for '{model.model_name}'."
        ) from exc
    except ResourceExhausted as exc:
        raise ResourceExhausted(
            "Gemini API quota exceeded. Wait a moment and retry, or check your "
            "usage limits in Google AI Studio."
        ) from exc
    except ServiceUnavailable as exc:
        raise ServiceUnavailable(
            "Gemini API is temporarily unavailable. Retry in a few seconds."
        ) from exc
    except RetryError as exc:
        raise RuntimeError(
            f"Request to Gemini API failed after multiple retries: {exc}"
        ) from exc
    except GoogleAPICallError as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc

    try:
        text = response.text
        logger.debug("Gemini response preview: %s", text[:200])
    except ValueError as exc:
        block_reason = (
            response.prompt_feedback.block_reason
            if response.prompt_feedback
            else "unknown"
        )
        raise ValueError(
            f"Gemini returned no content (block_reason={block_reason}). "
            "Revise the prompt or adjust safety settings."
        ) from exc

    if not text or not text.strip():
        raise ValueError("Gemini returned an empty response for the given prompt.")

    logger.debug("Received response from Gemini (%d chars).", len(text))
    return text.strip()