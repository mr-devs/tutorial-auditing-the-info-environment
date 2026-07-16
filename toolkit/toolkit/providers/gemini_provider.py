"""
Provider: Google Gemini (google-genai SDK) with Pydantic structured outputs.

`run_parsed(model, system_prompt, user_text, response_format)` returns
`(parsed_pydantic_object_or_None, raw_response_dict)` — the same interface as
`toolkit.providers.openai_provider`, so callers can swap providers freely.

The system prompt maps to Gemini's ``system_instruction`` config field and the
Pydantic schema to ``response_schema`` (with JSON output mode), mirroring the
system/user message split used with OpenAI.
"""

import logging
from functools import lru_cache

from google import genai
from google.genai import errors, types
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from toolkit.providers._keys import PROVIDER_ENV, load_api_key

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Transient failures only: 429 rate limits and 5xx server errors."""
    return isinstance(exc, errors.APIError) and (
        exc.code == 429 or (exc.code or 0) >= 500
    )


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    # One shared client per process; the genai client is thread-safe, so
    # ThreadPoolExecutor workers can issue concurrent requests through it.
    return genai.Client(api_key=load_api_key(PROVIDER_ENV["gemini"]))


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _call(model, system_prompt, user_text, response_format):
    return _get_client().models.generate_content(
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=response_format,
        ),
    )


def run(model, system_prompt, user_text, response_format):
    """Call the Gemini API and return the raw response dict."""
    response = _call(model, system_prompt, user_text, response_format)
    return response.model_dump(mode="json", exclude_none=True)


def run_parsed(model, system_prompt, user_text, response_format):
    """Like ``run`` but also return the schema-validated Pydantic object.

    Returns ``(parsed, raw_dict)`` where ``parsed`` is an instance of
    ``response_format`` (or ``None`` if the model's output failed to parse).
    """
    response = _call(model, system_prompt, user_text, response_format)
    return response.parsed, response.model_dump(mode="json", exclude_none=True)
