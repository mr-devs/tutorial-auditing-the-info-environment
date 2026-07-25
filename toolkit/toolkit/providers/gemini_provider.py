"""
Provider: Google Gemini (google-genai SDK) with Pydantic structured outputs.

`run_parsed(model, system_prompt, user_text, response_format,
use_web_search=False)` returns `(parsed_pydantic_object_or_None,
raw_response_dict)` — the same interface as
`toolkit.providers.openai_provider`, so callers can swap providers freely.

The system prompt maps to Gemini's ``system_instruction`` config field and the
Pydantic schema to ``response_schema`` (with JSON output mode), mirroring the
system/user message split used with OpenAI.

Web search is off by default; ``use_web_search=True`` adds the built-in
``google_search`` grounding tool. Gemini 3-series models accept the search
tool and a JSON ``response_schema`` in the same call (this combination was
rejected in the Gemini 2.x era). If a future model 400s on the combination,
the fallback — not implemented here, by design — is two calls: a schema-free
grounded call followed by a structured extraction.
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

# Built-in Google Search grounding tool, mirroring OPENAI_WEBSEARCH_TOOLS.
# Defined here (not in _keys.py) so _keys.py stays free of SDK imports.
GEMINI_WEBSEARCH_TOOLS = [types.Tool(google_search=types.GoogleSearch())]


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient failures: 429 rate limits and 5xx server errors."""
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
def _call(model, system_prompt, user_text, response_format, use_web_search):
    """Issue one retried ``generate_content`` call and return the raw response."""
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=response_format,
    )
    if use_web_search:
        config.tools = GEMINI_WEBSEARCH_TOOLS
    return _get_client().models.generate_content(
        model=model,
        contents=user_text,
        config=config,
    )


def run(model, system_prompt, user_text, response_format, use_web_search=False):
    """Call the Gemini API and return the raw response dict.

    Parameters
    ----------
    model : str
        Gemini model name.
    system_prompt : str
        The system message (mapped to Gemini's ``system_instruction``).
    user_text : str
        The user message.
    response_format : type
        Pydantic model class describing the expected structured output
        (mapped to ``response_schema`` with JSON output mode).
    use_web_search : bool, default False
        Enable the built-in ``google_search`` grounding tool. Off by
        default so Step 2's question generation stays grounded in the
        supplied article.

    Returns
    -------
    dict
        The raw response as a dict (``model_dump``).
    """
    response = _call(model, system_prompt, user_text, response_format, use_web_search)
    return response.model_dump(mode="json", exclude_none=True)


def run_parsed(model, system_prompt, user_text, response_format, use_web_search=False):
    """Like ``run`` but also return the schema-validated Pydantic object.

    Parameters
    ----------
    model : str
        Gemini model name.
    system_prompt : str
        The system message (mapped to Gemini's ``system_instruction``).
    user_text : str
        The user message.
    response_format : type
        Pydantic model class describing the expected structured output
        (mapped to ``response_schema`` with JSON output mode).
    use_web_search : bool, default False
        Enable the built-in ``google_search`` grounding tool. Off by
        default so Step 2's question generation stays grounded in the
        supplied article.

    Returns
    -------
    tuple of (object or None, dict)
        ``(parsed, raw_dict)`` where ``parsed`` is an instance of
        ``response_format`` (or ``None`` if the model's output failed to
        parse) and ``raw_dict`` is the raw response as a dict.
    """
    response = _call(model, system_prompt, user_text, response_format, use_web_search)
    return response.parsed, response.model_dump(mode="json", exclude_none=True)
