"""
Provider: xAI / Grok (OpenAI-compatible Responses API) with Pydantic
structured outputs.

`run_parsed(model, system_prompt, user_text, response_format,
use_web_search=False)` returns `(parsed_pydantic_object_or_None,
raw_response_dict)` — the same interface as
`sitekit.providers.openai_provider` and `sitekit.providers.gemini_provider`,
so callers can swap providers freely.

xAI serves an OpenAI-compatible Responses API at ``https://api.x.ai/v1``,
so this module reuses the ``openai`` SDK with a custom ``base_url`` and the
same ``web_search`` server-side tool spec as the OpenAI provider (xAI's
older ``search_parameters`` Live Search API was retired 2026-01-12).
"""

import logging
from functools import lru_cache

from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from sitekit.providers._keys import (
    OPENAI_WEBSEARCH_TOOLS,
    PROVIDER_ENV,
    load_api_key,
)

logger = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1"

# Transient failures only — never retry auth (401) or bad-request (400) errors.
RETRYABLE_EXCEPTIONS = (APIConnectionError, RateLimitError, InternalServerError)


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    # One shared client per process; the OpenAI client is thread-safe, so
    # ThreadPoolExecutor workers can issue concurrent requests through it.
    return OpenAI(api_key=load_api_key(PROVIDER_ENV["xai"]), base_url=XAI_BASE_URL)


@retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _call(model, system_prompt, user_text, response_format, use_web_search):
    """Issue one retried ``responses.parse`` call and return the raw response."""
    kwargs = dict(
        model=model,
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        text_format=response_format,
    )
    if use_web_search:
        kwargs["tools"] = OPENAI_WEBSEARCH_TOOLS
    return _get_client().responses.parse(**kwargs)


def run(model, system_prompt, user_text, response_format, use_web_search=False):
    """Call the xAI Responses API and return the raw response dict.

    Parameters
    ----------
    model : str
        xAI model name (e.g. ``"grok-4.5"``).
    system_prompt : str
        The system (developer) message.
    user_text : str
        The user message.
    response_format : type
        Pydantic model class describing the expected structured output.
    use_web_search : bool, default False
        Enable the server-side web search tool.

    Returns
    -------
    dict
        The raw response as a dict (``model_dump``).
    """
    response = _call(model, system_prompt, user_text, response_format, use_web_search)
    return response.model_dump(warnings=False)


def run_parsed(model, system_prompt, user_text, response_format, use_web_search=False):
    """Like ``run`` but also return the schema-validated Pydantic object.

    Parameters
    ----------
    model : str
        xAI model name (e.g. ``"grok-4.5"``).
    system_prompt : str
        The system (developer) message.
    user_text : str
        The user message.
    response_format : type
        Pydantic model class describing the expected structured output.
    use_web_search : bool, default False
        Enable the server-side web search tool.

    Returns
    -------
    tuple of (object or None, dict)
        ``(parsed, raw_dict)`` where ``parsed`` is an instance of
        ``response_format`` (or ``None`` if the model's output failed to
        parse) and ``raw_dict`` is the raw response as a dict.
    """
    response = _call(model, system_prompt, user_text, response_format, use_web_search)
    return response.output_parsed, response.model_dump(warnings=False)
