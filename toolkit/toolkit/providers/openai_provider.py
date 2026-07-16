"""
Provider: OpenAI (Responses API) with Pydantic structured outputs.

`run_parsed(model, system_prompt, user_text, response_format)` returns
`(parsed_pydantic_object_or_None, raw_response_dict)` — the same interface as
`toolkit.providers.gemini_provider`, so callers can swap providers freely.

Web search is off by default (Step 2's question generation must stay grounded
in the supplied article); Step 4's open-book condition turns it on explicitly.
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

from toolkit.providers._keys import (
    OPENAI_WEBSEARCH_TOOLS,
    PROVIDER_ENV,
    load_api_key,
)

logger = logging.getLogger(__name__)

# Transient failures only — never retry auth (401) or bad-request (400) errors.
RETRYABLE_EXCEPTIONS = (APIConnectionError, RateLimitError, InternalServerError)


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    # One shared client per process; the OpenAI client is thread-safe, so
    # ThreadPoolExecutor workers can issue concurrent requests through it.
    return OpenAI(api_key=load_api_key(PROVIDER_ENV["openai"]))


@retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _call(model, system_prompt, user_text, response_format, use_web_search):
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
    """Call the OpenAI Responses API and return the raw response dict."""
    response = _call(model, system_prompt, user_text, response_format, use_web_search)
    return response.model_dump(warnings=False)


def run_parsed(model, system_prompt, user_text, response_format, use_web_search=False):
    """Like ``run`` but also return the schema-validated Pydantic object.

    Returns ``(parsed, raw_dict)`` where ``parsed`` is an instance of
    ``response_format`` (or ``None`` if the model's output failed to parse).
    """
    response = _call(model, system_prompt, user_text, response_format, use_web_search)
    return response.output_parsed, response.model_dump(warnings=False)
