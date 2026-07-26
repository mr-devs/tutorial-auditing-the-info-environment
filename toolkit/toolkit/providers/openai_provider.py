"""
Provider: OpenAI (Responses API) with Pydantic structured outputs.

`run_parsed(model, system_prompt, user_text, response_format,
use_web_search=False)` returns `(parsed_pydantic_object_or_None,
raw_response_dict)` — the same interface as
`toolkit.providers.gemini_provider`, so callers can swap providers freely.

Web search is off by default (Step 2's question generation must stay grounded
in the supplied article); Step 4's web_search condition turns it on explicitly.
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


def _build_websearch_tools(include_domains, exclude_domains):
    """Build the web-search tool list, with optional domain filters.

    Parameters
    ----------
    include_domains : list of str or None
        Domains to allow-list (``filters.allowed_domains``).
    exclude_domains : list of str or None
        Domains to block (``filters.blocked_domains``).

    Returns
    -------
    list of dict
        A one-element tool list for the Responses API.
    """
    if not include_domains and not exclude_domains:
        return OPENAI_WEBSEARCH_TOOLS
    filters = {}
    if include_domains:
        filters["allowed_domains"] = list(include_domains)
    if exclude_domains:
        filters["blocked_domains"] = list(exclude_domains)
    return [{"type": "web_search", "filters": filters}]


@retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _call(
    model,
    system_prompt,
    user_text,
    response_format,
    use_web_search,
    temperature=None,
    include_domains=None,
    exclude_domains=None,
):
    """Issue one retried ``responses.parse`` call and return the raw response."""
    if (include_domains or exclude_domains) and not use_web_search:
        raise ValueError("include_domains/exclude_domains require use_web_search=True.")
    kwargs = dict(
        model=model,
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        text_format=response_format,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    if use_web_search:
        kwargs["tools"] = _build_websearch_tools(include_domains, exclude_domains)
    return _get_client().responses.parse(**kwargs)


def run(
    model,
    system_prompt,
    user_text,
    response_format,
    use_web_search=False,
    temperature=None,
    include_domains=None,
    exclude_domains=None,
):
    """Call the OpenAI Responses API and return the raw response dict.

    Parameters
    ----------
    model : str
        OpenAI model name.
    system_prompt : str
        The system (developer) message.
    user_text : str
        The user message.
    response_format : type
        Pydantic model class describing the expected structured output.
    use_web_search : bool, default False
        Enable the web search tool. Off by default so Step 2's question
        generation stays grounded in the supplied article.
    temperature : float or None, default None
        Sampling temperature. ``None`` uses the provider default. Note
        that GPT-5-series reasoning models reject non-default values.
    include_domains : list of str or None, default None
        Restrict web search to these domains (``filters.allowed_domains``,
        up to 100; omit the URL scheme). Requires ``use_web_search=True``.
    exclude_domains : list of str or None, default None
        Exclude these domains from web search
        (``filters.blocked_domains``, up to 100; omit the URL scheme).
        Requires ``use_web_search=True``.

    Returns
    -------
    dict
        The raw response as a dict (``model_dump``).

    Raises
    ------
    ValueError
        If domain filters are given without ``use_web_search=True``.
    """
    response = _call(
        model,
        system_prompt,
        user_text,
        response_format,
        use_web_search,
        temperature=temperature,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )
    return response.model_dump(warnings=False)


def run_parsed(
    model,
    system_prompt,
    user_text,
    response_format,
    use_web_search=False,
    temperature=None,
    include_domains=None,
    exclude_domains=None,
):
    """Like ``run`` but also return the schema-validated Pydantic object.

    Parameters
    ----------
    model : str
        OpenAI model name.
    system_prompt : str
        The system (developer) message.
    user_text : str
        The user message.
    response_format : type
        Pydantic model class describing the expected structured output.
    use_web_search : bool, default False
        Enable the web search tool. Off by default so Step 2's question
        generation stays grounded in the supplied article.
    temperature : float or None, default None
        Sampling temperature. ``None`` uses the provider default. Note
        that GPT-5-series reasoning models reject non-default values.
    include_domains : list of str or None, default None
        Restrict web search to these domains (``filters.allowed_domains``,
        up to 100; omit the URL scheme). Requires ``use_web_search=True``.
    exclude_domains : list of str or None, default None
        Exclude these domains from web search
        (``filters.blocked_domains``, up to 100; omit the URL scheme).
        Requires ``use_web_search=True``.

    Returns
    -------
    tuple of (object or None, dict)
        ``(parsed, raw_dict)`` where ``parsed`` is an instance of
        ``response_format`` (or ``None`` if the model's output failed to
        parse) and ``raw_dict`` is the raw response as a dict.

    Raises
    ------
    ValueError
        If domain filters are given without ``use_web_search=True``.
    """
    response = _call(
        model,
        system_prompt,
        user_text,
        response_format,
        use_web_search,
        temperature=temperature,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )
    return response.output_parsed, response.model_dump(mode="json", warnings=False)
