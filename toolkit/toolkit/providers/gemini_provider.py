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
    """Issue one retried ``generate_content`` call and return the raw response."""
    if include_domains or exclude_domains:
        raise ValueError(
            "Domain filtering is not supported by the Gemini Developer API "
            "(GoogleSearch.exclude_domains is Enterprise-only, and there is "
            "no allow-list at all). Use an OpenAI model for domain filters."
        )
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=response_format,
    )
    if temperature is not None:
        config.temperature = temperature
    if use_web_search:
        config.tools = GEMINI_WEBSEARCH_TOOLS
    return _get_client().models.generate_content(
        model=model,
        contents=user_text,
        config=config,
    )


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
    temperature : float or None, default None
        Sampling temperature (0.0–2.0). ``None`` uses the model default.
    include_domains : list of str or None, default None
        Not supported by the Gemini Developer API; always raises
        ``ValueError`` if set. Present only so the provider interface
        matches ``openai_provider``.
    exclude_domains : list of str or None, default None
        Not supported by the Gemini Developer API
        (``GoogleSearch.exclude_domains`` is Enterprise-only); always
        raises ``ValueError`` if set.

    Returns
    -------
    dict
        The raw response as a dict (``model_dump``).

    Raises
    ------
    ValueError
        If ``include_domains`` or ``exclude_domains`` is set.
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
    return response.model_dump(mode="json", exclude_none=True)


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
    temperature : float or None, default None
        Sampling temperature (0.0–2.0). ``None`` uses the model default.
    include_domains : list of str or None, default None
        Not supported by the Gemini Developer API; always raises
        ``ValueError`` if set. Present only so the provider interface
        matches ``openai_provider``.
    exclude_domains : list of str or None, default None
        Not supported by the Gemini Developer API
        (``GoogleSearch.exclude_domains`` is Enterprise-only); always
        raises ``ValueError`` if set.

    Returns
    -------
    tuple of (object or None, dict)
        ``(parsed, raw_dict)`` where ``parsed`` is an instance of
        ``response_format`` (or ``None`` if the model's output failed to
        parse) and ``raw_dict`` is the raw response as a dict.

    Raises
    ------
    ValueError
        If ``include_domains`` or ``exclude_domains`` is set.
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
    return response.parsed, response.model_dump(mode="json", exclude_none=True)
