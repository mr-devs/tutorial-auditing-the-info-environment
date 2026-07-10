"""
Provider: OpenAI (Responses API). Copied from llm-vs-human-fc-agreement and
adapted for the tutorial: web search is switchable (closed-book vs. open-book
conditions) and the structured-output schema is parameterizable.

`run(model, system_prompt, user_text, ...)` → raw_dict
`run_parsed(model, system_prompt, user_text, ...)` → (parsed_pydantic_object, raw_dict)
"""

from functools import lru_cache

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from toolkit.providers._keys import (
    OPENAI_WEBSEARCH_TOOLS,
    PROVIDER_ENV,
    resolve_api_key,
)
from toolkit.response_structure import OpenAIFactCheckingResponse


@lru_cache(maxsize=1)
def _get_client():
    return OpenAI(api_key=resolve_api_key(PROVIDER_ENV["openai"]))


@retry(wait=wait_random_exponential(min=1, max=90), stop=stop_after_attempt(7))
def _call(model, system_prompt, user_text, use_web_search, text_format):
    kwargs = dict(
        model=model,
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        text_format=text_format,
    )
    if use_web_search:
        kwargs["tools"] = OPENAI_WEBSEARCH_TOOLS
    return _get_client().responses.parse(**kwargs)


def run(
    model,
    system_prompt,
    user_text,
    use_web_search=True,
    text_format=OpenAIFactCheckingResponse,
):
    """Call the OpenAI Responses API and return the raw response dict."""
    response = _call(model, system_prompt, user_text, use_web_search, text_format)
    return response.model_dump(warnings=False)


def run_parsed(
    model,
    system_prompt,
    user_text,
    use_web_search=True,
    text_format=OpenAIFactCheckingResponse,
):
    """Like ``run`` but also return the schema-validated Pydantic object.

    Returns ``(parsed, raw_dict)`` where ``parsed`` is an instance of
    ``text_format`` (or None if the model returned nothing parseable).
    """
    response = _call(model, system_prompt, user_text, use_web_search, text_format)
    return response.output_parsed, response.model_dump(warnings=False)
