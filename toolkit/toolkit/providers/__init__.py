"""
Provider adapters for the tutorial's LLM calls (OpenAI and Gemini).

Each provider module exposes the same interface so callers can treat
providers as interchangeable:

    run_parsed(model, system_prompt, user_text, response_format)
        -> (parsed_pydantic_object_or_None, raw_response_dict)

API keys are resolved with :func:`load_api_key`, which prefers the lab
machines' ``SML_``-prefixed environment variables and falls back to the
standard names in :data:`PROVIDER_ENV`.

Import provider modules directly so consumers only pay for the SDKs they use:

    from toolkit.providers import openai_provider
    from toolkit.providers import gemini_provider
"""

from toolkit.providers._keys import PROVIDER_ENV, load_api_key

__all__ = ["PROVIDER_ENV", "load_api_key"]
