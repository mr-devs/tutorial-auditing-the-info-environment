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

__all__ = ["PROVIDER_ENV", "get_run_parsed", "load_api_key"]


def get_run_parsed(provider: str):
    """Return ``run_parsed`` for a provider name ('openai' or 'gemini').

    Provider modules are imported lazily so consumers only pay for the SDKs
    they actually use.
    """
    if provider == "openai":
        from toolkit.providers import openai_provider

        return openai_provider.run_parsed
    if provider == "gemini":
        from toolkit.providers import gemini_provider

        return gemini_provider.run_parsed
    raise ValueError(f"Unknown provider: {provider!r} (expected 'openai' or 'gemini')")
