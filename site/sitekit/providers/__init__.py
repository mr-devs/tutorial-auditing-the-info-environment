"""
Provider adapters for the site's LLM calls (OpenAI, Gemini, and xAI).

Each provider module exposes the same interface so callers can treat
providers as interchangeable:

    run_parsed(model, system_prompt, user_text, response_format,
               use_web_search=False)
        -> (parsed_pydantic_object_or_None, raw_response_dict)

API keys are resolved with :func:`load_api_key`, which prefers the lab
machines' ``SML_``-prefixed environment variables and falls back to the
standard names in :data:`PROVIDER_ENV`.

Import provider modules directly so consumers only pay for the SDKs they use:

    from sitekit.providers import openai_provider
    from sitekit.providers import gemini_provider
    from sitekit.providers import xai_provider
"""

from sitekit.providers._keys import PROVIDER_ENV, load_api_key

__all__ = ["PROVIDER_ENV", "get_run_parsed", "load_api_key"]


def get_run_parsed(provider: str):
    """Return ``run_parsed`` for a provider name ('openai', 'gemini', or 'xai').

    Provider modules are imported lazily so consumers only pay for the SDKs
    they actually use.

    Parameters
    ----------
    provider : str
        Provider name: 'openai', 'gemini', or 'xai'.

    Returns
    -------
    callable
        The provider module's ``run_parsed`` function, with signature
        ``run_parsed(model, system_prompt, user_text, response_format,
        use_web_search=False)``.

    Raises
    ------
    ValueError
        If ``provider`` is not 'openai', 'gemini', or 'xai'.
    """
    if provider == "openai":
        from sitekit.providers import openai_provider

        return openai_provider.run_parsed
    if provider == "gemini":
        from sitekit.providers import gemini_provider

        return gemini_provider.run_parsed
    if provider == "xai":
        from sitekit.providers import xai_provider

        return xai_provider.run_parsed
    raise ValueError(
        f"Unknown provider: {provider!r} (expected 'openai', 'gemini', or 'xai')"
    )
