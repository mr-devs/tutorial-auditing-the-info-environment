"""
SML_-prefixed API-key resolution shared by all provider modules.

``PROVIDER_ENV`` is the single source of truth mapping provider names to their
canonical environment-variable names. Both ``require_env()`` (startup
validation) and each provider's ``_get_client()`` derive the variable name from
this dict, so the two never drift apart.
"""

import os

API_KEY_PREFIX = "SML_"

# Web-search tool spec shared by every OpenAI-compatible provider (openai, xai).
# Defined here so both providers import the same constant rather than duplicating
# the string literal; a single edit keeps them in sync if the spec ever changes.
OPENAI_WEBSEARCH_TOOLS = [{"type": "web_search"}]

# Single source of truth: provider name → canonical env-var name.
# Both providers/__init__.py (require_env) and each *_provider.py (_get_client)
# import this dict so there is no hardcoded duplication.
PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "xai": "XAI_API_KEY",
}


def resolve_api_key(base_name):
    """
    Return the API key for ``base_name``, preferring the SML_-prefixed variant.

    The project owner can set personal ``SML_``-prefixed keys, while anyone
    reproducing the work uses the standard variable names — both flow through
    this single code path. Lookup happens in strict priority order:

        1. ``SML_<base_name>``  (e.g. ``SML_OPENAI_API_KEY``)  -- tried first
        2. ``<base_name>``      (e.g. ``OPENAI_API_KEY``)      -- fallback

    The name of the variable that supplied the key is always printed to stdout.
    The key value itself is never printed. Raises ``RuntimeError`` if neither
    variable is set.

    Parameters
    ----------
    base_name : str
        The canonical environment variable name, e.g. ``"OPENAI_API_KEY"``.

    Returns
    -------
    str
        The API key value.

    Raises
    ------
    RuntimeError
        If neither ``SML_<base_name>`` nor ``<base_name>`` is set.
    """
    prefixed = f"{API_KEY_PREFIX}{base_name}"
    for name in (prefixed, base_name):  # SML_ first, standard second
        value = os.environ.get(name)
        if value:
            print(f"Loaded API key from {name}")
            return value
    raise RuntimeError(f"Missing required API key: set {prefixed} or {base_name}.")
