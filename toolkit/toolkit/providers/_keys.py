"""
API-key resolution shared by all provider modules.

``PROVIDER_ENV`` is the single source of truth mapping provider names to their
canonical environment-variable names, and ``load_api_key()`` resolves a key
from the environment (preferring the lab machines' ``SML_``-prefixed variant,
logging which variable was used, and raising ``ValueError`` if none is set).
"""

import logging
import os

# Web-search tool spec shared by every OpenAI-compatible provider (openai, xai).
# Defined here so both providers import the same constant rather than duplicating
# the string literal; a single edit keeps them in sync if the spec ever changes.
OPENAI_WEBSEARCH_TOOLS = [{"type": "web_search"}]

# Single source of truth: provider name → canonical env-var name.
PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "xai": "XAI_API_KEY",
}


def load_api_key(key_name: str) -> str:
    """
    Load an API key from the environment, preferring an ``SML_``-prefixed variant.

    The lab's machines set keys prefixed with ``SML_`` (e.g. ``SML_OPENAI_API_KEY``).
    This function checks for ``SML_<key_name>`` first and falls back to ``<key_name>``,
    so the maintainer's personal keys are used when present while external reproducers
    of the code can simply set the standard variable (e.g. ``OPENAI_API_KEY``). The name
    of the variable actually used is logged (INFO) for transparency.

    Parameters
    ----------
    key_name : str
        The base environment variable name, e.g. ``"OPENAI_API_KEY"``.

    Returns
    -------
    str
        The API key value.

    Raises
    ------
    ValueError
        If neither the ``SML_``-prefixed nor the base variable is set.
    """
    prefixed_name = f"SML_{key_name}"
    for name in (prefixed_name, key_name):
        value = os.getenv(name)
        if value:
            logging.getLogger(__name__).info(
                "Loaded API key from environment variable: %s", name
            )
            return value
    raise ValueError(
        f"No API key found. Set {prefixed_name} or {key_name} in your environment."
    )
