"""Global configuration for the tutorial.

All API keys are read from environment variables — never hard-code keys.
Export them before launching Jupyter/VS Code:

    export GUARDIAN_API_KEY="..."      # free: https://open-platform.theguardian.com/access/
    export OPENAI_API_KEY="sk-..."
    export GEMINI_API_KEY="..."        # free: https://aistudio.google.com/apikey
    export OPENROUTER_API_KEY="sk-or-..."

The Guardian also offers the shared public key ``test`` for light
experimentation without registering. LLM keys are resolved through
``toolkit.providers.load_api_key``, which prefers ``SML_``-prefixed variants
(the lab machines' convention) before the standard names above.
"""

import os

GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Step 1 output: one JSONL file of Guardian articles per collection run.
ARTICLES_DIR = os.getenv("ARTICLES_DIR", "./data/articles")

# Step 2 output: one JSONL file of generated questions per provider run.
QUESTIONS_DIR = os.getenv("QUESTIONS_DIR", "./data/questions")

# Models supported by the tutorial scripts, mapped to the provider that
# serves them (used to route --model choices to the right SDK adapter).
SUPPORTED_MODELS = {
    "gpt-5.4-mini-2026-03-17": "openai",
    "gpt-5.6-terra": "openai",
    "gemini-3.1-flash-lite": "gemini",
    "gemini-3.5-flash": "gemini",
}

# Default generator models when only a provider is specified.
DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


def get_openai_client():
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise ValueError("Please export OPENAI_API_KEY as an environment variable.")
    return OpenAI(api_key=OPENAI_API_KEY)


def get_openrouter_client():
    from openrouter import OpenRouter

    if not OPENROUTER_API_KEY:
        raise ValueError("Please export OPENROUTER_API_KEY as an environment variable.")
    return OpenRouter(api_key=OPENROUTER_API_KEY)
