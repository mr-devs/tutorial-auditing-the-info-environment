"""Global configuration for the tutorial.

All API keys are read from environment variables — never hard-code keys.
Export them before launching Jupyter/VS Code:

    export GUARDIAN_API_KEY="..."      # free: https://open-platform.theguardian.com/access/
    export OPENAI_API_KEY="sk-..."
    export OPENROUTER_API_KEY="sk-or-..."

The Guardian also offers the shared public key ``test`` for light
experimentation without registering.
"""

import os

GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Step 1 output: one JSONL file of Guardian articles per collection run.
ARTICLES_DIR = os.getenv("ARTICLES_DIR", "./data/articles")

DEFAULT_LLM = "gpt-5.6-terra"


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
