"""Global configuration for the LLM fact-checking tutorial.

All API keys are read from environment variables — never hard-code keys.
Export them before launching Jupyter/VS Code:

    export OPENAI_API_KEY="sk-..."
    export OPENROUTER_API_KEY="sk-or-..."
"""

import os

from openai import OpenAI
from openrouter import OpenRouter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Real PolitiFact fact-checks (scraped 2024-10-10), copied from the
# llm-vs-human-fc-agreement project.
FACTCHECK_DATA_PATH = os.getenv(
    "FACTCHECK_DATA_PATH", "./data/fact_checks/2024-10-10_factchecks_cleaned.parquet"
)
ARTICLES_DIR = "./data/articles"
QUESTIONS_DIR = "./data/questions"

# Second-best current OpenAI model (flagship tier below gpt-5.6-sol), used via
# the Responses API with the native `web_search` tool where open-book behavior
# is wanted.
DEFAULT_LLM = "gpt-5.6-terra"

# Heterogeneous model roster for the multi-agent debate module — the
# second-best current model from each provider, mapped to a different
# provider/model family via OpenRouter (OpenRouter model ids).
DEBATE_MODEL_ROSTER = {
    "explainer": "openai/gpt-5.6-terra",
    "debater_general": "anthropic/claude-opus-4.8",
    "debater_typology": "google/gemini-3.5-flash",
    "judge": "x-ai/grok-4.3",
    "refiner": "openai/gpt-5.6-terra",
}

# OpenRouter web-search plugin spec (works with any model on OpenRouter);
# passed as `plugins=` to `client.chat.send` for roles that should search.
OPENROUTER_WEB_PLUGIN = [{"id": "web"}]


def get_openai_client():
    if not OPENAI_API_KEY:
        raise ValueError("Please export OPENAI_API_KEY as an environment variable.")
    return OpenAI(api_key=OPENAI_API_KEY)


def get_openrouter_client():
    if not OPENROUTER_API_KEY:
        raise ValueError("Please export OPENROUTER_API_KEY as an environment variable.")
    return OpenRouter(api_key=OPENROUTER_API_KEY)
