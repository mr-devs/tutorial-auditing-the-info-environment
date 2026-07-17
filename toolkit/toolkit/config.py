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
from pathlib import Path

GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Data paths are anchored to the repository root (two levels above this
# file), NOT the current working directory — so scripts and notebooks read
# and write the same data/ no matter where they are launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Step 1 output: one JSONL file of Guardian articles per collection run.
ARTICLES_DIR = os.getenv("ARTICLES_DIR", str(REPO_ROOT / "data" / "articles"))

# Step 2 output: one JSONL file of generated questions per provider run.
QUESTIONS_DIR = os.getenv("QUESTIONS_DIR", str(REPO_ROOT / "data" / "questions"))

# Step 3 output: one JSONL file of judgments per judge-model run.
JUDGMENTS_DIR = os.getenv("JUDGMENTS_DIR", str(REPO_ROOT / "data" / "judgments"))

# Datetime-stamped log files from the scripts' --create-log-file flag.
LOGS_DIR = os.getenv("LOGS_DIR", str(REPO_ROOT / "logs"))

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

# Judge models for Step 3 (run 03-1_generate_judgments.py once per model),
# mapped to the provider that serves them.
JUDGE_MODELS = {
    "gpt-5.6-luna": "openai",
    "gpt-5.5-2026-04-23": "openai",
    "gpt-5.4-mini-2026-03-17": "openai",
}


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
