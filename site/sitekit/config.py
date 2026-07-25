"""Configuration for the horse-race site's pipeline code (``sitekit``).

``sitekit`` is a copy of the tutorial's ``toolkit`` package, edited for the
website: paths anchor to ``site/`` instead of the repo root, and the xAI
(Grok) provider is added. The tutorial toolkit itself is never imported or
modified by the site.

All API keys are read from environment variables — never hard-code keys:

    export GUARDIAN_API_KEY="..."      # free: https://open-platform.theguardian.com/access/
    export OPENAI_API_KEY="sk-..."
    export GEMINI_API_KEY="..."        # free: https://aistudio.google.com/apikey
    export XAI_API_KEY="xai-..."

LLM keys are resolved through ``sitekit.providers.load_api_key``, which
prefers ``SML_``-prefixed variants (the lab machines' convention) before the
standard names above.
"""

import os
from pathlib import Path

GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY", "")

# Paths are anchored to the site root (two levels above this file), NOT the
# current working directory — so the app and the refresh CLI read and write
# the same site/data/ no matter where they are launched from.
SITE_ROOT = Path(__file__).resolve().parents[2]

# Refresh-pipeline scratch output (articles, questions, judgments JSONL).
REFRESH_DIR = os.getenv("SITE_REFRESH_DIR", str(SITE_ROOT / "data" / "refresh"))

# Datetime-stamped log files.
LOGS_DIR = os.getenv("SITE_LOGS_DIR", str(SITE_ROOT / "logs"))

# Models supported by the pipeline, mapped to the provider that serves them
# (used to route model choices to the right SDK adapter).
SUPPORTED_MODELS = {
    "gpt-5.4-mini-2026-03-17": "openai",
    "gpt-5.6-terra": "openai",
    "gemini-3.1-flash-lite": "gemini",
    "gemini-3.5-flash": "gemini",
}

# Default generator models when only a provider is specified.
DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

# Judge models for the content-refresh pipeline (same three as the
# tutorial's Step 3), mapped to the provider that serves them.
JUDGE_MODELS = {
    "gpt-5.6-luna": "openai",
    "gpt-5.5-2026-04-23": "openai",
    "gpt-5.4-mini-2026-03-17": "openai",
}

# Models allowed to answer questions on the site — the tutorial contestants
# plus the site's Grok racer. sitekit.answers routes through this map.
ANSWER_MODELS = {**SUPPORTED_MODELS, **JUDGE_MODELS, "grok-4.5": "xai"}

# The single-call answering conditions (see sitekit.answers). The site's AI
# racers always use web_search.
ANSWER_METHODS = ("closed_book", "web_search")
