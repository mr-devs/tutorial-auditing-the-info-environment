"""
Site settings, all overridable via environment variables.

The AI contestants map model name -> provider name (as understood by
``toolkit.providers.get_run_parsed``). Model ids are env-overridable so a
model rename on a provider's side can be fixed on the droplet without a
redeploy.
"""

import os
from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = Path(os.getenv("SITE_DB_PATH", SITE_ROOT / "data" / "horserace.db"))

# model name -> provider ('openai' | 'gemini' | 'xai')
AI_CONTESTANTS = {
    os.getenv("SITE_OPENAI_MODEL", "gpt-5.6-terra"): "openai",
    os.getenv("SITE_GEMINI_MODEL", "gemini-3.5-flash"): "gemini",
    os.getenv("SITE_XAI_MODEL", "grok-4.5"): "xai",
}

# Friendly display names for the UI (fall back to the raw model id).
MODEL_DISPLAY_NAMES = {
    os.getenv("SITE_OPENAI_MODEL", "gpt-5.6-terra"): "ChatGPT",
    os.getenv("SITE_GEMINI_MODEL", "gemini-3.5-flash"): "Gemini",
    os.getenv("SITE_XAI_MODEL", "grok-4.5"): "Grok",
}

N_QUIZ_QUESTIONS = int(os.getenv("SITE_N_QUIZ_QUESTIONS", "10"))

# Results are hidden behind this many completed human sessions.
MIN_COMPLETED_SESSIONS = int(os.getenv("SITE_MIN_SESSIONS", "5"))

# Threadpool size for background AI calls (3 calls fire per presentation).
AI_MAX_WORKERS = int(os.getenv("SITE_AI_WORKERS", "12"))

# Optional brake on API spend: max sessions started per UTC day (0 = off).
DAILY_SESSION_CAP = int(os.getenv("SITE_DAILY_SESSION_CAP", "0"))

TOPIC_DEFAULT = "artificial intelligence"
