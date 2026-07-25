"""
Background AI answering: the machine side of the race.

When a question is presented to a human (a new ``presentations`` row), the
endpoint calls :func:`dispatch`, which fires one background task per AI
contestant. Each task runs the same prompt/schema as the tutorial's Step 4
web_search condition (via ``sitekit``), timed server-side, and writes its
result into ``ai_answers`` whenever it finishes — the human's flow never
waits on the models, and a provider outage degrades to ``status='error'``
rows rather than a broken quiz.

The provider SDK calls are synchronous, so they run in a module-level
``ThreadPoolExecutor`` sized by ``config.AI_MAX_WORKERS``.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from app import config, db
from sitekit import prompts, providers
from sitekit.answers import Answer, _detect_search_use

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None

# Models whose API key resolved at startup; the rest are disabled for the
# process (their pending rows go straight to 'error').
_enabled_models: dict[str, str] = {}


def startup() -> None:
    """Create the shared executor and probe each contestant's API key.

    A missing key logs a warning and disables that model — it must not
    crash the site.
    """
    global _executor, _enabled_models
    _executor = ThreadPoolExecutor(max_workers=config.AI_MAX_WORKERS)
    _enabled_models = {}
    for model, provider in config.AI_CONTESTANTS.items():
        try:
            providers.load_api_key(providers.PROVIDER_ENV[provider])
            _enabled_models[model] = provider
        except ValueError as exc:
            logger.warning("AI contestant %s (%s) disabled: %s", model, provider, exc)
    logger.info("AI contestants enabled: %s", sorted(_enabled_models) or "none")


def shutdown() -> None:
    """Stop the executor without waiting for in-flight calls."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


def enabled_models() -> dict:
    """Return the model -> provider map of contestants with working keys."""
    return dict(_enabled_models)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _call_model(provider: str, model: str, question: str, options: list):
    """One synchronous provider call — runs inside the thread pool."""
    run_parsed = providers.get_run_parsed(provider)
    return run_parsed(
        model,
        prompts.ANSWER_SYSTEM_PROMPT,
        prompts.build_answer_user_prompt(question, options),
        Answer,
        use_web_search=True,
    )


def _finish_ok(presentation_id, model, provider, correct_letter, parsed, raw):
    db.execute(
        """
        UPDATE ai_answers
        SET status = 'ok', answer_letter = ?, is_correct = ?, confidence = ?,
            reasoning = ?, search_used = ?, ended_at = ?,
            elapsed_ms = CAST(
                (julianday(?) - julianday(started_at)) * 86400000 AS INTEGER
            )
        WHERE presentation_id = ? AND model = ?
        """,
        (
            parsed.answer_letter,
            int(parsed.answer_letter == correct_letter),
            round(parsed.confidence * 100),
            parsed.reasoning,
            int(_detect_search_use(provider, raw)),
            _utcnow(),
            _utcnow(),
            presentation_id,
            model,
        ),
    )


def _finish_error(presentation_id, model, error: str):
    db.execute(
        """
        UPDATE ai_answers
        SET status = 'error', error = ?, ended_at = ?
        WHERE presentation_id = ? AND model = ?
        """,
        (error[:500], _utcnow(), presentation_id, model),
    )


async def _run_one(presentation_id: str, question_row: dict, model: str, provider: str):
    """Run one AI contestant on one presentation, start to DB writeback."""
    loop = asyncio.get_running_loop()
    db.execute(
        "UPDATE ai_answers SET started_at = ? WHERE presentation_id = ? AND model = ?",
        (_utcnow(), presentation_id, model),
    )
    try:
        parsed, raw = await loop.run_in_executor(
            _executor,
            _call_model,
            provider,
            model,
            question_row["question"],
            json.loads(question_row["options_json"]),
        )
        if parsed is None:
            _finish_error(presentation_id, model, "Response could not be parsed.")
            return
        _finish_ok(
            presentation_id,
            model,
            provider,
            question_row["correct_letter"],
            parsed,
            raw,
        )
        logger.info("AI answer stored: %s on presentation %s", model, presentation_id)
    except Exception as exc:  # noqa: BLE001 — an AI failure must never propagate
        logger.error("AI contestant %s failed on %s: %s", model, presentation_id, exc)
        _finish_error(presentation_id, model, str(exc))


def dispatch(presentation_id: str, question_row: dict) -> None:
    """Insert pending rows and fire one background task per AI contestant.

    Call this exactly once per presentation (the caller guards with the
    presentations UNIQUE constraint). Disabled models get an immediate
    ``error`` row so the UI can show them as unavailable rather than
    forever "thinking".

    Parameters
    ----------
    presentation_id : str
        The presentation the AI runs attach to.
    question_row : dict
        The questions table row (needs ``question``, ``options_json``,
        ``correct_letter``).
    """
    for model, provider in config.AI_CONTESTANTS.items():
        db.execute(
            """
            INSERT OR IGNORE INTO ai_answers (presentation_id, model, provider, status)
            VALUES (?, ?, ?, 'pending')
            """,
            (presentation_id, model, provider),
        )
        if model not in _enabled_models:
            _finish_error(presentation_id, model, "Model disabled: no API key.")
            continue
        asyncio.create_task(_run_one(presentation_id, question_row, model, provider))
