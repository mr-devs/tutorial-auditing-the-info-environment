"""
Aggregate results: humans vs. each AI contestant.

Fair-pairing rules: only presentations the human actually answered are
counted, and AI rows only when ``status = 'ok'``. Below the
``MIN_COMPLETED_SESSIONS`` gate the API returns ``enough_data = False`` so
the page can show a friendly message instead of noisy small-sample stats.
"""

import statistics

from app import config, db


def _contestant_stats(name: str, rows) -> dict:
    """Compute one contestant's summary from (is_correct, confidence, elapsed_ms) rows."""
    n = len(rows)
    correct = [r["is_correct"] for r in rows]
    confidences = [r["confidence"] for r in rows if r["confidence"] is not None]
    elapsed = [r["elapsed_ms"] for r in rows if r["elapsed_ms"] is not None]
    conf_correct = [
        r["confidence"] for r in rows if r["is_correct"] and r["confidence"] is not None
    ]
    conf_wrong = [
        r["confidence"]
        for r in rows
        if not r["is_correct"] and r["confidence"] is not None
    ]
    return {
        "name": name,
        "n_answers": n,
        "accuracy": round(100 * sum(correct) / n, 1) if n else None,
        "mean_confidence": round(statistics.mean(confidences), 1)
        if confidences
        else None,
        "median_seconds": round(statistics.median(elapsed) / 1000, 1)
        if elapsed
        else None,
        "conf_when_correct": round(statistics.mean(conf_correct), 1)
        if conf_correct
        else None,
        "conf_when_wrong": round(statistics.mean(conf_wrong), 1)
        if conf_wrong
        else None,
    }


def completed_session_count() -> int:
    """Return the number of completed human sessions (all questions answered)."""
    row = db.fetch_one(
        "SELECT COUNT(*) AS n FROM sessions WHERE completed_at IS NOT NULL"
    )
    return row["n"]


def aggregate() -> dict:
    """Build the full results payload, or the not-enough-data gate.

    Returns
    -------
    dict
        ``{"enough_data": False, "n_sessions", "sessions_needed"}`` below
        the gate; otherwise contestant summaries plus per-question human
        difficulty for the active set.
    """
    n_sessions = completed_session_count()
    if n_sessions < config.MIN_COMPLETED_SESSIONS:
        return {
            "enough_data": False,
            "n_sessions": n_sessions,
            "sessions_needed": config.MIN_COMPLETED_SESSIONS - n_sessions,
        }

    human_rows = db.fetch_all(
        """
        SELECT ha.is_correct, ha.confidence, ha.elapsed_ms
        FROM human_answers ha
        """
    )

    contestants = [_contestant_stats("Humans", human_rows)]
    ai_models = db.fetch_all(
        "SELECT DISTINCT model FROM ai_answers WHERE status = 'ok' ORDER BY model"
    )
    for m in ai_models:
        rows = db.fetch_all(
            """
            SELECT a.is_correct, a.confidence, a.elapsed_ms
            FROM ai_answers a
            JOIN presentations p ON p.id = a.presentation_id
            JOIN human_answers ha ON ha.presentation_id = p.id
            WHERE a.status = 'ok' AND a.model = ?
            """,
            (m["model"],),
        )
        display = config.MODEL_DISPLAY_NAMES.get(m["model"], m["model"])
        contestants.append(_contestant_stats(display, rows))

    active = db.fetch_one("SELECT * FROM question_sets WHERE is_active = 1")
    per_question = []
    if active is not None:
        per_question = [
            dict(r)
            for r in db.fetch_all(
                """
                SELECT q.position, q.headline,
                       COUNT(ha.presentation_id) AS n_human_answers,
                       ROUND(100.0 * SUM(ha.is_correct) /
                             COUNT(ha.presentation_id), 1) AS human_accuracy
                FROM questions q
                LEFT JOIN presentations p ON p.question_id = q.id
                LEFT JOIN human_answers ha ON ha.presentation_id = p.id
                WHERE q.set_id = ?
                GROUP BY q.id
                HAVING n_human_answers > 0
                ORDER BY q.position
                """,
                (active["id"],),
            )
        ]

    return {
        "enough_data": True,
        "n_sessions": n_sessions,
        "contestants": contestants,
        "per_question": per_question,
    }
