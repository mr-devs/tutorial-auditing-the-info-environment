"""
Quiz flow logic: sessions, presentations, and human answers.

Pure functions over the SQLite database — no FastAPI imports — so the
endpoints in ``app.main`` stay thin and everything here is unit-testable
with a temporary database.

Timing is authoritative on the server: a question's clock starts when the
presentation row is created and stops when the answer arrives, so client
clock skew (or tampering) cannot affect ``elapsed_ms``.
"""

import json
import uuid
from datetime import datetime, timezone

from app import config, db


class QuizError(Exception):
    """A quiz-flow rule was violated; ``status_code`` maps to HTTP."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def get_active_set():
    """Return the active question_sets row, or None."""
    return db.fetch_one("SELECT * FROM question_sets WHERE is_active = 1")


def count_set_questions(set_id: int) -> int:
    """Return how many questions belong to a question set."""
    row = db.fetch_one(
        "SELECT COUNT(*) AS n FROM questions WHERE set_id = ?", (set_id,)
    )
    return row["n"]


def create_session(user_agent: str = "") -> dict:
    """Start a new quiz session against the active question set.

    Parameters
    ----------
    user_agent : str, default ""
        The visitor's User-Agent header (context only, never shown).

    Returns
    -------
    dict
        ``{"session_id", "n_questions", "topic"}``.

    Raises
    ------
    QuizError
        409 if there is no active question set; 429 if the daily session
        cap is reached.
    """
    active = get_active_set()
    if active is None:
        raise QuizError("No question set is loaded yet.", status_code=409)

    if config.DAILY_SESSION_CAP:
        today = datetime.now(timezone.utc).date().isoformat()
        row = db.fetch_one(
            "SELECT COUNT(*) AS n FROM sessions WHERE started_at >= ?", (today,)
        )
        if row["n"] >= config.DAILY_SESSION_CAP:
            raise QuizError(
                "Today's race limit has been reached — come back tomorrow!",
                status_code=429,
            )

    session_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO sessions (id, set_id, started_at, user_agent) VALUES (?, ?, ?, ?)",
        (session_id, active["id"], _utcnow(), user_agent[:300]),
    )
    return {
        "session_id": session_id,
        "n_questions": count_set_questions(active["id"]),
        "topic": active["topic"],
    }


def get_session(session_id: str):
    """Return the sessions row, raising 404 if unknown."""
    session = db.fetch_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
    if session is None:
        raise QuizError("Unknown session.", status_code=404)
    return session


def present_question(session_id: str, position: int) -> tuple[dict, bool]:
    """Fetch a question for the human, creating its presentation if new.

    Creating the presentation is the moment the race starts for this
    question: the caller must dispatch the AI runs if (and only if) the
    presentation was newly created — repeat calls (page refreshes) return
    the existing presentation without re-triggering the AIs.

    Parameters
    ----------
    session_id : str
        The visitor's session id.
    position : int
        1-based question position within the session's set.

    Returns
    -------
    tuple of (dict, bool)
        ``(payload, created)`` where ``payload`` is the client-facing
        question dict (never containing the correct letter or explanation)
        and ``created`` says whether a new presentation row was inserted.

    Raises
    ------
    QuizError
        404 for unknown session/position, 403 if the previous question has
        not been answered yet.
    """
    session = get_session(session_id)
    question = db.fetch_one(
        "SELECT * FROM questions WHERE set_id = ? AND position = ?",
        (session["set_id"], position),
    )
    if question is None:
        raise QuizError("No such question.", status_code=404)

    total = count_set_questions(session["set_id"])

    if position > 1:
        answered = db.fetch_one(
            """
            SELECT 1 FROM human_answers ha
            JOIN presentations p ON p.id = ha.presentation_id
            JOIN questions q ON q.id = p.question_id
            WHERE p.session_id = ? AND q.position = ?
            """,
            (session_id, position - 1),
        )
        if answered is None:
            raise QuizError("Answer the previous question first.", status_code=403)

    presentation_id = str(uuid.uuid4())
    created = (
        db.execute(
            """
            INSERT OR IGNORE INTO presentations
                (id, session_id, question_id, presented_at)
            VALUES (?, ?, ?, ?)
            """,
            (presentation_id, session_id, question["id"], _utcnow()),
        )
        == 1
    )
    presentation = db.fetch_one(
        "SELECT * FROM presentations WHERE session_id = ? AND question_id = ?",
        (session_id, question["id"]),
    )

    already_answered = (
        db.fetch_one(
            "SELECT 1 FROM human_answers WHERE presentation_id = ?",
            (presentation["id"],),
        )
        is not None
    )

    payload = {
        "presentation_id": presentation["id"],
        "position": position,
        "total": total,
        "question": question["question"],
        "options": json.loads(question["options_json"]),
        "headline": question["headline"],
        "published": question["published"],
        "already_answered": already_answered,
    }
    return payload, created


def get_presentation_question(presentation_id: str):
    """Return (presentation, question) rows for a presentation id."""
    presentation = db.fetch_one(
        "SELECT * FROM presentations WHERE id = ?", (presentation_id,)
    )
    if presentation is None:
        raise QuizError("Unknown presentation.", status_code=404)
    question = db.fetch_one(
        "SELECT * FROM questions WHERE id = ?", (presentation["question_id"],)
    )
    return presentation, question


def submit_answer(presentation_id: str, answer_letter: str, confidence: int) -> dict:
    """Record the human's answer and return graded feedback.

    Parameters
    ----------
    presentation_id : str
        The presentation being answered.
    answer_letter : str
        One of 'A'-'D'.
    confidence : int
        Self-reported confidence, 0-100.

    Returns
    -------
    dict
        Feedback payload with ``is_correct``, ``correct_letter``,
        ``explanation``, ``elapsed_ms``, ``position``, ``total``, and
        ``quiz_complete``.

    Raises
    ------
    QuizError
        400 for invalid input, 404 for unknown presentation, 409 if the
        presentation was already answered.
    """
    if answer_letter not in ("A", "B", "C", "D"):
        raise QuizError("answer_letter must be one of A, B, C, D.")
    if not isinstance(confidence, int) or not 0 <= confidence <= 100:
        raise QuizError("confidence must be an integer between 0 and 100.")

    presentation, question = get_presentation_question(presentation_id)

    now = datetime.now(timezone.utc)
    elapsed_ms = int(
        (now - _parse_ts(presentation["presented_at"])).total_seconds() * 1000
    )
    is_correct = answer_letter == question["correct_letter"]

    inserted = db.execute(
        """
        INSERT OR IGNORE INTO human_answers
            (presentation_id, answer_letter, is_correct, confidence,
             started_at, ended_at, elapsed_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            presentation_id,
            answer_letter,
            int(is_correct),
            confidence,
            presentation["presented_at"],
            now.isoformat(),
            elapsed_ms,
        ),
    )
    if inserted == 0:
        raise QuizError("This question was already answered.", status_code=409)

    session_id = presentation["session_id"]
    session = get_session(session_id)
    total = count_set_questions(session["set_id"])
    answered = db.fetch_one(
        """
        SELECT COUNT(*) AS n FROM human_answers ha
        JOIN presentations p ON p.id = ha.presentation_id
        WHERE p.session_id = ?
        """,
        (session_id,),
    )["n"]
    quiz_complete = answered >= total
    if quiz_complete and session["completed_at"] is None:
        db.execute(
            "UPDATE sessions SET completed_at = ? WHERE id = ?",
            (now.isoformat(), session_id),
        )

    return {
        "is_correct": is_correct,
        "correct_letter": question["correct_letter"],
        "explanation": question["explanation"],
        "elapsed_ms": elapsed_ms,
        "position": question["position"],
        "total": total,
        "quiz_complete": quiz_complete,
    }


def session_summary(session_id: str) -> dict:
    """Return the human's score plus this session's AI runs (for polling).

    Parameters
    ----------
    session_id : str
        The visitor's session id.

    Returns
    -------
    dict
        Human totals, per-question rows, and per-model AI aggregates with
        per-question status (pending rows included so the end screen can
        show "still thinking...").
    """
    session = get_session(session_id)
    total = count_set_questions(session["set_id"])

    human_rows = db.fetch_all(
        """
        SELECT q.position, ha.answer_letter, ha.is_correct, ha.confidence,
               ha.elapsed_ms
        FROM human_answers ha
        JOIN presentations p ON p.id = ha.presentation_id
        JOIN questions q ON q.id = p.question_id
        WHERE p.session_id = ?
        ORDER BY q.position
        """,
        (session_id,),
    )

    ai_rows = db.fetch_all(
        """
        SELECT q.position, a.model, a.status, a.answer_letter, a.is_correct,
               a.confidence, a.elapsed_ms
        FROM ai_answers a
        JOIN presentations p ON p.id = a.presentation_id
        JOIN questions q ON q.id = p.question_id
        WHERE p.session_id = ?
        ORDER BY q.position
        """,
        (session_id,),
    )

    models = {}
    for row in ai_rows:
        m = models.setdefault(
            row["model"],
            {
                "model": row["model"],
                "display_name": config.MODEL_DISPLAY_NAMES.get(
                    row["model"], row["model"]
                ),
                "n_correct": 0,
                "n_answered": 0,
                "n_pending": 0,
                "n_error": 0,
                "total_elapsed_ms": 0,
                "questions": [],
            },
        )
        m["questions"].append(dict(row))
        if row["status"] == "ok":
            m["n_answered"] += 1
            m["n_correct"] += row["is_correct"] or 0
            m["total_elapsed_ms"] += row["elapsed_ms"] or 0
        elif row["status"] == "pending":
            m["n_pending"] += 1
        else:
            m["n_error"] += 1

    return {
        "session_id": session_id,
        "total": total,
        "completed": session["completed_at"] is not None,
        "human": {
            "n_correct": sum(r["is_correct"] for r in human_rows),
            "n_answered": len(human_rows),
            "total_elapsed_ms": sum(r["elapsed_ms"] for r in human_rows),
            "questions": [dict(r) for r in human_rows],
        },
        "ai": sorted(models.values(), key=lambda m: m["display_name"]),
    }


def first_unanswered_position(session_id: str) -> int:
    """Return the 1-based position the session should resume at."""
    session = get_session(session_id)
    total = count_set_questions(session["set_id"])
    row = db.fetch_one(
        """
        SELECT COUNT(*) AS n FROM human_answers ha
        JOIN presentations p ON p.id = ha.presentation_id
        WHERE p.session_id = ?
        """,
        (session_id,),
    )
    return min(row["n"] + 1, total)
