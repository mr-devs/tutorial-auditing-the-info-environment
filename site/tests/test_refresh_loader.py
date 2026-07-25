"""The refresh CLI's database loader: versioning and activation rules."""

import argparse
import json

from app import db
from refresh_content import load_question_set


def _args(**overrides):
    defaults = dict(topic="artificial intelligence", days=7, seed=42, activate=True)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fake_pipeline(n=3, prefix="q"):
    questions_by_id = {
        f"{prefix}{i}": {
            "id": f"{prefix}{i}",
            "article_id": f"a{i}",
            "question": f"Question {i}?",
            "options": ["W", "X", "Y", "Z"],
            "correct_letter": "A",
            "explanation": "Because.",
            "model": "gpt-test",
        }
        for i in range(n)
    }
    articles_by_id = {
        f"a{i}": {
            "id": f"a{i}",
            "url": f"https://example.org/a{i}",
            "headline": f"Headline {i}",
            "published": "2026-07-24T10:00:00Z",
        }
        for i in range(n)
    }
    passing = {qid: 3 for qid in questions_by_id}
    return questions_by_id, articles_by_id, passing


def test_load_and_activate(tmp_db):
    questions_by_id, articles_by_id, passing = _fake_pipeline()
    selected = sorted(questions_by_id)
    set_id = load_question_set(
        _args(), selected, questions_by_id, articles_by_id, passing
    )

    active = db.fetch_one("SELECT * FROM question_sets WHERE is_active = 1")
    assert active["id"] == set_id
    rows = db.fetch_all(
        "SELECT * FROM questions WHERE set_id = ? ORDER BY position", (set_id,)
    )
    assert [r["position"] for r in rows] == [1, 2, 3]
    assert json.loads(rows[0]["options_json"]) == ["W", "X", "Y", "Z"]
    assert rows[0]["headline"] == "Headline 0"


def test_new_set_replaces_active_but_keeps_history(tmp_db):
    q1, a1, p1 = _fake_pipeline(prefix="first")
    first_id = load_question_set(_args(), sorted(q1), q1, a1, p1)

    q2, a2, p2 = _fake_pipeline(prefix="second")
    second_id = load_question_set(_args(), sorted(q2), q2, a2, p2)

    actives = db.fetch_all("SELECT id FROM question_sets WHERE is_active = 1")
    assert [r["id"] for r in actives] == [second_id]
    # The old set and its questions are untouched.
    old_qs = db.fetch_all("SELECT * FROM questions WHERE set_id = ?", (first_id,))
    assert len(old_qs) == 3


def test_no_activate_stages_only(tmp_db):
    q1, a1, p1 = _fake_pipeline(prefix="live")
    live_id = load_question_set(_args(), sorted(q1), q1, a1, p1)

    q2, a2, p2 = _fake_pipeline(prefix="staged")
    load_question_set(_args(activate=False), sorted(q2), q2, a2, p2)

    active = db.fetch_one("SELECT id FROM question_sets WHERE is_active = 1")
    assert active["id"] == live_id
