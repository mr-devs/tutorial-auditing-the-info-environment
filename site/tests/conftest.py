"""Shared fixtures: a temporary database seeded with one active question set."""

import json

import pytest

from app import config, db


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Point the app at a fresh temporary SQLite file."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    db.init_db()
    return db_path


@pytest.fixture()
def seeded_set(tmp_db):
    """Insert an active question set with 3 questions; return its id."""
    with db.connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO question_sets
                (topic, created_at, from_date, to_date,
                 n_articles, n_generated, n_passing, seed, is_active)
            VALUES ('artificial intelligence', '2026-07-25T00:00:00+00:00',
                    '2026-07-18', '2026-07-25', 10, 30, 12, 42, 1)
            """
        )
        set_id = cur.lastrowid
        for position in (1, 2, 3):
            conn.execute(
                """
                INSERT INTO questions
                    (set_id, position, source_question_id, article_id,
                     article_url, headline, published, question, options_json,
                     correct_letter, explanation, generator_model,
                     n_models_passing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    set_id,
                    position,
                    f"openai__article-{position}__q0",
                    f"article-{position}",
                    f"https://example.org/{position}",
                    f"Headline {position}",
                    "2026-07-24T10:00:00Z",
                    f"Question {position}?",
                    json.dumps(
                        [f"Opt A{position}", f"Opt B{position}", "Opt C", "Opt D"]
                    ),
                    "B",
                    f"Explanation {position}.",
                    "gpt-test",
                    3,
                ),
            )
    return set_id


@pytest.fixture()
def stub_ai(monkeypatch):
    """Replace ai_runner.dispatch with a recorder that leaves rows pending."""
    from app import ai_runner

    calls = []

    def fake_dispatch(presentation_id, question_row):
        calls.append(presentation_id)
        for model, provider in config.AI_CONTESTANTS.items():
            db.execute(
                """
                INSERT OR IGNORE INTO ai_answers
                    (presentation_id, model, provider, status)
                VALUES (?, ?, ?, 'pending')
                """,
                (presentation_id, model, provider),
            )

    monkeypatch.setattr(ai_runner, "dispatch", fake_dispatch)
    # main.py imports the module (not the function), so patching the module
    # attribute is enough.
    return calls


@pytest.fixture()
def client(seeded_set, stub_ai):
    """A TestClient over the seeded database with stubbed AI dispatch."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
