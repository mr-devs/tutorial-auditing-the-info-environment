"""Results gate and aggregation math on a hand-built database."""

from app import config, db, results


def _complete_session(client, n_correct: int):
    """Play one full 3-question session getting ``n_correct`` right."""
    sid = client.post("/api/sessions").json()["session_id"]
    for position in (1, 2, 3):
        pres = client.post(f"/api/sessions/{sid}/questions/{position}").json()
        letter = "B" if position <= n_correct else "A"
        client.post(
            f"/api/presentations/{pres['presentation_id']}/answer",
            json={"answer_letter": letter, "confidence": 70},
        )
    return sid


def test_gate_below_threshold(client, monkeypatch):
    monkeypatch.setattr(config, "MIN_COMPLETED_SESSIONS", 2)
    _complete_session(client, 3)
    payload = results.aggregate()
    assert payload["enough_data"] is False
    assert payload["n_sessions"] == 1
    assert payload["sessions_needed"] == 1

    page = client.get("/results")
    assert page.status_code == 200
    assert "Too early to call" in page.text


def test_aggregation_math(client, monkeypatch):
    monkeypatch.setattr(config, "MIN_COMPLETED_SESSIONS", 2)
    _complete_session(client, 3)
    _complete_session(client, 1)

    # Give one AI model completed answers on every presentation so the
    # fair-pairing join has rows: 4 correct, 2 wrong.
    rows = db.fetch_all("SELECT id FROM presentations ORDER BY presented_at")
    for i, row in enumerate(rows):
        db.execute(
            """
            UPDATE ai_answers
            SET status = 'ok', answer_letter = ?, is_correct = ?,
                confidence = 80, elapsed_ms = 12000
            WHERE presentation_id = ? AND model = ?
            """,
            (
                "B" if i < 4 else "A",
                int(i < 4),
                row["id"],
                list(config.AI_CONTESTANTS)[0],
            ),
        )

    payload = results.aggregate()
    assert payload["enough_data"] is True
    assert payload["n_sessions"] == 2

    by_name = {c["name"]: c for c in payload["contestants"]}
    humans = by_name["Humans"]
    assert humans["n_answers"] == 6
    assert humans["accuracy"] == round(100 * 4 / 6, 1)
    assert humans["mean_confidence"] == 70

    ai_name = config.MODEL_DISPLAY_NAMES[list(config.AI_CONTESTANTS)[0]]
    ai = by_name[ai_name]
    assert ai["n_answers"] == 6
    assert ai["accuracy"] == round(100 * 4 / 6, 1)
    assert ai["median_seconds"] == 12.0

    # Models with only pending rows are excluded entirely.
    other = config.MODEL_DISPLAY_NAMES[list(config.AI_CONTESTANTS)[1]]
    assert other not in by_name

    # Per-question difficulty covers the active set.
    assert len(payload["per_question"]) == 3
    assert payload["per_question"][0]["n_human_answers"] == 2
