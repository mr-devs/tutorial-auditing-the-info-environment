"""End-to-end quiz flow tests over the JSON API (AI dispatch stubbed)."""

from app import db


def start_session(client):
    res = client.post("/api/sessions")
    assert res.status_code == 201
    return res.json()


def test_create_session(client):
    data = start_session(client)
    assert data["n_questions"] == 3
    assert data["topic"] == "artificial intelligence"


def test_present_hides_answer_and_dispatches_once(client, stub_ai):
    session = start_session(client)
    sid = session["session_id"]

    res = client.post(f"/api/sessions/{sid}/questions/1")
    assert res.status_code == 200
    q = res.json()
    assert q["question"] == "Question 1?"
    assert len(q["options"]) == 4
    assert "correct_letter" not in q
    assert "explanation" not in q
    assert len(stub_ai) == 1

    # A page refresh re-presents the same question without re-dispatching.
    res2 = client.post(f"/api/sessions/{sid}/questions/1")
    assert res2.json()["presentation_id"] == q["presentation_id"]
    assert len(stub_ai) == 1

    # Pending AI rows were created for every contestant.
    rows = db.fetch_all(
        "SELECT * FROM ai_answers WHERE presentation_id = ?",
        (q["presentation_id"],),
    )
    assert len(rows) == 3
    assert all(r["status"] == "pending" for r in rows)


def test_cannot_skip_ahead(client):
    session = start_session(client)
    sid = session["session_id"]
    res = client.post(f"/api/sessions/{sid}/questions/2")
    assert res.status_code == 403


def test_answer_grading_and_completion(client):
    session = start_session(client)
    sid = session["session_id"]

    for position in (1, 2, 3):
        pres = client.post(f"/api/sessions/{sid}/questions/{position}").json()
        letter = "B" if position < 3 else "A"
        res = client.post(
            f"/api/presentations/{pres['presentation_id']}/answer",
            json={"answer_letter": letter, "confidence": 60 + position},
        )
        assert res.status_code == 200
        fb = res.json()
        assert fb["is_correct"] is (letter == "B")
        assert fb["correct_letter"] == "B"
        assert fb["explanation"].startswith("Explanation")
        assert fb["elapsed_ms"] >= 0
        assert fb["quiz_complete"] is (position == 3)

    summary = client.get(f"/api/sessions/{sid}/summary").json()
    assert summary["completed"] is True
    assert summary["human"]["n_correct"] == 2
    assert summary["human"]["n_answered"] == 3
    assert len(summary["ai"]) == 3
    assert all(m["n_pending"] == 3 for m in summary["ai"])


def test_duplicate_answer_409(client):
    session = start_session(client)
    sid = session["session_id"]
    pres = client.post(f"/api/sessions/{sid}/questions/1").json()
    body = {"answer_letter": "B", "confidence": 50}
    first = client.post(
        f"/api/presentations/{pres['presentation_id']}/answer", json=body
    )
    assert first.status_code == 200
    second = client.post(
        f"/api/presentations/{pres['presentation_id']}/answer", json=body
    )
    assert second.status_code == 409


def test_invalid_answer_rejected(client):
    session = start_session(client)
    sid = session["session_id"]
    pres = client.post(f"/api/sessions/{sid}/questions/1").json()
    res = client.post(
        f"/api/presentations/{pres['presentation_id']}/answer",
        json={"answer_letter": "E", "confidence": 50},
    )
    assert res.status_code == 422  # pydantic pattern
    res = client.post(
        f"/api/presentations/{pres['presentation_id']}/answer",
        json={"answer_letter": "A", "confidence": 150},
    )
    assert res.status_code == 422


def test_resume_points_at_first_unanswered(client):
    session = start_session(client)
    sid = session["session_id"]
    pres = client.post(f"/api/sessions/{sid}/questions/1").json()
    client.post(
        f"/api/presentations/{pres['presentation_id']}/answer",
        json={"answer_letter": "B", "confidence": 50},
    )
    res = client.get(f"/api/sessions/{sid}/resume").json()
    assert res["position"] == 2
    assert res["summary"]["completed"] is False


def test_unknown_session_404(client):
    assert client.get("/api/sessions/nope/summary").status_code == 404
    assert client.post("/api/sessions/nope/questions/1").status_code == 404
