"""
FastAPI app for the horse-race site.

Run locally from ``site/``::

    uv run uvicorn app.main:app --reload

In production this runs as a single uvicorn worker (background AI tasks
and the thread pool are per-process; SQLite likes it too) behind nginx —
see ``deploy/``.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app import ai_runner, config, db, quiz, results
from app.quiz import QuizError

logger = logging.getLogger(__name__)

SITE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=SITE_DIR / "templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    ai_runner.startup()
    yield
    ai_runner.shutdown()


app = FastAPI(title="The Horse Race", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=SITE_DIR / "static"), name="static")


@app.exception_handler(QuizError)
async def quiz_error_handler(request: Request, exc: QuizError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/")
async def index(request: Request):
    active = quiz.get_active_set()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "active_set": active,
            "n_questions": quiz.count_set_questions(active["id"]) if active else 0,
            "model_names": sorted(config.MODEL_DISPLAY_NAMES.values()),
        },
    )


@app.get("/quiz")
async def quiz_page(request: Request):
    active = quiz.get_active_set()
    return templates.TemplateResponse(request, "quiz.html", {"active_set": active})


@app.get("/results")
async def results_page(request: Request):
    payload = results.aggregate()
    return templates.TemplateResponse(request, "results.html", {"results": payload})


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


class AnswerIn(BaseModel):
    answer_letter: str = Field(pattern="^[ABCD]$")
    confidence: int = Field(ge=0, le=100)


@app.post("/api/sessions", status_code=201)
async def create_session(request: Request):
    return quiz.create_session(user_agent=request.headers.get("user-agent", ""))


@app.get("/api/sessions/{session_id}/resume")
async def resume_session(session_id: str):
    return {
        "position": quiz.first_unanswered_position(session_id),
        "summary": quiz.session_summary(session_id),
    }


@app.post("/api/sessions/{session_id}/questions/{position}")
async def present_question(session_id: str, position: int):
    payload, created = quiz.present_question(session_id, position)
    if created:
        # The race is on: fire the AI contestants for this presentation.
        session = quiz.get_session(session_id)
        question_row = db.fetch_one(
            "SELECT * FROM questions WHERE set_id = ? AND position = ?",
            (session["set_id"], position),
        )
        ai_runner.dispatch(payload["presentation_id"], dict(question_row))
    return payload


@app.post("/api/presentations/{presentation_id}/answer")
async def submit_answer(presentation_id: str, answer: AnswerIn):
    return quiz.submit_answer(presentation_id, answer.answer_letter, answer.confidence)


@app.get("/api/sessions/{session_id}/summary")
async def session_summary(session_id: str):
    return quiz.session_summary(session_id)


@app.get("/api/results")
async def api_results():
    return results.aggregate()
