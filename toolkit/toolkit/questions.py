"""Multiple-choice question generation from news articles.

Production version of the workflow taught in
``notebooks/02_question_generation.ipynb``: one LLM call per article (OpenAI
or Gemini, structured outputs validated by Pydantic), optionally fanned out
across a ``ThreadPoolExecutor``, with incremental JSONL persistence and
resume keyed on article id.

Typical use::

    from toolkit.questions import generate_for_articles
    from toolkit.utils import load_jsonl

    articles = load_jsonl("data/articles/guardian_articles.jsonl")
    summary = generate_for_articles(
        articles,
        output_fp="data/questions/questions_openai.jsonl",
        provider="openai",
        parallel=True,
    )
"""

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from toolkit import config, prompts, providers

logger = logging.getLogger(__name__)


class QuestionGenerationError(RuntimeError):
    """The model's response could not be parsed into the expected schema."""


class MCQuestion(BaseModel):
    """One multiple-choice question grounded in a single news article."""

    question: str = Field(
        description="A self-contained question about a specific fact in the article."
    )
    options: list[str] = Field(
        description="Exactly 4 answer options, in the order A, B, C, D.",
        min_length=4,
        max_length=4,
    )
    correct_letter: Literal["A", "B", "C", "D"] = Field(
        description="The letter of the one correct option."
    )
    explanation: str = Field(
        description=(
            "1-3 sentences quoting or closely paraphrasing the article text "
            "that makes the correct option right."
        )
    )


class ArticleQuestions(BaseModel):
    """The structured response for one article: a list of MCQs."""

    questions: list[MCQuestion]


def _resolve_provider(provider: str):
    """Return ``(run_parsed, default_model)`` for a provider name."""
    defaults = {
        "openai": config.DEFAULT_OPENAI_MODEL,
        "gemini": config.DEFAULT_GEMINI_MODEL,
    }
    if provider not in defaults:
        raise ValueError(
            f"Unknown provider: {provider!r} (expected 'openai' or 'gemini')"
        )
    return providers.get_run_parsed(provider), defaults[provider]


def to_question_records(
    article: dict, parsed: ArticleQuestions, provider: str, model: str
) -> list[dict]:
    """Flatten one article's validated response into one dict per question."""
    generated_at = datetime.now(timezone.utc).isoformat()
    records = []
    for i, q in enumerate(parsed.questions):
        records.append(
            {
                "id": f"{provider}__{article['id']}__q{i}",
                "article_id": article["id"],
                "question_index": i,
                "provider": provider,
                "model": model,
                "question": q.question,
                "options": q.options,
                "correct_letter": q.correct_letter,
                "explanation": q.explanation,
                "generated_at": generated_at,
            }
        )
    return records


def append_records(records, output_fp) -> int:
    """Append records to a JSONL file (creating parent dirs). Returns count."""
    output_fp = Path(output_fp)
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_fp, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_completed_article_ids(output_fp) -> set:
    """Return the set of article ids that already have questions saved.

    Missing file -> empty set. Malformed lines are skipped with a warning.
    """
    output_fp = Path(output_fp)
    ids = set()
    if not output_fp.exists():
        return ids
    with open(output_fp, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("article_id"):
                    ids.add(record["article_id"])
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", lineno, output_fp)
    return ids


def generate_questions_for_article(
    article: dict,
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    n_questions: int = 3,
) -> list[dict]:
    """One article -> one LLM call -> a list of validated question records."""
    run_parsed, default_model = _resolve_provider(provider)
    model = model or default_model
    parsed, _raw = run_parsed(
        model,
        prompts.MCQ_SYSTEM_PROMPT,
        prompts.build_mcq_user_prompt(
            article["headline"], article["body_text"], n_questions
        ),
        ArticleQuestions,
    )
    if parsed is None:
        raise QuestionGenerationError(
            f"Response for article {article['id']!r} could not be parsed."
        )
    return to_question_records(article, parsed, provider, model)


def generate_for_articles(
    articles: list,
    *,
    output_fp,
    provider: str = "openai",
    model: Optional[str] = None,
    n_questions: int = 3,
    parallel: bool = False,
    max_workers: int = 8,
    resume: bool = True,
) -> dict:
    """Generate MCQs for many articles and append them to a JSONL file.

    With ``parallel=True``, articles are submitted to a ``ThreadPoolExecutor``
    (API calls are I/O-bound, so threads overlap the waiting); otherwise a
    plain sequential loop runs. Either way the file is written only from the
    main thread — worker threads just call the API and return records — and
    each article's questions are appended as soon as they arrive (crash-safe;
    Ctrl-C waits for in-flight calls, and everything completed is saved).

    One failing article does not stop the run: its error is logged, nothing
    is written for it, and a re-run with ``resume=True`` (which skips
    articles whose ids already appear in ``output_fp``) retries exactly the
    failures.

    Returns a summary dict::

        {"provider", "model", "articles_processed", "articles_skipped",
         "articles_failed", "new_questions", "elapsed_seconds", "output_fp"}
    """
    # Fail fast on an unknown provider or missing API key BEFORE spawning
    # threads (otherwise every worker raises the same error). Key loading is
    # lazy inside each provider's _get_client, so touch it once up front.
    run_parsed, default_model = _resolve_provider(provider)
    model = model or default_model
    warm_client = getattr(sys.modules[run_parsed.__module__], "_get_client", None)
    if warm_client is not None:
        warm_client()  # raises ValueError if no API key is set

    unique_articles = list({a["id"]: a for a in articles}.values())
    done = load_completed_article_ids(output_fp) if resume else set()
    todo = [a for a in unique_articles if a["id"] not in done]
    skipped = len(unique_articles) - len(todo)
    if skipped:
        logger.info("Resuming: skipping %d articles already in %s", skipped, output_fp)

    processed = failed = new_questions = 0
    start = time.monotonic()

    def _one(article):
        return generate_questions_for_article(
            article, provider=provider, model=model, n_questions=n_questions
        )

    def _handle(article, records_or_error):
        nonlocal processed, failed, new_questions
        if isinstance(records_or_error, Exception):
            failed += 1
            logger.error(
                "Failed to generate questions for %s: %s",
                article["id"],
                records_or_error,
            )
            return
        new_questions += append_records(records_or_error, output_fp)
        processed += 1
        logger.info(
            "[%d/%d] %s -> %d questions",
            processed + failed,
            len(todo),
            article["id"],
            len(records_or_error),
        )

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_one, a): a for a in todo}
            for future in as_completed(futures):
                article = futures[future]
                try:
                    _handle(article, future.result())
                except Exception as exc:  # noqa: BLE001 — isolate per-article errors
                    _handle(article, exc)
    else:
        for article in todo:
            try:
                _handle(article, _one(article))
            except Exception as exc:  # noqa: BLE001 — isolate per-article errors
                _handle(article, exc)

    return {
        "provider": provider,
        "model": model,
        "articles_processed": processed,
        "articles_skipped": skipped,
        "articles_failed": failed,
        "new_questions": new_questions,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "output_fp": str(output_fp),
    }
