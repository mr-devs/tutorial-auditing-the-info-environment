"""LLM-as-judge evaluation of generated multiple-choice questions.

Step 3 of the tutorial pipeline: a judge model reads the source article and
one question (options + marked correct answer) and returns one binary
judgment — ``faithful`` — plus a short rationale, validated by Pydantic. Judgments are appended incrementally to
JSONL with resume keyed on question id, optionally fanned out across a
``ThreadPoolExecutor``.

The script ``scripts/03-1_generate_judgments.py`` is run once per judge model;
``scripts/03-2_combine_judgments.py`` merges the per-model files into a tidy CSV,
and ``scripts/03-3_select_questions.py`` draws the final question set.

Typical use::

    from toolkit.judgments import judge_questions
    from toolkit.utils import load_jsonl

    questions = load_jsonl("data/questions/questions_gpt-5.6-terra.jsonl")
    articles_by_id = {a["id"]: a for a in load_jsonl("data/articles/guardian_articles.jsonl")}
    summary = judge_questions(
        questions,
        articles_by_id,
        output_fp="data/judgments/judgments_gpt-5.6-luna.jsonl",
        model="gpt-5.6-luna",
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

from pydantic import BaseModel, Field

from toolkit import config, prompts, providers

logger = logging.getLogger(__name__)


class JudgmentError(RuntimeError):
    """The judge model's response could not be parsed into the schema."""


class Judgment(BaseModel):
    """One judge model's binary verdicts on one question."""

    faithful: bool = Field(
        description=(
            "True if the MARKED correct option is stated or directly "
            "supported by the article text, and no other option is equally "
            "defensible given the article."
        )
    )
    rationale: str = Field(description="1-2 sentences explaining the verdict.")


def to_judgment_record(question: dict, parsed: Judgment, model: str) -> dict:
    """Flatten one validated judgment into a JSONL-ready dict.

    Both sides of the evaluation are recorded explicitly: ``judge_model``
    (who judged) and ``generator_provider``/``generator_model`` (whose
    question was judged, copied from the Step 2 question record).
    """
    return {
        "id": f"{model}__{question['id']}",
        "question_id": question["id"],
        "article_id": question["article_id"],
        "generator_provider": question.get("provider"),
        "generator_model": question.get("model"),
        "judge_model": model,
        "faithful": parsed.faithful,
        "rationale": parsed.rationale,
        "judged_at": datetime.now(timezone.utc).isoformat(),
    }


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


def load_completed_question_ids(output_fp) -> set:
    """Return the set of question ids already judged in a JSONL file.

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
                if record.get("question_id"):
                    ids.add(record["question_id"])
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", lineno, output_fp)
    return ids


def judge_question(question: dict, article: dict, *, model: str) -> dict:
    """One question -> one judge-model call -> a validated judgment record."""
    run_parsed = providers.get_run_parsed(config.JUDGE_MODELS[model])
    parsed, _raw = run_parsed(
        model,
        prompts.JUDGE_SYSTEM_PROMPT,
        prompts.build_judge_user_prompt(
            article["headline"],
            article["body_text"],
            question["question"],
            question["options"],
            question["correct_letter"],
        ),
        Judgment,
    )
    if parsed is None:
        raise JudgmentError(
            f"Response for question {question['id']!r} could not be parsed."
        )
    return to_judgment_record(question, parsed, model)


def judge_questions(
    questions: list,
    articles_by_id: dict,
    *,
    output_fp,
    model: str,
    parallel: bool = False,
    max_workers: int = 8,
    resume: bool = True,
) -> dict:
    """Judge many questions with one model and append results to JSONL.

    Orchestration mirrors ``toolkit.questions.generate_for_articles``: with
    ``parallel=True`` questions are submitted to a ``ThreadPoolExecutor``
    (the API calls are I/O-bound, so the waits overlap); the file is written
    only from the main thread, one judgment appended as each arrives
    (crash-safe). One failing question — including a question whose
    ``article_id`` is missing from ``articles_by_id`` — is logged and
    skipped; a re-run with ``resume=True`` retries exactly the failures.

    Returns a summary dict::

        {"judge_model", "judged", "skipped", "failed", "new_records",
         "elapsed_seconds", "output_fp"}
    """
    if model not in config.JUDGE_MODELS:
        raise ValueError(
            f"Unknown judge model: {model!r} "
            f"(expected one of {sorted(config.JUDGE_MODELS)})"
        )
    # Fail fast on a missing API key BEFORE spawning threads.
    run_parsed = providers.get_run_parsed(config.JUDGE_MODELS[model])
    warm_client = getattr(sys.modules[run_parsed.__module__], "_get_client", None)
    if warm_client is not None:
        warm_client()  # raises ValueError if no API key is set

    unique_questions = list({q["id"]: q for q in questions}.values())
    done = load_completed_question_ids(output_fp) if resume else set()
    todo = [q for q in unique_questions if q["id"] not in done]
    skipped = len(unique_questions) - len(todo)
    if skipped:
        logger.info("Resuming: skipping %d questions already in %s", skipped, output_fp)

    judged = failed = new_records = 0
    start = time.monotonic()

    def _one(question):
        article = articles_by_id.get(question["article_id"])
        if article is None:
            raise JudgmentError(
                f"Article {question['article_id']!r} not found in the "
                "articles file — pass the same articles JSONL the questions "
                "were generated from."
            )
        return judge_question(question, article, model=model)

    def _handle(question, record_or_error):
        nonlocal judged, failed, new_records
        if isinstance(record_or_error, Exception):
            failed += 1
            logger.error(
                "Failed to judge question %s: %s", question["id"], record_or_error
            )
            return
        new_records += append_records([record_or_error], output_fp)
        judged += 1
        logger.info("[%d/%d] judged %s", judged + failed, len(todo), question["id"])

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_one, q): q for q in todo}
            for future in as_completed(futures):
                question = futures[future]
                try:
                    _handle(question, future.result())
                except Exception as exc:  # noqa: BLE001 — isolate per-question errors
                    _handle(question, exc)
    else:
        for question in todo:
            try:
                _handle(question, _one(question))
            except Exception as exc:  # noqa: BLE001 — isolate per-question errors
                _handle(question, exc)

    return {
        "judge_model": model,
        "judged": judged,
        "skipped": skipped,
        "failed": failed,
        "new_records": new_records,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "output_fp": str(output_fp),
    }
