"""LLM contestants answering the vetted multiple-choice questions.

Step 4 of the tutorial pipeline: a contestant model answers each Step 3
question under one of the two single-call conditions ("methods"):

- ``closed_book`` — the question and options only; no article, no tools.
  Measures what the model knows from its weights alone.
- ``web_search`` — the same prompt with the provider's web-search tool
  enabled. Measures what retrieval adds.

The third condition, the multi-agent **debate**, lives in
``toolkit.debate`` (built on the openai-agents SDK) and shares this
module's ``Answer`` schema, record shape, and persistence helpers.

Answers are appended incrementally to JSONL with resume keyed on question
id, optionally fanned out across a ``ThreadPoolExecutor``.

The script ``scripts/04-1_generate_answers.py`` is run once per
method x model pair; ``scripts/04-3_combine_answers.py`` merges the
per-run files (including the debate script's) into a tidy CSV.

Typical use::

    from toolkit.answers import answer_questions
    from toolkit.utils import load_jsonl

    questions = load_jsonl("data/questions/selected_questions.jsonl")
    summary = answer_questions(
        questions,
        output_fp="data/answers/answers_closed_book_gpt-5.6-terra.jsonl",
        model="gpt-5.6-terra",
        method="closed_book",
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
from typing import Literal

from pydantic import BaseModel, Field

from toolkit import config, prompts, providers

logger = logging.getLogger(__name__)


class AnswerError(RuntimeError):
    """The contestant model's response could not be parsed into the schema."""


class Answer(BaseModel):
    """One model's structured answer to one multiple-choice question.

    Field semantics live in the ``Field`` descriptions below.

    Attributes
    ----------
    answer_letter : {"A", "B", "C", "D"}
        The single option chosen as the best answer.
    confidence : float
        Self-reported confidence in the chosen letter, between 0 and 1.
    reasoning : str
        1-2 sentences explaining the choice.
    """

    answer_letter: Literal["A", "B", "C", "D"] = Field(
        description="The single best answer option: exactly one of A, B, C, or D."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence in the chosen letter, between 0 and 1.",
    )
    reasoning: str = Field(description="1-2 sentences explaining the choice.")


def to_answer_record(
    question: dict,
    parsed: Answer,
    *,
    model: str,
    method: str,
    search_used=None,
    debate=None,
    raw=None,
) -> dict:
    """Flatten one validated answer into a JSONL-ready dict.

    The question's ``correct_letter`` is copied in and graded on the spot
    (``is_correct``) so each record is self-contained for analysis.

    Parameters
    ----------
    question : dict
        The Step 3 question record that was answered.
    parsed : Answer
        The contestant model's validated structured response (for debate,
        the synthesized majority-vote answer).
    model : str
        Name of the contestant model.
    method : str
        The answering condition: one of ``config.ANSWER_METHODS``, or
        ``"debate"`` (used by ``toolkit.debate``).
    search_used : bool, optional
        Whether the raw response shows the web-search tool was actually
        invoked. Only set for the ``web_search`` method.
    debate : dict, optional
        Full debate state (transcript, vote counts, tie-break rule used).
        Only set by ``toolkit.debate``.
    raw : dict, optional
        Full raw provider response as returned by ``run_parsed``.
        ``None`` for debate records.

    Returns
    -------
    dict
        JSONL-ready answer record with a unique ``id`` of the form
        ``{method}__{model}__{question_id}``. Every field is scalar
        except ``debate`` and ``raw``.
    """
    return {
        "id": f"{method}__{model}__{question['id']}",
        "question_id": question["id"],
        "article_id": question["article_id"],
        "method": method,
        "model": model,
        "provider": config.ANSWER_MODELS[model],
        "answer_letter": parsed.answer_letter,
        "correct_letter": question["correct_letter"],
        "is_correct": parsed.answer_letter == question["correct_letter"],
        "confidence": parsed.confidence,
        "reasoning": parsed.reasoning,
        "search_used": search_used,
        "debate": debate,
        "raw": raw,
        "answered_at": datetime.now(timezone.utc).isoformat(),
    }


def append_records(records, output_fp) -> int:
    """Append records to a JSONL file, creating parent directories as needed.

    Parameters
    ----------
    records : iterable of dict
        Records to append, one JSON object per line.
    output_fp : str or Path
        Path of the JSONL file to append to.

    Returns
    -------
    int
        Number of records written.
    """
    output_fp = Path(output_fp)
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_fp, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_completed_question_ids(output_fp) -> set:
    """Return the set of question ids already answered in a JSONL file.

    Safe as a resume key because each output file holds exactly one
    method x model pair.

    Parameters
    ----------
    output_fp : str or Path
        Path of the answers JSONL file to scan.

    Returns
    -------
    set
        Question ids found in the file. Empty if the file does not exist.

    Notes
    -----
    Malformed lines are skipped with a warning.
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


def _detect_search_use(provider: str, raw: dict) -> bool:
    """Return True if the raw response shows the search tool was invoked.

    Parameters
    ----------
    provider : str
        Provider name: 'openai' or 'gemini'.
    raw : dict
        The raw response dict returned by ``run_parsed``.

    Returns
    -------
    bool
        True if a web-search call (OpenAI) or search grounding (Gemini)
        appears in the response; False otherwise, including when the
        expected structure is absent.
    """
    try:
        if provider == "openai":
            return any(
                item.get("type") == "web_search_call" for item in raw.get("output", [])
            )
        if provider == "gemini":
            candidates = raw.get("candidates") or []
            if not candidates:
                return False
            metadata = candidates[0].get("grounding_metadata") or {}
            return bool(metadata.get("web_search_queries"))
    except (AttributeError, TypeError):
        return False
    return False


def _answer_direct(
    question: dict,
    *,
    model: str,
    use_web_search: bool,
    temperature: float | None = None,
    include_domains: list | None = None,
    exclude_domains: list | None = None,
):
    """Answer one question with one model call.

    Shared by the ``closed_book`` (``use_web_search=False``) and
    ``web_search`` (``use_web_search=True``) methods.

    Parameters
    ----------
    question : dict
        Question record with ``"id"``, ``"question"``, and ``"options"``.
    model : str
        Contestant model name; must be a key of ``config.ANSWER_MODELS``.
    use_web_search : bool
        Enable the provider's web-search tool for this call.
    temperature : float or None, default None
        Sampling temperature. ``None`` uses the provider default.
    include_domains : list of str or None, default None
        Restrict web search to these domains (OpenAI models only).
    exclude_domains : list of str or None, default None
        Exclude these domains from web search.

    Returns
    -------
    tuple of (Answer, dict)
        The validated answer and the raw response dict.

    Raises
    ------
    AnswerError
        If the model's response could not be parsed into the schema.
    """
    run_parsed = providers.get_run_parsed(config.ANSWER_MODELS[model])
    parsed, raw = run_parsed(
        model,
        prompts.ANSWER_SYSTEM_PROMPT,
        prompts.build_answer_user_prompt(question["question"], question["options"]),
        Answer,
        use_web_search=use_web_search,
        temperature=temperature,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )
    if parsed is None:
        raise AnswerError(
            f"Response for question {question['id']!r} could not be parsed."
        )
    return parsed, raw


def answer_question(
    question: dict,
    *,
    model: str,
    method: str,
    temperature: float | None = None,
    include_domains: list | None = None,
    exclude_domains: list | None = None,
) -> dict:
    """Answer one question under one condition and build its record.

    Parameters
    ----------
    question : dict
        Question record with ``"id"``, ``"article_id"``, ``"question"``,
        ``"options"``, and ``"correct_letter"``.
    model : str
        Contestant model name; must be a key of ``config.ANSWER_MODELS``.
    method : str
        The answering condition, one of ``config.ANSWER_METHODS``.
    temperature : float or None, default None
        Sampling temperature passed to the provider. ``None`` (the
        default) uses the provider default. GPT-5-series reasoning
        models reject non-default values; Gemini accepts 0.0–2.0.
    include_domains : list of str or None, default None
        Restrict web search to these domains (``web_search`` method,
        OpenAI models only; omit the URL scheme, e.g.
        ``["theguardian.com"]``).
    exclude_domains : list of str or None, default None
        Exclude these domains from web search (``web_search`` method,
        OpenAI models only — the Gemini Developer API has no domain
        filters).

    Returns
    -------
    dict
        A validated answer record, as built by :func:`to_answer_record`.

    Raises
    ------
    AnswerError
        If any model response could not be parsed into the schema.
    ValueError
        If ``method`` is not a known answering method, or a domain
        filter is combined with ``closed_book`` / an unsupported
        provider.
    """
    if method == "closed_book":
        parsed, raw = _answer_direct(
            question,
            model=model,
            use_web_search=False,
            temperature=temperature,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        return to_answer_record(question, parsed, model=model, method=method, raw=raw)
    if method == "web_search":
        parsed, raw = _answer_direct(
            question,
            model=model,
            use_web_search=True,
            temperature=temperature,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        search_used = _detect_search_use(config.ANSWER_MODELS[model], raw)
        return to_answer_record(
            question,
            parsed,
            model=model,
            method=method,
            search_used=search_used,
            raw=raw,
        )
    raise ValueError(
        f"Unknown method: {method!r} (expected one of {config.ANSWER_METHODS})"
    )


def answer_questions(
    questions: list,
    *,
    output_fp,
    model: str,
    method: str,
    parallel: bool = False,
    max_workers: int = 8,
    resume: bool = True,
) -> dict:
    """Answer many questions with one model x method and append to JSONL.

    Parameters
    ----------
    questions : list of dict
        Step 3 question records. Duplicate ids are dropped before
        processing.
    output_fp : str or Path
        Path of the JSONL file answer records are appended to.
    model : str
        Contestant model name; must be a key of ``config.ANSWER_MODELS``.
    method : str
        The answering condition, one of ``config.ANSWER_METHODS``.
    parallel : bool, default False
        Fan questions out across a ``ThreadPoolExecutor``.
    max_workers : int, default 8
        Thread-pool size when ``parallel=True``.
    resume : bool, default True
        Skip questions whose ids already appear in ``output_fp``.

    Returns
    -------
    dict
        Summary dict::

            {"model", "method", "answered", "skipped", "failed",
             "new_records", "elapsed_seconds", "output_fp"}

    Raises
    ------
    ValueError
        If ``model`` or ``method`` is unknown, or the provider's API key
        is not set.

    Notes
    -----
    Orchestration mirrors ``toolkit.judgments.judge_questions``: with
    ``parallel=True`` questions are submitted to a ``ThreadPoolExecutor``
    (the API calls are I/O-bound, so the waits overlap); the file is
    written only from the main thread, one answer appended as each
    arrives (crash-safe). One failing question is logged and skipped; a
    re-run with ``resume=True`` retries exactly the failures.
    """
    if model not in config.ANSWER_MODELS:
        raise ValueError(
            f"Unknown answer model: {model!r} "
            f"(expected one of {sorted(config.ANSWER_MODELS)})"
        )
    if method not in config.ANSWER_METHODS:
        raise ValueError(
            f"Unknown method: {method!r} (expected one of {config.ANSWER_METHODS})"
        )
    # Fail fast on a missing API key BEFORE spawning threads.
    run_parsed = providers.get_run_parsed(config.ANSWER_MODELS[model])
    warm_client = getattr(sys.modules[run_parsed.__module__], "_get_client", None)
    if warm_client is not None:
        warm_client()  # raises ValueError if no API key is set

    unique_questions = list({q["id"]: q for q in questions}.values())
    done = load_completed_question_ids(output_fp) if resume else set()
    todo = [q for q in unique_questions if q["id"] not in done]
    skipped = len(unique_questions) - len(todo)
    if skipped:
        logger.info("Resuming: skipping %d questions already in %s", skipped, output_fp)

    answered = failed = new_records = 0
    start = time.monotonic()

    def _one(question):
        """Answer one question under the run's method."""
        return answer_question(question, model=model, method=method)

    def _handle(question, record_or_error):
        """Record one question's outcome: append its record or log its error."""
        nonlocal answered, failed, new_records
        if isinstance(record_or_error, Exception):
            failed += 1
            logger.error(
                "Failed to answer question %s: %s", question["id"], record_or_error
            )
            return
        new_records += append_records([record_or_error], output_fp)
        answered += 1
        logger.info("[%d/%d] answered %s", answered + failed, len(todo), question["id"])

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
        "model": model,
        "method": method,
        "answered": answered,
        "skipped": skipped,
        "failed": failed,
        "new_records": new_records,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "output_fp": str(output_fp),
    }
