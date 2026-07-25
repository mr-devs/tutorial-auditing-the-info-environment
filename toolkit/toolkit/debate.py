"""Society-of-minds debate answering, built on the openai-agents SDK.

Step 4's third condition: ``config.DEBATE_N_AGENTS`` agents backed by the
SAME model answer a question independently (closed book), then revise over
``config.DEBATE_N_ROUNDS`` rounds after reading the other agents' answers;
the final answer is a majority vote. Costs ``K * (1 + R)`` LLM calls per
question (3 * 3 = 9 with the defaults).

Unlike ``toolkit.answers`` (which calls the provider adapters directly),
the debaters here are openai-agents ``Agent`` objects run through
``Runner.run``. The SDK targets OpenAI's Responses API, so only the GPT
models in ``config.DEBATE_MODELS`` participate in this condition. The
round structure is still orchestrated deterministically in Python — a
debate's turn order is fixed, so there is nothing for LLM-driven handoffs
to decide. Structured outputs come from ``output_type=Answer``: the SDK
validates each turn against the same Pydantic schema Steps 2-4 use.

Records share ``toolkit.answers``' shape (``method="debate"``, plus a
nested ``debate`` dict with the full transcript), so the combine script
treats both sources identically.

The SDK is async-native, so the orchestrator is an ``asyncio`` event loop
rather than a thread pool: agents within a round run concurrently via
``asyncio.gather``, and ``parallel=True`` additionally overlaps whole
questions under a semaphore. Appends still happen only in the main task
as each question finishes (crash-safe), with resume keyed on question id.

Typical use::

    from toolkit.debate import debate_questions
    from toolkit.utils import load_jsonl

    questions = load_jsonl("data/questions/selected_questions.jsonl")
    summary = debate_questions(
        questions,
        output_fp="data/answers/answers_debate_gpt-5.6-terra.jsonl",
        model="gpt-5.6-terra",
        parallel=True,
    )
"""

import asyncio
import logging
import time

from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from toolkit import config, prompts
from toolkit.answers import (
    Answer,
    AnswerError,
    append_records,
    load_completed_question_ids,
    to_answer_record,
)
from toolkit.providers import PROVIDER_ENV, load_api_key
from toolkit.providers.openai_provider import RETRYABLE_EXCEPTIONS

logger = logging.getLogger(__name__)


def _configure_sdk() -> None:
    """Point the openai-agents SDK at our key with a loop-fresh client.

    The SDK's default client is created once and bound to the first event
    loop that uses it; installing a fresh ``AsyncOpenAI`` before each
    ``asyncio.run`` avoids closed-loop errors on repeated runs. Also
    resolves the key through :func:`load_api_key` (``SML_`` fallback) and
    fails fast if none is set, and disables the SDK's tracing uploads.
    """
    set_default_openai_client(AsyncOpenAI(api_key=load_api_key(PROVIDER_ENV["openai"])))
    set_tracing_disabled(True)


def _aggregate_votes(final_round: list):
    """Aggregate the final debate round into one deterministic answer.

    The rule: (1) the letter with the most votes wins (with 3 agents a
    unique leader always holds a strict majority); (2) on a tie for the
    most votes, the tied letter with the highest MEAN self-reported
    confidence wins; (3) on a residual tie, the alphabetically first
    letter wins.

    Parameters
    ----------
    final_round : list of dict
        One entry per agent, each with ``"answer_letter"`` and
        ``"confidence"``.

    Returns
    -------
    tuple of (str, dict, str or None)
        ``(letter, vote_counts, tie_break)`` where ``tie_break`` is
        ``None`` when a single letter had the most votes, ``"confidence"``
        when mean confidence broke the tie, and ``"alphabetical"`` when
        the final fallback was needed.
    """
    vote_counts: dict = {}
    for entry in final_round:
        vote_counts[entry["answer_letter"]] = (
            vote_counts.get(entry["answer_letter"], 0) + 1
        )
    top = max(vote_counts.values())
    leaders = sorted(letter for letter, n in vote_counts.items() if n == top)
    if len(leaders) == 1:
        return leaders[0], vote_counts, None

    def mean_confidence(letter):
        scores = [e["confidence"] for e in final_round if e["answer_letter"] == letter]
        return sum(scores) / len(scores)

    best = max(mean_confidence(letter) for letter in leaders)
    confident = [letter for letter in leaders if mean_confidence(letter) == best]
    if len(confident) == 1:
        return confident[0], vote_counts, "confidence"
    return min(confident), vote_counts, "alphabetical"


@retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _run_turn(agent: Agent, user_prompt: str, question_id: str) -> Answer:
    """Run one retried agent turn and return its schema-validated ``Answer``.

    Same transient-error retry policy as the provider adapters' ``_call``
    (429/connection/5xx only, exponential backoff, 5 attempts) — a debate
    costs 9 calls, so one blip should not fail the whole question.
    """
    result = await Runner.run(agent, user_prompt)
    answer = result.final_output
    if not isinstance(answer, Answer):
        raise AnswerError(
            f"Debate response for question {question_id!r} could not be parsed."
        )
    return answer


async def _debate_async(question: dict, *, model: str) -> dict:
    """Run one debate and build its answer record (async core).

    Round 0: the independent-answer agents respond concurrently
    (``asyncio.gather``). Rounds 1..R: reviser agents respond
    concurrently, each shown the OTHER agents' previous-round answers.

    Parameters
    ----------
    question : dict
        Question record with ``"id"``, ``"article_id"``, ``"question"``,
        ``"options"``, and ``"correct_letter"``.
    model : str
        Contestant model name; must be a key of ``config.DEBATE_MODELS``.

    Returns
    -------
    dict
        A validated answer record (``method="debate"``), as built by
        :func:`toolkit.answers.to_answer_record`. The final ``Answer``
        takes its letter from the vote, its confidence from the mean
        confidence of the winning agents, and its reasoning from the
        highest-confidence winning agent's final reasoning.

    Raises
    ------
    AnswerError
        If any agent's response could not be parsed — the whole question
        fails and is retried on resume.
    """
    n_agents = config.DEBATE_N_AGENTS
    n_rounds = config.DEBATE_N_ROUNDS

    answerer = Agent(
        name="Debater",
        model=model,
        instructions=prompts.ANSWER_SYSTEM_PROMPT,
        output_type=Answer,
    )
    reviser = Agent(
        name="Reviser",
        model=model,
        instructions=prompts.DEBATE_REVISION_SYSTEM_PROMPT,
        output_type=Answer,
    )

    def entry(agent_idx, answer):
        return {
            "agent": agent_idx,
            "answer_letter": answer.answer_letter,
            "confidence": answer.confidence,
            "reasoning": answer.reasoning,
        }

    # Round 0: independent answers, all agents concurrently.
    first_prompt = prompts.build_answer_user_prompt(
        question["question"], question["options"]
    )
    answers = await asyncio.gather(
        *(_run_turn(answerer, first_prompt, question["id"]) for _ in range(n_agents))
    )
    transcript = [[entry(i, a) for i, a in enumerate(answers)]]

    # Revision rounds: each agent sees the others' previous-round answers.
    for _ in range(n_rounds):
        previous = transcript[-1]
        turns = []
        for agent_idx in range(n_agents):
            peers = [p for p in previous if p["agent"] != agent_idx]
            turns.append(
                _run_turn(
                    reviser,
                    prompts.build_debate_revision_prompt(
                        question["question"], question["options"], peers
                    ),
                    question["id"],
                )
            )
        answers = await asyncio.gather(*turns)
        transcript.append([entry(i, a) for i, a in enumerate(answers)])

    letter, vote_counts, tie_break = _aggregate_votes(transcript[-1])
    winners = [e for e in transcript[-1] if e["answer_letter"] == letter]
    best = max(winners, key=lambda e: e["confidence"])
    final = Answer(
        answer_letter=letter,
        confidence=sum(e["confidence"] for e in winners) / len(winners),
        reasoning=best["reasoning"],
    )
    debate = {
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "transcript": transcript,
        "vote_counts": vote_counts,
        "tie_break": tie_break,
    }
    return to_answer_record(
        question, final, model=model, method="debate", debate=debate
    )


async def debate_question_async(question: dict, *, model: str) -> dict:
    """Run one society-of-minds debate for one question (async).

    Use this from an already-running event loop — e.g. a Jupyter notebook,
    where IPython supports top-level ``await``::

        record = await debate_question_async(question, model="gpt-5.6-terra")

    Parameters
    ----------
    question : dict
        Question record with ``"id"``, ``"article_id"``, ``"question"``,
        ``"options"``, and ``"correct_letter"``.
    model : str
        Contestant model name; must be a key of ``config.DEBATE_MODELS``.

    Returns
    -------
    dict
        A validated answer record with ``method="debate"`` and the full
        per-round transcript in its ``debate`` field.

    Raises
    ------
    AnswerError
        If any agent's response could not be parsed.
    ValueError
        If ``model`` is not a known debate model or no OpenAI API key is
        set.
    """
    if model not in config.DEBATE_MODELS:
        raise ValueError(
            f"Unknown debate model: {model!r} "
            f"(expected one of {sorted(config.DEBATE_MODELS)})"
        )
    _configure_sdk()
    return await _debate_async(question, model=model)


def debate_question(question: dict, *, model: str) -> dict:
    """Run one society-of-minds debate for one question (sync wrapper).

    Starts its own event loop, so it cannot be called from code that is
    already inside one — in a Jupyter notebook, ``await``
    :func:`debate_question_async` instead.

    Parameters
    ----------
    question : dict
        Question record with ``"id"``, ``"article_id"``, ``"question"``,
        ``"options"``, and ``"correct_letter"``.
    model : str
        Contestant model name; must be a key of ``config.DEBATE_MODELS``.

    Returns
    -------
    dict
        A validated answer record with ``method="debate"`` and the full
        per-round transcript in its ``debate`` field.

    Raises
    ------
    AnswerError
        If any agent's response could not be parsed.
    ValueError
        If ``model`` is not a known debate model or no OpenAI API key is
        set.
    """
    return asyncio.run(debate_question_async(question, model=model))


def debate_questions(
    questions: list,
    *,
    output_fp,
    model: str,
    parallel: bool = False,
    max_workers: int = 4,
    resume: bool = True,
) -> dict:
    """Debate many questions with one model and append results to JSONL.

    Parameters
    ----------
    questions : list of dict
        Step 3 question records. Duplicate ids are dropped before
        processing.
    output_fp : str or Path
        Path of the JSONL file answer records are appended to.
    model : str
        Contestant model name; must be a key of ``config.DEBATE_MODELS``.
    parallel : bool, default False
        Overlap multiple questions' debates concurrently.
    max_workers : int, default 4
        Maximum concurrent debates when ``parallel=True``. Each debate
        already runs its agents concurrently within a round, so N
        debates burst up to ``N * DEBATE_N_AGENTS`` simultaneous calls.
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
        If ``model`` is not a known debate model or no OpenAI API key is
        set.

    Notes
    -----
    Same contract as ``toolkit.answers.answer_questions``, different
    engine: the openai-agents SDK is async-native, so concurrency comes
    from one ``asyncio`` event loop (a semaphore caps in-flight debates)
    instead of a thread pool. Records are appended one at a time from the
    main task as each debate finishes (crash-safe); one failing question
    is logged and skipped, and a re-run with ``resume=True`` retries
    exactly the failures.
    """
    if model not in config.DEBATE_MODELS:
        raise ValueError(
            f"Unknown debate model: {model!r} "
            f"(expected one of {sorted(config.DEBATE_MODELS)})"
        )
    _configure_sdk()  # raises ValueError if no API key is set

    unique_questions = list({q["id"]: q for q in questions}.values())
    done = load_completed_question_ids(output_fp) if resume else set()
    todo = [q for q in unique_questions if q["id"] not in done]
    skipped = len(unique_questions) - len(todo)
    if skipped:
        logger.info("Resuming: skipping %d questions already in %s", skipped, output_fp)

    answered = failed = new_records = 0
    start = time.monotonic()

    async def _run_all():
        nonlocal answered, failed, new_records
        semaphore = asyncio.Semaphore(max_workers if parallel else 1)

        async def _one(question):
            async with semaphore:
                return question, await _debate_async(question, model=model)

        tasks = [asyncio.create_task(_one(q)) for q in todo]
        for future in asyncio.as_completed(tasks):
            try:
                question, record = await future
            except Exception as exc:  # noqa: BLE001 — isolate per-question errors
                failed += 1
                logger.error("Failed to debate a question: %s", exc)
                continue
            new_records += append_records([record], output_fp)
            answered += 1
            logger.info(
                "[%d/%d] debated %s", answered + failed, len(todo), question["id"]
            )

    asyncio.run(_run_all())

    return {
        "model": model,
        "method": "debate",
        "answered": answered,
        "skipped": skipped,
        "failed": failed,
        "new_records": new_records,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "output_fp": str(output_fp),
    }
