"""
Purpose: Have one LLM contestant model answer the vetted multiple-choice
questions (from Step 3) under one single-call answering condition —
closed_book or web_search — writing one JSONL answer per question.

Run this script once per method x model pair. The third condition, the
multi-agent debate, has its own script
(scripts/04-2_generate_debate_answers.py); merge all per-run files with
scripts/04-3_combine_answers.py.

Examples
--------
# One model, closed book, small demo
uv run python scripts/04-1_generate_answers.py \
    --model gpt-5.4-mini-2026-03-17 --method closed_book \
    --max-questions 5 --parallel

# The full sweep: both methods x every model, threaded
for METHOD in closed_book web_search; do
  for M in gpt-5.4-mini-2026-03-17 gpt-5.5-2026-04-23 gpt-5.6-luna \
           gpt-5.6-terra gemini-3.1-flash-lite gemini-3.5-flash; do
    uv run python scripts/04-1_generate_answers.py \
        --model $M --method $METHOD --parallel
  done
done

Inputs
------
- --questions: vetted questions JSONL from Step 3
  (default data/questions/selected_questions.jsonl)
- --model: the contestant model (required). One of: gemini-3.1-flash-lite,
  gemini-3.5-flash, gpt-5.4-mini-2026-03-17, gpt-5.5-2026-04-23,
  gpt-5.6-luna, gpt-5.6-terra
- --method: the answering condition (required): closed_book (no article,
  no tools) or web_search (provider search tool on); one LLM call per
  question either way
- --max-questions: cap for demos (default: all)
- --parallel: submit multiple questions in parallel via a ThreadPoolExecutor
- --max-workers: threads when --parallel (default 8)
- --output: output JSONL path
  (default data/answers/answers_<method>_<model>.jsonl)
- --no-resume: skip the dedup/resume pass and re-answer everything
- API keys: via SML_OPENAI_API_KEY / OPENAI_API_KEY and
  SML_GEMINI_API_KEY / GEMINI_API_KEY (SML_ variant wins)
- --log-level, --log-file: logging controls
- --create-log-file: also log to a datetime-stamped file under logs/

Outputs
-------
A JSONL file (default data/answers/answers_<method>_<model>.jsonl), one
answer per line with keys: id, question_id, article_id, method, model,
provider, answer_letter, correct_letter, is_correct (graded on the spot),
confidence, reasoning, citations (web_search only — the model's
self-reported source URLs, requested via prompts.ANSWER_WEBSEARCH_ADDENDUM),
search_used (web_search only), answered_at. The contestant sees
ONLY the question and its lettered options — never the article. Appended
incrementally (crash-safe); re-running the same command skips questions
already answered.

Author: Matthew DeVerna
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

from toolkit import config
from toolkit.answers import answer_questions
from toolkit.utils import load_jsonl, resolve_path, setup_logging

DEFAULT_QUESTIONS = f"{config.QUESTIONS_DIR}/selected_questions.jsonl"


def rel_to_root(path) -> str:
    """Show a path relative to the repo root (keeps --help output readable)."""
    return os.path.relpath(path, config.REPO_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inputs = parser.add_argument_group("input")
    inputs.add_argument(
        "--questions",
        default=DEFAULT_QUESTIONS,
        metavar="PATH",
        help=(
            "Path to the vetted questions .jsonl file produced by "
            "scripts/03-3_select_questions.py; relative paths resolve from "
            f"the repo root (default: {rel_to_root(DEFAULT_QUESTIONS)})."
        ),
    )

    answering = parser.add_argument_group("answering")
    answering.add_argument(
        "--model",
        choices=sorted(config.ANSWER_MODELS),
        required=True,
        help=(
            "The contestant model. Run the script once per method x model "
            "pair; scripts/04-3_combine_answers.py merges the outputs."
        ),
    )
    answering.add_argument(
        "--method",
        choices=list(config.ANSWER_METHODS),
        required=True,
        help=(
            "The answering condition: closed_book (question + options only) "
            "or web_search (the same prompt plus a citation addendum, with "
            "the provider's search tool on) — one LLM call per question "
            "either way. The debate "
            "condition has its own script, "
            "scripts/04-2_generate_debate_answers.py."
        ),
    )
    answering.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help=(
            "Only answer the first N questions. Keep small for demos — every "
            "question costs one LLM call (default: all)."
        ),
    )

    concurrency = parser.add_argument_group("concurrency")
    concurrency.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Include to submit multiple questions in parallel via a "
            "ThreadPoolExecutor instead of a sequential loop. Most of each "
            "API call is spent waiting for the provider's response, so "
            "several calls can wait at the same time — much faster overall."
        ),
    )
    concurrency.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help=(
            "Thread count when --parallel is set (default: %(default)s). "
            "More is not always faster once you hit the provider's rate limit."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Path for the output .jsonl file, appended to incrementally; "
            "relative paths resolve from the repo root "
            "(default: data/answers/answers_<method>_<model>.jsonl)."
        ),
    )
    output.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Skip the resume/dedup pass. By default the script reads the "
            "output file first and skips questions already answered, so "
            "re-running the same command is safe."
        ),
    )

    ops = parser.add_argument_group("operational")
    ops.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console logging verbosity (default: %(default)s).",
    )
    log_dest = ops.add_mutually_exclusive_group()
    log_dest.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to a file to also write logs to (appended); relative "
            "paths resolve from the repo root."
        ),
    )
    log_dest.add_argument(
        "--create-log-file",
        action="store_true",
        help=(
            "Also write logs to an auto-named, datetime-stamped file, e.g. "
            "logs/generate_answers_2026-07-16_14-03-27.log."
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.questions = resolve_path(args.questions)
    if args.output:
        args.output = resolve_path(args.output)
    if args.log_file:
        args.log_file = resolve_path(args.log_file)

    output_fp = (
        args.output or f"{config.ANSWERS_DIR}/answers_{args.method}_{args.model}.jsonl"
    )

    log_file = args.log_file
    if args.create_log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"{config.LOGS_DIR}/generate_answers_{timestamp}.log"

    setup_logging(
        log_level=args.log_level,
        log_file=log_file,
        console_output=True,
        append_mode=True,
    )
    logger = logging.getLogger("generate_answers")
    if log_file:
        logger.info("Logging to %s", rel_to_root(log_file))

    if not Path(args.questions).exists():
        logger.error(
            "Questions file %s not found — run scripts/03-3_select_questions.py first.",
            rel_to_root(args.questions),
        )
        return 1

    questions = load_jsonl(args.questions)
    if args.max_questions is not None:
        questions = questions[: args.max_questions]

    logger.info(
        "Answering %d questions with %s under %s",
        len(questions),
        args.model,
        args.method,
    )

    try:
        summary = answer_questions(
            questions,
            output_fp=output_fp,
            model=args.model,
            method=args.method,
            parallel=args.parallel,
            max_workers=args.max_workers,
            resume=not args.no_resume,
        )
    except ValueError as e:
        # Missing API key (message names both env vars) or bad model/method.
        logger.error(str(e))
        return 1
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted — completed answers are already saved to %s; "
            "re-run the same command to resume.",
            rel_to_root(output_fp),
        )
        return 130

    logger.info(
        "Done: %d answers by %s under %s (%d skipped, %d failed) in %.1fs -> %s",
        summary["new_records"],
        summary["model"],
        summary["method"],
        summary["skipped"],
        summary["failed"],
        summary["elapsed_seconds"],
        rel_to_root(summary["output_fp"]),
    )
    if summary["failed"]:
        logger.warning(
            "%d questions failed — re-run the same command to retry just those.",
            summary["failed"],
        )
        if summary["answered"] == 0 and summary["skipped"] == 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
