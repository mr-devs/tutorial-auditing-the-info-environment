"""
Purpose: Have one OpenAI GPT model answer the vetted multiple-choice
questions (from Step 3) through a multi-agent society-of-minds DEBATE,
writing one JSONL answer (with the full transcript) per question.

Three agents backed by the same model — openai-agents SDK Agents run
through Runner.run — answer independently, then revise over two rounds
after reading each other's answers; a majority vote decides (ties break by
mean confidence, then alphabetically). That is 9 LLM calls per question.
Run this script once per model, then merge the per-run files with
scripts/04-3_combine_answers.py. The single-call conditions (closed_book,
web_search) live in scripts/04-1_generate_answers.py.

Examples
--------
# One model, small demo
uv run python scripts/04-2_generate_debate_answers.py \
    --model gpt-5.4-mini-2026-03-17 --max-questions 5 --parallel

# All four GPT contestants
for M in gpt-5.4-mini-2026-03-17 gpt-5.5-2026-04-23 \
         gpt-5.6-luna gpt-5.6-terra; do
  uv run python scripts/04-2_generate_debate_answers.py \
      --model $M --parallel
done

Inputs
------
- --questions: vetted questions JSONL from Step 3
  (default data/questions/selected_questions.jsonl)
- --model: the contestant model (required). One of:
  gpt-5.4-mini-2026-03-17, gpt-5.5-2026-04-23, gpt-5.6-luna, gpt-5.6-terra
  (the openai-agents SDK targets OpenAI models, so Gemini sits this
  condition out)
- --max-questions: cap for demos — every question costs 9 LLM calls
  (default: all)
- --parallel: overlap multiple questions' debates concurrently (asyncio)
- --max-workers: concurrent debates when --parallel (default 4; each
  debate already runs its 3 agents concurrently within a round)
- --output: output JSONL path
  (default data/answers/answers_debate_<model>.jsonl)
- --no-resume: skip the dedup/resume pass and re-debate everything
- API key: via SML_OPENAI_API_KEY / OPENAI_API_KEY (SML_ variant wins)
- --log-level, --log-file: logging controls
- --create-log-file: also log to a datetime-stamped file under logs/

Outputs
-------
A JSONL file (default data/answers/answers_debate_<model>.jsonl), one
answer per line with the same keys as the 04-1 script plus a nested
debate field: {n_agents, n_rounds, transcript (per round, per agent),
vote_counts, tie_break}. The contestants see ONLY the question and its
lettered options — never the article. Appended incrementally
(crash-safe); re-running the same command skips questions already
debated.

Author: Matthew DeVerna
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

from toolkit import config
from toolkit.debate import debate_questions
from toolkit.utils import load_jsonl, resolve_path, setup_logging

DEFAULT_QUESTIONS = f"{config.QUESTIONS_DIR}/selected_questions.jsonl"

# LLM calls per question: K agents x (1 independent + R revision rounds).
CALLS_PER_QUESTION = config.DEBATE_N_AGENTS * (1 + config.DEBATE_N_ROUNDS)


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

    debating = parser.add_argument_group("debating")
    debating.add_argument(
        "--model",
        choices=sorted(config.DEBATE_MODELS),
        required=True,
        help=(
            "The contestant model backing all three debaters. OpenAI GPT "
            "models only — the debate runs on the openai-agents SDK. Run "
            "the script once per model; scripts/04-3_combine_answers.py "
            "merges the outputs."
        ),
    )
    debating.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help=(
            "Only debate the first N questions. Keep small for demos — "
            f"every question costs {CALLS_PER_QUESTION} LLM calls "
            f"({config.DEBATE_N_AGENTS} agents x "
            f"{1 + config.DEBATE_N_ROUNDS} rounds) (default: all)."
        ),
    )

    concurrency = parser.add_argument_group("concurrency")
    concurrency.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Include to overlap multiple questions' debates concurrently "
            "(asyncio). Each debate already runs its three agents "
            "concurrently within a round; this additionally overlaps whole "
            "questions."
        ),
    )
    concurrency.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help=(
            "Concurrent debates when --parallel is set (default: "
            "%(default)s). N debates burst up to N x "
            f"{config.DEBATE_N_AGENTS} simultaneous calls, so keep this "
            "modest to respect rate limits."
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
            "(default: data/answers/answers_debate_<model>.jsonl)."
        ),
    )
    output.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Skip the resume/dedup pass. By default the script reads the "
            "output file first and skips questions already debated, so "
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
            "logs/generate_debate_answers_2026-07-16_14-03-27.log."
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

    output_fp = args.output or f"{config.ANSWERS_DIR}/answers_debate_{args.model}.jsonl"

    log_file = args.log_file
    if args.create_log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"{config.LOGS_DIR}/generate_debate_answers_{timestamp}.log"

    setup_logging(
        log_level=args.log_level,
        log_file=log_file,
        console_output=True,
        append_mode=True,
    )
    logger = logging.getLogger("generate_debate_answers")
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
        "Debating %d questions with %s (up to %d LLM calls)",
        len(questions),
        args.model,
        len(questions) * CALLS_PER_QUESTION,
    )

    try:
        summary = debate_questions(
            questions,
            output_fp=output_fp,
            model=args.model,
            parallel=args.parallel,
            max_workers=args.max_workers,
            resume=not args.no_resume,
        )
    except ValueError as e:
        # Missing API key (message names both env vars) or bad model.
        logger.error(str(e))
        return 1
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted — completed debates are already saved to %s; "
            "re-run the same command to resume.",
            rel_to_root(output_fp),
        )
        return 130

    logger.info(
        "Done: %d debates by %s (%d skipped, %d failed) in %.1fs -> %s",
        summary["new_records"],
        summary["model"],
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
