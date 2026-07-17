"""
Purpose: Have one LLM judge model evaluate generated multiple-choice
questions against their source articles on three binary dimensions
(answerable, faithful, guessable), writing one JSONL judgment per question.

Run this script once per judge model, then merge the per-model files with
scripts/03-2_combine_judgments.py.

Examples
--------
# Judge one question set with one model
uv run python scripts/03-1_generate_judgments.py \
    --questions data/questions/questions_gpt-5.6-terra.jsonl \
    --articles data/articles/guardian_articles.jsonl \
    --model gpt-5.6-luna

# All three judges (run one after another), threaded
for M in gpt-5.6-luna gpt-5.5-2026-04-23 gpt-5.4-mini-2026-03-17; do
  uv run python scripts/03-1_generate_judgments.py \
      --questions data/questions/questions_gpt-5.6-terra.jsonl \
      --articles data/articles/guardian_articles.jsonl \
      --model $M --parallel
done

# Small demo run
uv run python scripts/03-1_generate_judgments.py \
    --questions data/questions/questions_gpt-5.6-terra.jsonl \
    --articles data/articles/guardian_articles.jsonl \
    --model gpt-5.4-mini-2026-03-17 --max-questions 5 --parallel

Inputs
------
- --questions: questions JSONL from Step 2 (required, no default)
- --articles: articles JSONL from Step 1 the questions were generated from
  (required, no default; supplies each article's headline and full text)
- --model: the judge model (required). One of: gpt-5.6-luna,
  gpt-5.5-2026-04-23, gpt-5.4-mini-2026-03-17
- --max-questions: cap for demos (default: all)
- --parallel: submit multiple questions in parallel via a ThreadPoolExecutor
- --max-workers: threads when --parallel (default 8)
- --output: output JSONL path (default data/judgments/judgments_<model>.jsonl)
- --no-resume: skip the dedup/resume pass and re-judge everything
- API key: via SML_OPENAI_API_KEY / OPENAI_API_KEY (SML_ variant wins)
- --log-level, --log-file: logging controls
- --create-log-file: also log to a datetime-stamped file under logs/

Outputs
-------
A JSONL file (default data/judgments/judgments_<model>.jsonl), one judgment
per line with keys: id, question_id, article_id, model, answerable (bool),
faithful (bool), guessable (bool), rationale, judged_at. The judge sees the
article headline + full text, the question, its lettered options, and the
marked correct answer (never the generator's explanation). Appended
incrementally (crash-safe); re-running the same command skips questions
already judged.

Author: Matthew DeVerna
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

from toolkit import config
from toolkit.judgments import judge_questions
from toolkit.utils import load_jsonl, setup_logging


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
        required=True,
        metavar="PATH",
        help=(
            "Path to the questions .jsonl file produced by "
            "scripts/02_generate_questions.py, e.g. "
            "data/questions/questions_gpt-5.6-terra.jsonl."
        ),
    )
    inputs.add_argument(
        "--articles",
        required=True,
        metavar="PATH",
        help=(
            "Path to the articles .jsonl file produced by "
            "scripts/01_collect_guardian_news.py, e.g. "
            "data/articles/guardian_articles.jsonl — must be the file the "
            "questions were generated from, so every question's article "
            "(headline + full text) can be shown to the judge."
        ),
    )

    judging = parser.add_argument_group("judging")
    judging.add_argument(
        "--model",
        choices=sorted(config.JUDGE_MODELS),
        required=True,
        help=(
            "The judge model. Run the script once per model; "
            "scripts/03-2_combine_judgments.py merges the per-model outputs."
        ),
    )
    judging.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help=(
            "Only judge the first N questions. Keep small for demos — every "
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
            "Path for the output .jsonl file, appended to incrementally "
            "(default: data/judgments/judgments_<model>.jsonl)."
        ),
    )
    output.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Skip the resume/dedup pass. By default the script reads the "
            "output file first and skips questions already judged, so "
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
        help="Path to a file to also write logs to (appended).",
    )
    log_dest.add_argument(
        "--create-log-file",
        action="store_true",
        help=(
            "Also write logs to an auto-named, datetime-stamped file, e.g. "
            "logs/generate_judgments_2026-07-16_14-03-27.log."
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_fp = args.output or f"{config.JUDGMENTS_DIR}/judgments_{args.model}.jsonl"

    log_file = args.log_file
    if args.create_log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"{config.LOGS_DIR}/generate_judgments_{timestamp}.log"

    setup_logging(
        log_level=args.log_level,
        log_file=log_file,
        console_output=True,
        append_mode=True,
    )
    logger = logging.getLogger("generate_judgments")
    if log_file:
        logger.info("Logging to %s", rel_to_root(log_file))

    for label, path, producer in [
        ("Questions", args.questions, "scripts/02_generate_questions.py"),
        ("Articles", args.articles, "scripts/01_collect_guardian_news.py"),
    ]:
        if not Path(path).exists():
            logger.error(
                "%s file %s not found — run %s first.",
                label,
                rel_to_root(path),
                producer,
            )
            return 1

    questions = load_jsonl(args.questions)
    articles_by_id = {a["id"]: a for a in load_jsonl(args.articles)}
    if args.max_questions is not None:
        questions = questions[: args.max_questions]

    missing = sum(1 for q in questions if q["article_id"] not in articles_by_id)
    if missing:
        logger.warning(
            "%d of %d questions reference articles missing from %s — "
            "those will be counted as failed.",
            missing,
            len(questions),
            rel_to_root(args.articles),
        )
    logger.info("Judging %d questions with %s", len(questions), args.model)

    try:
        summary = judge_questions(
            questions,
            articles_by_id,
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
            "Interrupted — completed judgments are already saved to %s; "
            "re-run the same command to resume.",
            rel_to_root(output_fp),
        )
        return 130

    logger.info(
        "Done: %d judgments by %s (%d skipped, %d failed) in %.1fs -> %s",
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
        if summary["judged"] == 0 and summary["skipped"] == 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
