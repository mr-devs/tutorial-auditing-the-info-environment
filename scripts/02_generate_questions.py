"""
Purpose: Generate multiple-choice questions from Step 1's Guardian articles
using an LLM (OpenAI or Gemini) with structured outputs, optionally in
parallel via a thread pool.

Examples
--------
# 3 questions per article for every Step-1 article, with an OpenAI model
uv run python scripts/02_generate_questions.py --model gpt-5.6-terra

# A Gemini question set (run several models to enable Step 3+ comparisons)
uv run python scripts/02_generate_questions.py --model gemini-3.1-flash-lite

# Fast demo: 5 articles, 8 threads
uv run python scripts/02_generate_questions.py \
    --model gpt-5.4-mini-2026-03-17 --max-articles 5 --parallel --max-workers 8

Inputs
------
- --input: articles JSONL from Step 1 (default data/articles/guardian_articles.jsonl)
- --model: which model generates the questions (required). One of:
  gpt-5.4-mini-2026-03-17, gpt-5.6-terra (OpenAI);
  gemini-3.1-flash-lite, gemini-3.5-flash (Gemini).
  The provider is inferred from the model.
- --n-questions: questions per article (default 3)
- --max-articles: cap for demos (default: all)
- --parallel: submit articles to a ThreadPoolExecutor instead of a sequential loop
- --max-workers: threads when --parallel (default 8)
- --output: output JSONL path (default data/questions/questions_<model>.jsonl)
- --no-resume: skip the dedup/resume pass and regenerate everything
- --api keys: via SML_OPENAI_API_KEY / OPENAI_API_KEY or SML_GEMINI_API_KEY /
  GEMINI_API_KEY (the SML_ variant, used on the lab machines, wins)
- --log-level, --log-file: logging controls
- --create-log-file: also log to a datetime-stamped file under logs/

Outputs
-------
A JSONL file (default data/questions/questions_<model>.jsonl), one QUESTION
per line with keys: id, article_id, question_index, provider, model, question,
options (4), correct_letter (A-D), explanation, generated_at. Appended
incrementally as each article completes (crash-safe); re-running the same
command skips articles that already have questions saved.

Author: Matthew DeVerna
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from toolkit import config
from toolkit.questions import generate_for_articles
from toolkit.utils import load_jsonl, setup_logging

DEFAULT_INPUT = f"{config.ARTICLES_DIR}/guardian_articles.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inputs = parser.add_argument_group("input")
    inputs.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Articles JSONL produced by Step 1 (default: %(default)s).",
    )

    generation = parser.add_argument_group("generation")
    generation.add_argument(
        "--model",
        choices=sorted(config.SUPPORTED_MODELS),
        required=True,
        help=(
            "Which model generates the questions; the provider (OpenAI or "
            "Gemini) is inferred from the model id. Run the script once per "
            "model to produce comparable question sets."
        ),
    )
    generation.add_argument(
        "--n-questions",
        type=int,
        default=3,
        help="Questions to generate per article (default: %(default)s).",
    )
    generation.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help=(
            "Only process the first N articles. Keep small for demos — every "
            "article costs one LLM call (default: all)."
        ),
    )

    concurrency = parser.add_argument_group("concurrency")
    concurrency.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Submit articles to a ThreadPoolExecutor instead of a sequential "
            "loop. API calls are I/O-bound, so threads overlap the waiting."
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
        help=(
            "Output JSONL file, appended to incrementally "
            "(default: data/questions/questions_<model>.jsonl)."
        ),
    )
    output.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Skip the resume/dedup pass. By default the script reads the "
            "output file first and skips articles that already have "
            "questions, so re-running the same command is safe."
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
        help="Also write logs to this exact file path (appended).",
    )
    log_dest.add_argument(
        "--create-log-file",
        action="store_true",
        help=(
            "Also write logs to an auto-named, datetime-stamped file, e.g. "
            "logs/generate_questions_2026-07-15_14-03-27.log."
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    provider = config.SUPPORTED_MODELS[args.model]
    output_fp = args.output or f"{config.QUESTIONS_DIR}/questions_{args.model}.jsonl"

    log_file = args.log_file
    if args.create_log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"logs/generate_questions_{timestamp}.log"

    setup_logging(
        log_level=args.log_level,
        log_file=log_file,
        console_output=True,
        append_mode=True,
    )
    logger = logging.getLogger("generate_questions")
    if log_file:
        logger.info("Logging to %s", log_file)

    if not Path(args.input).exists():
        logger.error(
            "Input file %s not found — run scripts/01_collect_guardian_news.py first.",
            args.input,
        )
        return 1

    articles = load_jsonl(args.input)
    if args.max_articles is not None:
        articles = articles[: args.max_articles]
    logger.info(
        "Generating up to %d questions each for %d articles via %s (%s)",
        args.n_questions,
        len(articles),
        args.model,
        provider,
    )

    try:
        summary = generate_for_articles(
            articles,
            output_fp=output_fp,
            provider=provider,
            model=args.model,
            n_questions=args.n_questions,
            parallel=args.parallel,
            max_workers=args.max_workers,
            resume=not args.no_resume,
        )
    except ValueError as e:
        # Missing API key (message names both env vars) or bad provider.
        logger.error(str(e))
        return 1
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted — completed articles are already saved to %s; "
            "re-run the same command to resume.",
            output_fp,
        )
        return 130

    logger.info(
        "Done: %d new questions from %d articles (%d skipped, %d failed) "
        "in %.1fs -> %s",
        summary["new_questions"],
        summary["articles_processed"],
        summary["articles_skipped"],
        summary["articles_failed"],
        summary["elapsed_seconds"],
        summary["output_fp"],
    )
    if summary["articles_failed"]:
        logger.warning(
            "%d articles failed — re-run the same command to retry just those.",
            summary["articles_failed"],
        )
        if summary["articles_processed"] == 0 and summary["articles_skipped"] == 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
