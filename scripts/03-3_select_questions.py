"""
Purpose: Select a seeded-random set of questions that passed the LLM judges,
writing the full question records to JSONL for the downstream steps.

A question "passes" one judge model when that model marked it
answerable = True and faithful = True. A question is
eligible for selection when at least --min-passing (default 2) judge models
passed it.

Examples
--------
# The standard draw: 100 questions passing >= 2 judges, seed 42
uv run python scripts/03-3_select_questions.py \
    --input data/judgments/judgments_combined.csv \
    --questions data/questions/questions_gpt-5.4-mini-2026-03-17.jsonl

# Question sets from several generators, custom size and seed
uv run python scripts/03-3_select_questions.py \
    --input data/judgments/judgments_combined.csv \
    --questions data/questions/questions_gpt-5.4-mini-2026-03-17.jsonl \
                data/questions/questions_gemini-3.1-flash-lite.jsonl \
    --n 50 --seed 7

Inputs
------
- --input: combined judgments CSV from scripts/03-2_combine_judgments.py (required)
- --questions: one or more questions JSONL files from Step 2, used to
  recover the full question records for the selected ids (required)
- --n: how many questions to select (default 100). If fewer pass, all
  passers are written with a warning.
- --seed: random seed for the selection draw (default 42) — same seed and
  inputs give the same selection
- --min-passing: minimum judge models that must pass a question (default 2)
- --output: output JSONL path (default data/questions/selected_questions.jsonl)
- --log-level, --log-file: logging controls
- --create-log-file: also log to a datetime-stamped file under logs/

Outputs
-------
A JSONL file of the selected questions — the full Step 2 record (id,
article_id, question, options, correct_letter, explanation, provider, model,
generated_at, ...) plus n_models_passing. One question per line.

Author: Matthew DeVerna
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import pandas as pd

from toolkit import config
from toolkit.utils import load_jsonl, resolve_path, setup_logging

DEFAULT_OUTPUT = f"{config.QUESTIONS_DIR}/selected_questions.jsonl"


def rel_to_root(path) -> str:
    """Show a path relative to the repo root (keeps --help output readable)."""
    return os.path.relpath(path, config.REPO_ROOT)


def find_passing_question_ids(df: pd.DataFrame, min_passing: int) -> dict:
    """Return {question_id: n_models_passing} for questions passing the rule.

    A row passes when answerable & faithful; a question passes
    overall when at least ``min_passing`` distinct models passed it.
    """
    df = df.copy()
    df["passing"] = df["answerable"] & df["faithful"]
    counts = df[df["passing"]].groupby("question_id")["judge_model"].nunique().to_dict()
    return {qid: n for qid, n in counts.items() if n >= min_passing}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inputs = parser.add_argument_group("input")
    inputs.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help=(
            "Path to the combined judgments .csv file produced by "
            "scripts/03-2_combine_judgments.py, e.g. "
            "data/judgments/judgments_combined.csv; relative paths resolve from the repo root."
        ),
    )
    inputs.add_argument(
        "--questions",
        required=True,
        nargs="+",
        metavar="PATH",
        help=(
            "Path(s) to one or more questions .jsonl files from Step 2 "
            "(space-separated), e.g. data/questions/questions_gpt-5.6-terra.jsonl "
            "— the source of the full question records written to the output; "
            "relative paths resolve from the repo root."
        ),
    )

    selection = parser.add_argument_group("selection")
    selection.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of questions to select (default: %(default)s).",
    )
    selection.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for the draw (default: %(default)s). The same seed "
            "and inputs always produce the same selection."
        ),
    )
    selection.add_argument(
        "--min-passing",
        type=int,
        default=2,
        help=(
            "Minimum number of judge models that must mark a question as "
            "passing — answerable=True and faithful=True "
            "(default: %(default)s)."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=(
            "Path for the output .jsonl file of selected questions; "
            f"relative paths resolve from the repo root (default: {rel_to_root(DEFAULT_OUTPUT)})."
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
        help="Path to a file to also write logs to (appended); relative paths resolve from the repo root.",
    )
    log_dest.add_argument(
        "--create-log-file",
        action="store_true",
        help=(
            "Also write logs to an auto-named, datetime-stamped file, e.g. "
            "logs/select_questions_2026-07-16_14-03-27.log."
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.input = resolve_path(args.input)
    args.questions = [resolve_path(q) for q in args.questions]
    args.output = resolve_path(args.output)
    if args.log_file:
        args.log_file = resolve_path(args.log_file)

    log_file = args.log_file
    if args.create_log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"{config.LOGS_DIR}/select_questions_{timestamp}.log"

    setup_logging(
        log_level=args.log_level,
        log_file=log_file,
        console_output=True,
        append_mode=True,
    )
    logger = logging.getLogger("select_questions")
    if log_file:
        logger.info("Logging to %s", rel_to_root(log_file))

    if not Path(args.input).exists():
        logger.error(
            "Judgments CSV %s not found — run scripts/03-2_combine_judgments.py first.",
            rel_to_root(args.input),
        )
        return 1
    for fp in args.questions:
        if not Path(fp).exists():
            logger.error("Questions file %s not found.", rel_to_root(fp))
            return 1

    df = pd.read_csv(args.input)
    passing = find_passing_question_ids(df, args.min_passing)
    total_questions = df["question_id"].nunique()
    logger.info(
        "%d of %d judged questions pass (>= %d of %d models: answerable "
        "and faithful)",
        len(passing),
        total_questions,
        args.min_passing,
        df["judge_model"].nunique(),
    )

    if not passing:
        logger.error("No questions pass the selection rule — nothing to write.")
        return 1

    if len(passing) < args.n:
        logger.warning(
            "Only %d questions pass — fewer than the requested %d; "
            "writing all of them.",
            len(passing),
            args.n,
        )
        selected_ids = sorted(passing)
    else:
        selected_ids = random.Random(args.seed).sample(sorted(passing), args.n)

    questions_by_id = {}
    for fp in args.questions:
        for q in load_jsonl(fp):
            questions_by_id[q["id"]] = q

    missing = [qid for qid in selected_ids if qid not in questions_by_id]
    if missing:
        logger.error(
            "%d selected question ids not found in the provided --questions "
            "files (first: %s). Pass the same files the judges evaluated.",
            len(missing),
            missing[0],
        )
        return 1

    output_fp = Path(args.output)
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(output_fp, "w", encoding="utf-8") as f:
        for qid in selected_ids:
            record = {**questions_by_id[qid], "n_models_passing": passing[qid]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Done: judged %d -> passing %d -> selected %d (seed %d) -> %s",
        total_questions,
        len(passing),
        len(selected_ids),
        args.seed,
        rel_to_root(output_fp),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
