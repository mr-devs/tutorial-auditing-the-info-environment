"""
Purpose: Combine per-model judgment JSONL files (from
scripts/03-1_generate_judgments.py) into one clean, tidy CSV with one row per
judgment.

Examples
--------
# Combine every judgment file in the default judgments directory
uv run python scripts/03-2_combine_judgments.py \
    --input-dir data/judgments --glob 'judgments_*.jsonl'

# Combine a specific subset, custom output
uv run python scripts/03-2_combine_judgments.py \
    --input-dir data/judgments --glob 'judgments_gpt-*.jsonl' \
    --output data/judgments/openai_judges.csv

Inputs
------
- --input-dir: directory containing judgment JSONL files (required, no default)
- --glob: filename pattern to match within --input-dir, e.g.
  'judgments_*.jsonl' — quote it so your shell does not expand it
  (required, no default)
- --output: output CSV path (default data/judgments/judgments_combined.csv)
- --log-level, --log-file: logging controls
- --create-log-file: also log to a datetime-stamped file under logs/

Outputs
-------
A tidy CSV where each row is ONE judgment (one question by one judge model),
with columns: question_id, article_id, generator_provider, generator_model
(whose question was judged), judge_model (who judged), faithful,
rationale, judged_at. Rows are sorted by question_id then judge_model.
Every input line is validated against the judgment record schema; invalid or
malformed lines are skipped with a warning.

Author: Matthew DeVerna
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ValidationError

from toolkit import config
from toolkit.utils import resolve_path, setup_logging

DEFAULT_OUTPUT = f"{config.JUDGMENTS_DIR}/judgments_combined.csv"

CSV_COLUMNS = [
    "question_id",
    "article_id",
    "generator_provider",
    "generator_model",
    "judge_model",
    "faithful",
    "rationale",
    "judged_at",
]


class JudgmentRecord(BaseModel):
    """On-disk judgment record schema (validates each JSONL line)."""

    id: str
    question_id: str
    article_id: str
    generator_provider: Optional[str]
    generator_model: Optional[str]
    judge_model: str
    faithful: bool
    rationale: str
    judged_at: str


def rel_to_root(path) -> str:
    """Show a path relative to the repo root (keeps --help output readable)."""
    return os.path.relpath(path, config.REPO_ROOT)


def load_valid_records(filepath, logger) -> list:
    """Read one JSONL file, validating each line; skip bad lines with a warning."""
    records = []
    skipped = 0
    with open(filepath, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(JudgmentRecord(**json.loads(line)))
            except (json.JSONDecodeError, ValidationError, TypeError):
                skipped += 1
                logger.warning("Skipping invalid line %d in %s", lineno, filepath)
    if skipped:
        logger.warning("%s: skipped %d invalid lines", filepath, skipped)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inputs = parser.add_argument_group("input")
    inputs.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help=(
            "Path to the directory containing the per-model judgment .jsonl "
            "files, e.g. data/judgments; relative paths resolve from the repo root."
        ),
    )
    inputs.add_argument(
        "--glob",
        required=True,
        metavar="PATTERN",
        help=(
            "Filename pattern matched inside --input-dir (not a path), e.g. "
            "'judgments_*.jsonl'. Quote it so your shell does not expand it."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=(
            "Path for the output .csv file; "
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
            "logs/combine_judgments_2026-07-16_14-03-27.log."
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.input_dir = resolve_path(args.input_dir)
    args.output = resolve_path(args.output)
    if args.log_file:
        args.log_file = resolve_path(args.log_file)

    log_file = args.log_file
    if args.create_log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"{config.LOGS_DIR}/combine_judgments_{timestamp}.log"

    setup_logging(
        log_level=args.log_level,
        log_file=log_file,
        console_output=True,
        append_mode=True,
    )
    logger = logging.getLogger("combine_judgments")
    if log_file:
        logger.info("Logging to %s", rel_to_root(log_file))

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error("Input directory %s does not exist.", rel_to_root(input_dir))
        return 1

    files = sorted(input_dir.glob(args.glob))
    if not files:
        logger.error(
            "No files in %s match %r — run scripts/03-1_generate_judgments.py first.",
            rel_to_root(input_dir),
            args.glob,
        )
        return 1

    all_records = []
    for fp in files:
        records = load_valid_records(fp, logger)
        logger.info("%s: %d judgments", rel_to_root(fp), len(records))
        all_records.extend(records)

    if not all_records:
        logger.error("No valid judgment records found in %d files.", len(files))
        return 1

    df = pd.DataFrame([r.model_dump() for r in all_records])
    df = (
        df[CSV_COLUMNS]
        .sort_values(["question_id", "judge_model"])
        .reset_index(drop=True)
    )

    output_fp = Path(args.output)
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_fp, index=False)

    logger.info(
        "Done: %d judgments (%d questions x %d models) from %d files -> %s",
        len(df),
        df["question_id"].nunique(),
        df["judge_model"].nunique(),
        len(files),
        rel_to_root(output_fp),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
