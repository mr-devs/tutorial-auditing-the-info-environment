"""
Purpose: Collect Guardian articles into a JSONL file via the Guardian Content
API, with rate limiting, a daily call budget, retries, and resume.

Examples
--------
# 100 newest articles about climate policy
uv run python scripts/01_collect_guardian_news.py \
    --query "climate policy" --max-articles 100

# Several queries sharing one daily budget, restricted to a date window
uv run python scripts/01_collect_guardian_news.py \
    --query "misinformation" "artificial intelligence" \
    --from-date 2026-06-01 --to-date 2026-07-15 \
    --output data/articles/june_july.jsonl

# Everything in the politics section this week (no keyword query)
uv run python scripts/01_collect_guardian_news.py \
    --section politics --from-date 2026-07-08

Inputs
------
- --query: one or more search phrases (at least one of query/section/tag)
- --section, --tag: restrict to a Guardian section or editorial tag
- --from-date, --to-date: publication date window (YYYY-MM-DD)
- --order-by: newest (default), oldest, or relevance
- --show-fields: which article fields to request (default: bodyText headline
  byline trailText wordcount; if passed, only your list is requested)
- --page-size: articles per API call, 1-50
- --max-articles: cap per query (default 200)
- --daily-budget: max API calls for the run (default 500, the free-tier cap)
- --output: output JSONL path; --no-resume: skip the dedup/resume pass
- --api-key: overrides GUARDIAN_API_KEY ('test' works for light testing)
- --log-level, --log-file: logging controls
- --create-log-file: also log to a datetime-stamped file under logs/

Outputs
-------
A JSONL file (default data/articles/guardian_articles.jsonl), one article per
line with keys: id, url, published, section, plus one snake_case key per
requested show-field (defaults: headline, byline, trail_text, wordcount,
body_text). Appended incrementally (crash-safe); re-running the same command
skips articles already saved.

References:
----------
- https://open-platform.theguardian.com/documentation/

Author: Matthew DeVerna
"""

import argparse
import logging
import os
from datetime import date, datetime

from toolkit import config
from toolkit.guardian import (
    DEFAULT_DAILY_BUDGET,
    DEFAULT_FIELDS,
    GuardianAPIError,
    GuardianClient,
    collect,
)
from toolkit.utils import resolve_path, setup_logging

DEFAULT_OUTPUT = f"{config.ARTICLES_DIR}/guardian_articles.jsonl"


def rel_to_root(path) -> str:
    """Show a path relative to the repo root (keeps --help output readable)."""
    return os.path.relpath(path, config.REPO_ROOT)


def iso_date(value: str) -> str:
    """Validate a YYYY-MM-DD date string for argparse."""
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid date (expected YYYY-MM-DD)"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    search = parser.add_argument_group("search (at least one of query/section/tag)")
    search.add_argument(
        "--query",
        nargs="+",
        metavar="PHRASE",
        help=(
            "One or more search phrases; each runs as a separate search "
            "sharing the same daily budget (e.g. --query 'climate policy' "
            "misinformation)."
        ),
    )
    search.add_argument(
        "--section",
        help="Restrict to a Guardian section id, e.g. politics, technology, world.",
    )
    search.add_argument(
        "--tag",
        help=(
            "Restrict to a Guardian tag, e.g. environment/climate-crisis. "
            "Browse tags at https://content.guardianapis.com/tags?api-key=test"
        ),
    )
    search.add_argument(
        "--from-date",
        type=iso_date,
        metavar="YYYY-MM-DD",
        help="Only articles published on/after this date.",
    )
    search.add_argument(
        "--to-date",
        type=iso_date,
        metavar="YYYY-MM-DD",
        help="Only articles published on/before this date.",
    )
    search.add_argument(
        "--order-by",
        choices=["newest", "oldest", "relevance"],
        default="newest",
        help="Result ordering (default: %(default)s).",
    )
    search.add_argument(
        "--show-fields",
        nargs="+",
        metavar="FIELD",
        default=None,
        help=(
            "Article fields to request beyond the basics (id, url, publication "
            "date, section). Space-separated, camelCase, as named by the API. "
            "If omitted, the default set is requested: "
            f"{' '.join(DEFAULT_FIELDS.split(','))}. If passed, ONLY the "
            "fields you list are requested (e.g. --show-fields headline "
            "standfirst thumbnail). Full field list: "
            "https://open-platform.theguardian.com/documentation/search"
        ),
    )

    volume = parser.add_argument_group("volume & pacing")
    volume.add_argument(
        "--page-size",
        type=int,
        choices=range(1, 51),
        metavar="1-50",
        default=10,
        help=(
            "Articles per API call, max 50 (default: %(default)s, the API's "
            "own default). Use 50 to stretch the daily call budget furthest."
        ),
    )
    volume.add_argument(
        "--max-articles",
        type=int,
        default=200,
        help=(
            "Maximum articles to fetch PER QUERY (default: %(default)s). "
            "Keep small for demos; raise for research runs."
        ),
    )
    volume.add_argument(
        "--daily-budget",
        type=int,
        default=DEFAULT_DAILY_BUDGET,
        help=(
            "Maximum API calls for this run — the Guardian free tier allows "
            "500/day. The script stops cleanly when the budget is spent "
            "(default: %(default)s)."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=(
            "Path for the output .jsonl file, appended to incrementally; "
            f"relative paths resolve from the repo root (default: {rel_to_root(DEFAULT_OUTPUT)})."
        ),
    )
    output.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Skip the resume/dedup pass and append blindly. By default the "
            "script reads the output file first and skips articles whose ids "
            "are already saved, so re-running the same command is safe."
        ),
    )

    ops = parser.add_argument_group("operational")
    ops.add_argument(
        "--api-key",
        default=None,
        help=(
            "Guardian API key (overrides the GUARDIAN_API_KEY environment "
            "variable). The shared public key 'test' works for light testing."
        ),
    )
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
            "logs/collect_guardian_news_2026-07-15_14-03-27.log."
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.output = resolve_path(args.output)
    if args.log_file:
        args.log_file = resolve_path(args.log_file)

    if not (args.query or args.section or args.tag):
        parser.error("provide at least one of --query, --section, or --tag")

    log_file = args.log_file
    if args.create_log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"{config.LOGS_DIR}/collect_guardian_news_{timestamp}.log"

    setup_logging(
        log_level=args.log_level,
        log_file=log_file,
        console_output=True,
        append_mode=True,
    )
    logger = logging.getLogger("collect_guardian_news")
    if log_file:
        logger.info("Logging to %s", rel_to_root(log_file))

    try:
        client = GuardianClient(
            api_key=args.api_key or "",
            daily_budget=args.daily_budget,
        )
    except GuardianAPIError as e:
        logger.error(str(e))
        return 1

    queries = args.query or [None]  # filter-only search when no query given
    total_new = 0
    try:
        for query in queries:
            label = query or f"(section={args.section}, tag={args.tag})"
            logger.info("Collecting: %s", label)
            try:
                summary = collect(
                    query,
                    output_fp=args.output,
                    client=client,
                    resume=not args.no_resume,
                    max_articles=args.max_articles,
                    section=args.section,
                    tag=args.tag,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    order_by=args.order_by,
                    page_size=args.page_size,
                    show_fields=(
                        ",".join(args.show_fields)
                        if args.show_fields
                        else DEFAULT_FIELDS
                    ),
                )
            except GuardianAPIError as e:
                logger.error(str(e))
                return 1

            total_new += summary["new"]
            logger.info(
                "'%s': %d new, %d skipped (already saved), %d calls used, "
                "%s total matches available",
                label,
                summary["new"],
                summary["skipped"],
                summary["calls_used"],
                summary["total_available"],
            )
            if summary["budget_exhausted"]:
                logger.warning(
                    "Daily budget exhausted — partial results saved. "
                    "Re-run later to resume where you left off."
                )
                break
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted — %d new records already saved to %s; "
            "re-run the same command to resume.",
            total_new,
            rel_to_root(args.output),
        )
        return 130

    logger.info(
        "Done: %d new articles saved to %s (%d/%d calls used).",
        total_new,
        rel_to_root(args.output),
        client.calls_made,
        client.daily_budget,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
