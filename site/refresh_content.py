"""
Purpose: Refresh the horse-race site's quiz content, end to end.

Runs the tutorial pipeline (via the site's ``sitekit`` copy) and loads the
result into the site's SQLite database as a new, versioned question set:

1. Collect up to --max-articles Guardian articles about --topic from the
   past week (full body text).
2. Generate multiple-choice questions for each article.
3. Judge every question with the three judge models; a question passes
   when at least --min-passing (default 2) judges mark it faithful.
4. Randomly select --n-select passing questions (seeded draw).
5. Insert the selection as a new row in ``question_sets`` plus its
   ``questions``, and (with --activate, the default) make it the live set.

Old question sets, sessions, and answers are never modified: history stays
tied to its own set id, and in-flight quiz sessions on the previous set can
still finish.

Cost note: a full refresh with defaults makes ~2-3 Guardian calls,
~100 question-generation calls, and ~900 judge calls (3 questions/article
x 100 articles x 3 judges). Use --max-articles 10 for a cheap dev seed.

Examples
--------
# The standard weekly refresh (run from site/)
uv run python refresh_content.py

# A cheap development seed
uv run python refresh_content.py --max-articles 10 --n-select 5

# Stage a set without making it live
uv run python refresh_content.py --no-activate

Author: Matthew DeVerna
"""

import argparse
import json
import logging
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from app import db
from app import config as site_config
from sitekit import config, guardian, judgments, questions
from sitekit.utils import load_jsonl, setup_logging

logger = logging.getLogger("refresh_content")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    content = parser.add_argument_group("content")
    content.add_argument(
        "--topic",
        default=site_config.TOPIC_DEFAULT,
        help="Guardian search phrase (default: %(default)r).",
    )
    content.add_argument(
        "--max-articles",
        type=int,
        default=100,
        help="Maximum articles to collect (default: %(default)s).",
    )
    content.add_argument(
        "--days",
        type=int,
        default=7,
        help="Collection window: articles from the past N days (default: %(default)s).",
    )
    content.add_argument(
        "--n-questions",
        type=int,
        default=3,
        help="Questions generated per article (default: %(default)s).",
    )
    content.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
        help="Question-generator provider (default: %(default)s).",
    )

    selection = parser.add_argument_group("selection")
    selection.add_argument(
        "--n-select",
        type=int,
        default=site_config.N_QUIZ_QUESTIONS,
        help="Number of questions in the live quiz (default: %(default)s).",
    )
    selection.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the selection draw (default: %(default)s).",
    )
    selection.add_argument(
        "--min-passing",
        type=int,
        default=2,
        help=(
            "Minimum judge models that must mark a question faithful "
            "(default: %(default)s)."
        ),
    )

    ops = parser.add_argument_group("operational")
    ops.add_argument(
        "--no-activate",
        dest="activate",
        action="store_false",
        help="Load the new question set without making it the live one.",
    )
    ops.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Thread pool size for LLM calls (default: %(default)s).",
    )
    ops.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console logging verbosity (default: %(default)s).",
    )
    return parser


def run_pipeline(args, scratch: Path) -> tuple[list, dict, dict]:
    """Collect, generate, and judge; return the pipeline's artifacts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    scratch : Path
        Directory the intermediate JSONL files are written to.

    Returns
    -------
    tuple of (list, dict, dict)
        ``(question_records, articles_by_id, passing)`` where ``passing``
        maps each passing question id to its number of passing judges.
    """
    from_date = (date.today() - timedelta(days=args.days)).isoformat()
    to_date = date.today().isoformat()

    articles_fp = scratch / "articles.jsonl"
    logger.info(
        "Step 1/3: collecting up to %d Guardian articles on %r (%s to %s)",
        args.max_articles,
        args.topic,
        from_date,
        to_date,
    )
    summary = guardian.collect(
        args.topic,
        output_fp=articles_fp,
        max_articles=args.max_articles,
        from_date=from_date,
        to_date=to_date,
        order_by="newest",
    )
    logger.info(
        "Collected %d articles (%d API calls)", summary["new"], summary["calls_used"]
    )
    articles = load_jsonl(articles_fp)
    if not articles:
        raise SystemExit(
            "No articles collected — try a broader --topic or longer --days."
        )

    questions_fp = scratch / "questions.jsonl"
    logger.info("Step 2/3: generating %d questions per article", args.n_questions)
    questions.generate_for_articles(
        articles,
        output_fp=questions_fp,
        provider=args.provider,
        n_questions=args.n_questions,
        parallel=True,
        max_workers=args.max_workers,
    )
    question_records = load_jsonl(questions_fp)
    logger.info("Generated %d questions", len(question_records))
    if not question_records:
        raise SystemExit("No questions generated — check the generator model/key.")

    articles_by_id = {a["id"]: a for a in articles}
    all_judgments = []
    for i, judge_model in enumerate(config.JUDGE_MODELS, start=1):
        judgments_fp = scratch / f"judgments_{judge_model}.jsonl"
        logger.info(
            "Step 3/3: judging with %s (%d/%d)",
            judge_model,
            i,
            len(config.JUDGE_MODELS),
        )
        judgments.judge_questions(
            question_records,
            articles_by_id,
            output_fp=judgments_fp,
            model=judge_model,
            parallel=True,
            max_workers=args.max_workers,
        )
        all_judgments.extend(load_jsonl(judgments_fp))

    passing = judgments.find_passing_question_ids(all_judgments, args.min_passing)
    logger.info(
        "%d of %d questions pass (>= %d of %d judges faithful)",
        len(passing),
        len(question_records),
        args.min_passing,
        len(config.JUDGE_MODELS),
    )
    return question_records, articles_by_id, passing


def load_question_set(args, selected, questions_by_id, articles_by_id, passing) -> int:
    """Insert the new question set and its questions in one transaction.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    selected : list of str
        Selected question ids, in quiz order.
    questions_by_id : dict
        Full Step 2 question records keyed by id.
    articles_by_id : dict
        Article records keyed by id (for headline/url/published).
    passing : dict
        Mapping of question id to number of judges passing it.

    Returns
    -------
    int
        The new question set's id.
    """
    from_date = (date.today() - timedelta(days=args.days)).isoformat()
    with db.connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO question_sets
                (topic, created_at, from_date, to_date,
                 n_articles, n_generated, n_passing, seed, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                args.topic,
                datetime.now(timezone.utc).isoformat(),
                from_date,
                date.today().isoformat(),
                len(articles_by_id),
                len(questions_by_id),
                len(passing),
                args.seed,
            ),
        )
        set_id = cur.lastrowid
        for position, qid in enumerate(selected, start=1):
            q = questions_by_id[qid]
            article = articles_by_id.get(q["article_id"], {})
            conn.execute(
                """
                INSERT INTO questions
                    (set_id, position, source_question_id, article_id,
                     article_url, headline, published, question, options_json,
                     correct_letter, explanation, generator_model,
                     n_models_passing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    set_id,
                    position,
                    q["id"],
                    q["article_id"],
                    article.get("url"),
                    article.get("headline"),
                    article.get("published"),
                    q["question"],
                    json.dumps(q["options"], ensure_ascii=False),
                    q["correct_letter"],
                    q.get("explanation"),
                    q.get("model"),
                    passing[qid],
                ),
            )
        if args.activate:
            conn.execute("UPDATE question_sets SET is_active = 0 WHERE is_active = 1")
            conn.execute(
                "UPDATE question_sets SET is_active = 1 WHERE id = ?", (set_id,)
            )
    return set_id


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(log_level=args.log_level, console_output=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    scratch = Path(config.REFRESH_DIR) / timestamp
    scratch.mkdir(parents=True, exist_ok=True)
    logger.info("Scratch dir: %s", scratch)

    question_records, articles_by_id, passing = run_pipeline(args, scratch)

    if len(passing) < args.n_select:
        logger.error(
            "Only %d questions pass — fewer than the requested %d. "
            "Re-run with more --max-articles or a longer --days window.",
            len(passing),
            args.n_select,
        )
        return 1

    selected = random.Random(args.seed).sample(sorted(passing), args.n_select)
    # Shuffle quiz order too (deterministically) so position is not
    # correlated with question id ordering.
    random.Random(args.seed + 1).shuffle(selected)

    questions_by_id = {q["id"]: q for q in question_records}
    db.init_db()
    set_id = load_question_set(args, selected, questions_by_id, articles_by_id, passing)

    logger.info(
        "Done: question set %d loaded (%s) — %d articles -> %d questions -> "
        "%d passing -> %d selected. %s",
        set_id,
        args.topic,
        len(articles_by_id),
        len(questions_by_id),
        len(passing),
        len(selected),
        "ACTIVE (live now)"
        if args.activate
        else "staged (use --activate later or flip is_active manually)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
