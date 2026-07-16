# Step 1 — Scrape news from the Guardian

**Status: built.**

## Goal

Collect fresh Guardian articles (with full body text) into JSONL for Step 2's
question generation. Teach the room how a polite, production-grade API
collector is built: one raw request → pagination → rate limits → budgets →
incremental saving → resume → retries.

## Artifacts

| Artifact | Path |
|---|---|
| Teaching notebook | `notebooks/01_guardian_news_collection.ipynb` |
| CLI script | `scripts/01_collect_guardian_news.py` |
| Toolkit module | `toolkit/toolkit/guardian.py` |
| Reference prototype (untouched) | `scripts/collect_guardian_news.py` |

## Toolkit design (`toolkit.guardian`)

- `RateLimiter` — monotonic-clock pacer; ≥1.05 s between calls (free tier is
  1 req/sec).
- `GuardianClient` — holds api key (falls back to `config.GUARDIAN_API_KEY`),
  a `requests.Session`, the rate limiter, and a daily call budget (default
  500, the free-tier cap). Its single HTTP method `_get()` is decorated with
  tenacity: exponential backoff with jitter (initial 2 s, max 60 s, 5
  attempts) retrying only transient failures (connection/timeout errors and
  HTTP 429/500/502/503/504). Permanent 4xx errors raise `GuardianAPIError`
  immediately with an actionable message; exceeding the budget raises
  `BudgetExhausted`.
  - `fetch_page(...)` — one results page; full search surface (query,
    section, tag, from_date, to_date, order_by, page_size, show_fields).
  - `search(...)` — generator yielding flattened records across pages,
    stopping at max_articles / max_pages / last page / budget.
- `to_record(article)` — flatten to
  `{id, url, published, section, headline, byline, trail_text, wordcount, body_text}`.
- `append_records(records, output_fp)` — JSONL append (creates parent dirs).
- `load_existing_ids(output_fp)` — ids already saved (for resume).
- `collect(...)` — high-level entry point: resume-aware, streams `search()`,
  skips duplicates, appends page-by-page (crash-safe), converts
  `BudgetExhausted` into a graceful partial summary. Returns
  `{new, skipped, calls_used, total_available, budget_exhausted}`.

## CLI (`scripts/01_collect_guardian_news.py`)

Full Guardian search surface: `--query` (one or more; share one budget),
`--section`, `--tag`, `--from-date/--to-date`, `--order-by`, `--page-size`,
plus `--max-articles` (default 200/query), `--daily-budget` (default 500),
`--output` (JSONL, default `data/articles/guardian_articles.jsonl`),
resume-by-default with `--no-resume`, `--api-key` override, `--log-level`,
`--log-file`. Exit 0 on success (incl. graceful budget exhaustion), 1 on
config/API errors with a one-line actionable message.

## Notebook arc

Raw request → JSON envelope → `show-fields` (body text!) → filters →
flattening → pagination + 1.05 s pacing → call budgets → incremental JSONL →
resume → tenacity retries → "it all lives in `toolkit.guardian.collect()`" →
the CLI. Runs top-to-bottom on the public `api-key=test` (~15–20 calls).

## Verification

Offline unit checks (retry/budget/resume logic with a mocked session), import
hygiene, CLI `--help`/error paths, notebook executes with the test key, and a
small real-key smoke run (~10 articles) re-run once to confirm resume.
