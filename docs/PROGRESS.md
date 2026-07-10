# PROGRESS

Live status of each development chunk. Statuses: `not started` / `in progress` / `complete`.

## Revision 2 — editable toolkit, real PolitiFact data, current models (2026-07-09)

| Chunk | Status | Notes |
|-------|--------|-------|
| 00 — toolkit package + packaging + data | complete | Modules copied from llm-vs-human-fc-agreement; uv workspace editable install verified; parquet copied and tracked in git |
| 01 — closed- vs open-book notebook | complete | Real PolitiFact sample (seed 303, n=10); SCENARIOS scoring; live smoke test passed (gpt-5.6-terra, both conditions) |
| 02 — Google News scraper notebook | complete | Simplified to the current `./read` selector only; redirect resolution + captcha/offline fallback retained |
| 03 — question generation notebook | complete | Migrated to toolkit provider (Responses API, no tools) |
| 04 — real-time pipeline notebook | complete | Migrated to toolkit provider; explicit no-web-search rationale added |
| 05 — multi-agent debate notebook | complete | New roster (2nd-best per provider incl. x-ai judge), OpenRouter web plugin on debaters, 7-label vocabulary, joins NB1 baselines; live run needs OPENROUTER_API_KEY (not set in build session) |
| Docs + README | complete | PLAN/specs/README rewritten for toolkit layout, data provenance, model roster |

## Revision 1 — initial build (2026-07-09)

| Chunk | Status | Notes |
|-------|--------|-------|
| 00 — Shared infrastructure | complete | uv sync builds .venv cleanly; openrouter pin relaxed to >=0.11.0 (no 1.0 on PyPI) |
| 01 — PolitiFact baseline notebook | complete | Superseded by revision 2 (mock CSV replaced by real parquet) |
| 02 — Google News scraper notebook | complete | Verified live: ./read selector + two-hop redirect resolution added after real-site testing |
| 03 — Question generation notebook | complete | Verbatim-evidence validation gate included |
| 04 — Real-time pipeline notebook | complete | Per-article accuracy + failure-justification review |
| 05 — Multi-agent debate notebook | complete | Superseded by revision 2 (roster + data changes) |
| README | complete | Superseded by revision 2 |
