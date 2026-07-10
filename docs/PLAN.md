# PLAN — Computational Approaches to LLM Fact-Checking in Real-Time

A 1-hour hands-on workshop workspace (IC2S2 tutorial) consisting of five Jupyter
notebooks that build on one another, plus a local, editable-installed `toolkit`
package adapted from the `llm-vs-human-fc-agreement` project.

## Goal

Participants learn, end-to-end, how to computationally audit LLM fact-checking
behavior:

1. **Baseline evaluation** — closed-book (parametric memory) vs. open-book
   (native web search) fact-checking of real PolitiFact claims, scored at
   multi-class / ternary / binary granularity.
2. **Live data acquisition** — scrape breaking news from Google News with
   Playwright, so the audit targets information *newer than the model's
   training data*.
3. **Synthetic test generation** — automatically turn scraped articles into
   strict-JSON multiple-choice questions (no human annotators needed).
4. **Real-time audit** — quiz a *fresh, context-free, tool-less* model on
   those questions to measure genuine real-time awareness (avoiding context
   leakage).
5. **Multi-agent deliberation** — run the Multi-Agent Debate Refinement (MADR)
   framework with *heterogeneous* models (via OpenRouter) to produce and refine
   fact-checking explanations, comparing against both notebook-1 baselines.

### How the notebooks build on one another

```
01 (closed- vs open-book on PolitiFact sample → data/nb01_results.csv)
        │                                                 └────────────┐
02 (scrape live articles → data/articles/article_X.txt)                │
        │                                                              │
03 (articles → MCQs → data/questions/question_X.json)                  │
        │                                                              │
04 (questions → tool-less real-time audit → data/nb04_report.csv)      │
                                                                       │
05 (MADR debate on the same seeded PolitiFact sample; joins 01 results)
```

## Repository layout

```
.
├── README.md                  # Setup + walkthrough instructions
├── pyproject.toml             # Root project; uv workspace incl. toolkit/
├── requirements.txt           # Mirror (-e ./toolkit + deps)
├── docs/
│   ├── PLAN.md                # This file
│   ├── PROGRESS.md            # Live build-status checklist
│   └── specs/                 # One detailed spec per development chunk
├── toolkit/                   # Local package, installed EDITABLE via uv sync
│   ├── pyproject.toml         # name="toolkit"; deps declared at root (mirrors source project)
│   └── toolkit/
│       ├── __init__.py        # light star-imports (metrics, utils)
│       ├── config.py          # keys, paths, DEFAULT_LLM, DEBATE_MODEL_ROSTER
│       ├── playwright_helper.py  # async article scraper (+ sync wrapper)
│       ├── prompts.py         # SIMPLE_PROMPT (copied) + CLOSED_BOOK_PROMPT (added)
│       ├── text.py            # build_user_text, extract_label_justification
│       ├── response_structure.py # 7-label OpenAIFactCheckingResponse schema
│       ├── metrics.py         # VERACITY_VERDICTS, TO_TERNARY/TO_BINARY, SCENARIOS
│       ├── io.py              # jsonl/resume helpers
│       ├── utils.py           # setup_logging, load_jsonl, extract_domain
│       ├── string_helpers.py  # edit_distance, find_closest_string
│       └── providers/
│           ├── __init__.py    # minimal (OpenAI only)
│           ├── _keys.py       # SML_-prefixed key resolution (std names fallback)
│           └── openai_provider.py # Responses API; use_web_search + text_format params
├── data/
│   ├── fact_checks/2024-10-10_factchecks_cleaned.parquet  # tracked in git
│   ├── articles/              # written by notebook 02 (ignored)
│   └── questions/             # written by notebook 03 (ignored)
└── notebooks/
    ├── 01_politi_fact_checking.ipynb
    ├── 02_google_news_scraper.ipynb
    ├── 03_question_generation.ipynb
    ├── 04_real_time_pipeline.ipynb
    └── 05_multi_agent_debate.ipynb
```

## The `toolkit` package (provenance)

Copied from `llm-vs-human-fc-agreement/code/toolkit/toolkit/` and lightly
adapted; installed **editable** through the uv workspace (root
`[tool.uv.sources] toolkit = { workspace = true }`), so `uv sync` is the only
install step and edits to `toolkit/` take effect without reinstalling.

| Module | Status |
|--------|--------|
| `response_structure.py`, `metrics.py`, `text.py`, `io.py`, `utils.py`, `string_helpers.py`, `providers/_keys.py` | copied verbatim |
| `prompts.py` | copied + `CLOSED_BOOK_PROMPT` added |
| `providers/openai_provider.py` | copied + `use_web_search` / `text_format` params and `run_parsed()` added |
| `config.py`, `playwright_helper.py` | tutorial-local |
| `parsers.py` | **not copied** — bound to the source project's saved-JSONL corpus |
| `plotting.py` | **not copied** — no figures in the tutorial |
| `url_utils.py` | **not copied** — Gemini grounding-redirect specific |
| `providers/{anthropic,gemini,perplexity,xai}_provider.py` | **not copied** — would drag four SDKs; heterogeneity flows through OpenRouter |
| `data_models.py` | **not copied** — empty stub |

The toolkit declares zero dependencies (same convention as the source
project); runtime deps live in the root `pyproject.toml`.

## Data

`data/fact_checks/2024-10-10_factchecks_cleaned.parquet` — 24,964 real
PolitiFact fact-checks scraped 2024-10-10, copied from
`llm-vs-human-fc-agreement/data/processed/fact_checks/`. Columns: `verdict`
(6 Truth-O-Meter + 3 Flip-O-Meter labels; the latter filtered out via
`metrics.VERACITY_VERDICTS`, leaving 24,611 rows), `statement`,
`statement_originator`, `statement_date`, `factchecker_name`,
`factcheck_date`, `topics`, `factcheck_analysis_link` (unique key),
`date_retrieved`, `page`. Notebooks 1 and 5 draw the **same seeded sample**
(seed 303, n=10) so their results join on `factcheck_analysis_link`.

## Models

- `DEFAULT_LLM = "gpt-5.6-terra"` (second-best OpenAI tier: sol > terra >
  luna; verified against the live OpenAI models endpoint). Used via the
  Responses API; open-book calls attach the native `web_search` tool.
- `DEBATE_MODEL_ROSTER` (OpenRouter ids, second-best current model per
  provider, verified against the live OpenRouter catalog):
  explainer/refiner `openai/gpt-5.6-terra`, debater_general
  `anthropic/claude-opus-4.8`, debater_typology `google/gemini-3.5-flash`,
  judge `x-ai/grok-4.3`. (The judge replaced the original `meta-llama`
  entry: Meta's newest lineup, Apr-2025 Llama-4, is the weakest of the
  current majors.) Debater roles get OpenRouter's model-agnostic web-search
  plugin (`plugins=[{"id": "web"}]`).

## Environment & dependency management

- **`uv`** manages everything: `uv sync` builds `.venv/`, installs all deps,
  and editable-installs `toolkit/` (workspace member). Alternative:
  `uv venv` + `uv pip install -r requirements.txt` (which contains
  `-e ./toolkit`).
- Playwright browser binary: `uv run playwright install chromium`.
- API keys from env: `OPENAI_API_KEY` and `OPENROUTER_API_KEY` (the copied
  `_keys.py` also honors `SML_`-prefixed variants, preferring them).

## Development chunks

| # | Chunk | Spec |
|---|-------|------|
| 0 | Shared infrastructure (toolkit package, packaging, data) | [specs/00_shared_infrastructure.md](specs/00_shared_infrastructure.md) |
| 1 | Notebook 1 — closed- vs open-book baseline | [specs/01_politi_fact_checking.md](specs/01_politi_fact_checking.md) |
| 2 | Notebook 2 — Google News scraper | [specs/02_google_news_scraper.md](specs/02_google_news_scraper.md) |
| 3 | Notebook 3 — Question generation | [specs/03_question_generation.md](specs/03_question_generation.md) |
| 4 | Notebook 4 — Real-time pipeline audit | [specs/04_real_time_pipeline.md](specs/04_real_time_pipeline.md) |
| 5 | Notebook 5 — Multi-agent debate (MADR) | [specs/05_multi_agent_debate.md](specs/05_multi_agent_debate.md) |

## Assumptions, open questions, and known limitations

- **Live scraping is inherently brittle.** Google News markup drifts (story
  links are `./read/...` as of mid-2026), story URLs are JS redirect shells,
  and Google serves a `google.com/sorry` captcha under bursty automated
  access. Notebook 2 resolves redirects explicitly, detects the captcha, and
  falls back to bundled sample articles so notebooks 3–4 always run.
- **API rate limits / cost.** Volumes stay tiny (10 claims, 3 articles,
  ~9 questions, 2 debated claims); outputs are serialized to `data/` and
  reloaded rather than recomputed.
- **Open-book ≠ curated.** The web-search conditions measure what the model
  does with *uncurated* live retrieval — the ACL 2026 caveat applies and is
  part of the teaching narrative.
- **OpenRouter roster availability.** If a roster model is unavailable or
  rate-limited, `toolkit.config.DEBATE_MODEL_ROSTER` is the single place to
  swap substitutes; notebook 5 degrades gracefully per-call.
- **Structured outputs across providers.** Not every OpenRouter provider
  enforces `json_schema` strictly; notebook 5 validates with Pydantic and
  falls back to extracting the first balanced JSON object.
- **Real-time audit validity.** Notebook 4's results depend on the news of
  the run day — variance across participants is expected and is itself a
  teaching point.
