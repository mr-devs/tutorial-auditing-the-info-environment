# Computational Approaches to LLM Fact-Checking in Real-Time

Hands-on workspace for a 1-hour IC2S2 tutorial. Five Jupyter notebooks walk
through baseline LLM fact-checking on real PolitiFact claims (closed-book vs.
web-search open-book), live news acquisition, automated test generation,
real-time knowledge audits, and multi-agent debate with heterogeneous models.

Shared code lives in the local **`toolkit`** package (adapted from the
[`llm-vs-human-fc-agreement`] project's toolkit and installed *editable* by
`uv sync`). Design docs: [`docs/PLAN.md`](docs/PLAN.md) (architecture),
[`docs/specs/`](docs/specs/) (per-module specs),
[`docs/PROGRESS.md`](docs/PROGRESS.md) (build status).

## 1. Setup

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- An [OpenAI API key](https://platform.openai.com/api-keys) (notebooks 1, 3, 4)
- An [OpenRouter API key](https://openrouter.ai/keys) (notebook 5)

### Create the environment

From the repository root:

```bash
uv sync
```

This reads `pyproject.toml`, creates `.venv/`, installs every dependency, and
installs the local `toolkit/` package **editable** (it is a uv workspace
member) — edits to `toolkit/` take effect without reinstalling. If you prefer
the requirements-file route instead:

```bash
uv venv
uv pip install -r requirements.txt   # includes `-e ./toolkit`
```

### Install the Playwright browser

Playwright needs a Chromium binary (one-time, ~150 MB):

```bash
uv run playwright install chromium
```

### Export your API keys

```bash
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
```

Add these to your shell profile, or run them in the terminal you will launch
Jupyter/VS Code from. Keys are never stored in the repo. (`SML_`-prefixed
variants, e.g. `SML_OPENAI_API_KEY`, are also honored and take precedence —
same convention as the source project.)

## 2. Data & models

- **Data:** `data/fact_checks/2024-10-10_factchecks_cleaned.parquet` —
  24,964 real PolitiFact fact-checks (scraped 2024-10-10), copied from the
  `llm-vs-human-fc-agreement` project. Notebooks filter to the six
  Truth-O-Meter verdicts and draw a seeded sample.
- **Models:** the second-best current model per provider, all web-search
  capable. Direct OpenAI calls use `gpt-5.6-terra` (Responses API; open-book
  conditions attach the native `web_search` tool). The notebook-5 debate
  roster (OpenRouter ids, in `toolkit.config.DEBATE_MODEL_ROSTER`):
  `openai/gpt-5.6-terra` (explainer/refiner), `anthropic/claude-opus-4.8`
  (debater), `google/gemini-3.5-flash` (typology debater), `x-ai/grok-4.3`
  (judge); debaters get OpenRouter's `web` plugin.

## 3. Launch the notebooks

### Option A — classic Jupyter

```bash
uv run jupyter notebook
```

then open the `notebooks/` folder in the browser UI.

### Option B — VS Code

1. Launch VS Code **from a terminal where the keys are exported** so the
   notebook kernels inherit them:

   ```bash
   code .
   ```

2. Open any notebook under `notebooks/`.
3. Click the kernel picker (top right) → **Python Environments** → select
   **`.venv`** (created by `uv sync`; `ipykernel` is already installed in it).

## 4. Tutorial walkthrough (run in order)

| # | Notebook | What you do | Needs |
|---|----------|-------------|-------|
| 1 | `01_politi_fact_checking.ipynb` | Fact-check a seeded sample of real PolitiFact claims closed-book vs. open-book (web search); score at multi-class/ternary/binary granularity | `OPENAI_API_KEY` |
| 2 | `02_google_news_scraper.ipynb` | Scrape today's Google News Top Stories with Playwright; resolve redirects; save top-3 article texts to `data/articles/` | Chromium, network |
| 3 | `03_question_generation.ipynb` | Turn the articles into validated, strict-JSON multiple-choice questions in `data/questions/` | `OPENAI_API_KEY` |
| 4 | `04_real_time_pipeline.ipynb` | Quiz a fresh, context-free, **tool-less** model on those questions; report real-time knowledge accuracy | `OPENAI_API_KEY` |
| 5 | `05_multi_agent_debate.ipynb` | Run the MADR debate (explainer → debaters → exchange → judge → refiner) across four providers via OpenRouter; compare to notebook 1's baselines | `OPENROUTER_API_KEY` |

Each notebook writes its outputs to `data/`, so later notebooks reload files
rather than recomputing — you can restart kernels freely between modules.

## 5. Notes & troubleshooting

- **Scraping fails or you're offline?** Notebook 2 automatically substitutes
  bundled sample articles (clearly labeled) so notebooks 3–4 still run. It
  also detects Google's rate-limit captcha (`google.com/sorry`), which can
  appear when many participants scrape from one conference network.
- **Sync vs. async Playwright:** notebooks must use the async API with
  top-level `await` (already done); the sync API raises inside Jupyter.
- **OpenRouter model unavailable / rate-limited?** Swap the offending entry
  in `DEBATE_MODEL_ROSTER` in `toolkit/toolkit/config.py` — it is the single
  source of truth for the debate roster (editable install: no reinstall
  needed).
- **Costs:** volumes are tiny (10 claims × 2 conditions, 3 articles, ~9
  questions, 2 debated claims); open-book/web-search calls dominate but a
  complete run remains on the order of tens of cents.

[`llm-vs-human-fc-agreement`]: https://github.com/mr-devs/llm-vs-human-fc-agreement
