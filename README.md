# Auditing the Information Environment with LLMs

Hands-on workshop tutorial. In five steps we build a complete pipeline that
scrapes fresh news, turns it into a quiz, vets the quiz with an LLM judge,
tests LLM "contestants" on it three different ways, and finishes with a live
**horse race**: humans in the room vs. the LLMs.

| Step | Notebook | Script | What happens |
|---|---|---|---|
| 1 | `notebooks/01_guardian_news_collection.ipynb` | `scripts/01_collect_guardian_news.py` | Scrape Guardian articles (full body text) into JSONL |
| 2 | *(coming)* | *(coming)* | LLM generates multiple-choice questions from the articles |
| 3 | *(coming)* | *(coming)* | LLM-as-judge vets each question (quality, faithfulness, difficulty) |
| 4 | *(coming)* | *(coming)* | LLMs answer the quiz: closed-book vs. web search vs. multi-agent debate |
| 5 | *(coming)* | *(coming)* | Live website: humans vs. LLM methods, compared in real time |

Each step ships a **teaching notebook** (the live walkthrough), a
**research-ready CLI script**, and shared source code in the local
**`toolkit`** package (installed *editable* by `uv sync`). Plans and design
notes live in [`docs/plans/`](docs/plans/00_overview.md).

## Setup

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A free [Guardian API key](https://open-platform.theguardian.com/access/) (Step 1)
- An [OpenAI API key](https://platform.openai.com/api-keys) (Steps 2–4)
- An [OpenRouter API key](https://openrouter.ai/keys) (Step 4)

### Create the environment

From the repository root:

```bash
uv sync
```

This reads `pyproject.toml`, creates `.venv/`, installs every dependency, and
installs the local `toolkit/` package **editable** (it is a uv workspace
member) — edits to `toolkit/` take effect without reinstalling.

### Export your API keys

```bash
export GUARDIAN_API_KEY="..."
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
```

Add these to your shell profile, or run them in the terminal you launch
Jupyter/VS Code from. Keys are never stored in the repo.

> **No Guardian key yet?** The Guardian's shared public key — literally the
> string `test` — works for light experimentation, and the Step 1 notebook
> falls back to it automatically.

## Launch the notebooks

```bash
uv run jupyter notebook
```

or open the repo in VS Code (launched from a terminal where the keys are
exported), open a notebook under `notebooks/`, and select the `.venv`
kernel.

## Step 1 quick start

Teach yourself the pieces in the notebook, then collect for real with the
CLI:

```bash
uv run python scripts/01_collect_guardian_news.py \
    --query "climate policy" "misinformation" \
    --from-date 2026-07-01 \
    --max-articles 200 \
    --output data/articles/guardian_articles.jsonl
```

The script paces itself under the free tier's 1 call/second, stops cleanly at
the 500 calls/day budget, retries transient failures with exponential
backoff, saves incrementally (crash-safe JSONL), and **resumes**: re-running
the same command skips articles already saved. `--help` documents the full
Guardian search surface (query, section, tag, date window, ordering, ...).

## Repository layout

```
docs/plans/           # tutorial overview + one detailed plan per step
notebooks/            # teaching notebooks (01_, 02_, ...)
scripts/              # research-ready CLI scripts (+ legacy prototypes)
toolkit/toolkit/      # shared package: guardian.py, config.py, utils.py, providers/
data/                 # runtime outputs (git-ignored)
```
