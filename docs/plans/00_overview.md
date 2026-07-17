# Tutorial overview: Auditing the information environment with LLMs

A hands-on workshop tutorial in five steps. Participants build a full pipeline
that scrapes fresh news, turns it into a quiz, vets the quiz with an LLM judge,
tests several LLM "contestants" on it, and ends with a live human-vs-LLM
"horse race" in the room.

## The five steps

1. **Scrape news from the Guardian** — collect fresh articles (with full body
   text) via the Guardian Content API: pagination, rate limiting, daily call
   budgets, retries with exponential backoff, incremental JSONL saving, and
   resume.
2. **Generate multiple-choice questions** — use an LLM to convert each article
   into MCQs with structured (JSON) outputs.
3. **LLM-as-judge** — a second LLM pass scores each MCQ on faithfulness to
   the source article: the marked correct answer is stated/supported by the
   article (no hallucinated facts) and no other option is equally
   defensible. The judge emits a pass/fail verdict used to filter questions
   for Steps 4–5.
4. **Test LLM answering methods** — evaluate LLMs on the vetted MCQs three
   ways: (1) closed-book (weights/training data only, no web search),
   (2) with web search, (3) a multi-agent debate framework. Covers input
   parameters, structured responses, and batch input.
5. **The horse race** — a live website where participants in the room answer
   the same questions; humans race against each LLM method and results are
   compared live.

## Data contract between steps

```
Step 1  →  data/articles/*.jsonl      (one article per line)
Step 2  →  data/questions/questions_<model>.jsonl  (one MCQ per line, article id attached)
Step 3  →  data/judgments/judgments_<model>.jsonl → judgments_combined.csv
           → data/questions/selected_questions.jsonl (seeded n=100, ≥2 of 3 judges pass)
Step 4  →  data/predictions/*.jsonl   (per-method LLM answers + accuracy)
Step 5  →  live site consumes selected questions + Step 4 results
```

All intermediate data is JSONL (append-safe, streamable, crash-safe).

## Repo conventions

- Each step ships three artifacts:
  1. a teaching notebook `notebooks/0N_*.ipynb` (live-demo walkthrough that
     builds the step's logic piece by piece),
  2. a research-ready CLI script `scripts/0N_*.py` (argparse, logging,
     retries, resume),
  3. source code in the local editable **`toolkit`** package that the script
     imports (uv workspace member; `uv sync` installs it editable).
- API keys come from environment variables (`GUARDIAN_API_KEY`,
  `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`) — never stored in
  the repo. LLM keys are resolved by `toolkit.providers.load_api_key`, which
  prefers `SML_`-prefixed variants (the lab machines' convention).
- Environment management is `uv`-only (`uv sync`); no requirements.txt.

## toolkit module inventory

| Module | Status |
|---|---|
| `toolkit/guardian.py` | **new** (Step 1) — Guardian API client, rate limiter, JSONL persistence/resume |
| `toolkit/questions.py` | **new** (Step 2) — MCQ schema, generation orchestration (threadpool), JSONL/resume |
| `toolkit/judgments.py` | **new** (Step 3) — judgment schema (faithful boolean + rationale), judging orchestration, JSONL/resume |
| `toolkit/prompts.py` | **new** (Step 2) — all prompt text (system constants + user-template builders) |
| `toolkit/config.py` | kept, reworked — keys, paths, `SUPPORTED_MODELS` |
| `toolkit/utils.py` | kept — `setup_logging`, `extract_domain`, `load_jsonl` |
| `toolkit/providers/` | reworked (Step 2) — `load_api_key` (SML_ fallback), OpenAI + Gemini adapters with a shared `run_parsed` interface |
| `io.py`, `metrics.py`, `prompts.py`, `response_structure.py`, `text.py`, `string_helpers.py`, `playwright_helper.py` | deleted (PolitiFact-era) |

PolitiFact-era dependencies (`playwright`, `newspaper4k`, `beautifulsoup4`,
`pyarrow`) were removed from the root `pyproject.toml`; re-add per step if a
later step needs them (`tldextract` stays — `toolkit.utils` imports it).

**End of development:** revisit the packages in the environment (root
`pyproject.toml` dependencies) and clean them up — audit what Steps 1–5
actually import and remove anything unused.

## Target layout

```
docs/plans/           # this overview + one detailed plan per step
notebooks/            # 01_…, 02_…, … teaching notebooks
scripts/              # 01_…, 02_…, … CLI scripts (+ legacy references)
toolkit/toolkit/      # shared package (editable via uv workspace)
data/                 # runtime outputs (git-ignored)
```

Legacy references kept in `scripts/`: `collect_guardian_news.py` (the
original prototype the Step 1 notebook teaches from) and
`mad-agents-fact-checking.py` (multi-agent debate prototype for Step 4).

## Status

- [x] Step 1 — Guardian news collection (see `01_guardian_news.md`)
- [x] Step 2 — MCQ generation (`02_mcq_generation.md`)
- [x] Step 3 — LLM-as-judge (`03_llm_judge.md`)
- [ ] Step 4 — Answering methods (`04_answering_methods.md`)
- [ ] Step 5 — Horse-race website (`05_horse_race_site.md`)
- [ ] End of development — audit and clean up environment dependencies (root `pyproject.toml`)
