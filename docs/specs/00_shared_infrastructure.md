# Spec 00 — Shared infrastructure (toolkit package + packaging + data)

## Learning objective / role in the narrative

Not a teaching module itself; provides the plumbing every notebook imports so
participants spend the hour on concepts, not setup. The core is a local
**`toolkit`** package — modules copied from the
`llm-vs-human-fc-agreement` project's toolkit and lightly adapted — installed
**editable** via a uv workspace.

## Implementation plan

1. **`toolkit/` package** (uv workspace member):
   - `toolkit/pyproject.toml`: `name="toolkit"`, setuptools backend,
     `dependencies = []` (runtime deps declared at the root — same convention
     as the source project).
   - Copied verbatim from
     `llm-vs-human-fc-agreement/code/toolkit/toolkit/`:
     `response_structure.py` (7-label `OpenAIFactCheckingResponse`),
     `metrics.py` (`VERACITY_VERDICTS`, `TO_TERNARY`, `TO_BINARY`,
     `SCENARIOS`), `text.py` (`build_user_text`,
     `extract_label_justification`), `io.py`, `utils.py`,
     `string_helpers.py`, `providers/_keys.py` (SML_-prefixed key resolution
     with standard-name fallback).
   - Copied + edited: `prompts.py` (added `CLOSED_BOOK_PROMPT`),
     `providers/openai_provider.py` (added `use_web_search` and `text_format`
     params plus `run_parsed()` returning the validated Pydantic object).
   - Tutorial-local: `config.py` (keys, paths, `DEFAULT_LLM`,
     `DEBATE_MODEL_ROSTER`, `OPENROUTER_WEB_PLUGIN`, client factories) and
     `playwright_helper.py` (async scraper + sync wrapper).
   - Not copied (rationale in PLAN.md): `parsers.py`, `plotting.py`,
     `url_utils.py`, non-OpenAI providers, `data_models.py`.
2. **Root `pyproject.toml`** — depends on `toolkit` via
   `[tool.uv.workspace] members = ["toolkit"]` +
   `[tool.uv.sources] toolkit = { workspace = true }`, so `uv sync`
   editable-installs it. Deps: openai, openrouter (>=0.11 — no 1.0 on PyPI),
   playwright, pydantic, pandas, pyarrow, beautifulsoup4, tenacity,
   tldextract, typing_extensions, ipykernel, notebook.
3. **`requirements.txt`** — mirror, with `-e ./toolkit` at the top.
4. **Data** — `data/fact_checks/2024-10-10_factchecks_cleaned.parquet`
   copied from the source project (24,964 PolitiFact fact-checks) and
   tracked in git via a `.gitignore` negation (`data/*` ignored otherwise).

## Pydantic schemas

`toolkit.response_structure.OpenAIFactCheckingResponse` — `label` is a
`Literal` over the six Truth-O-Meter labels plus "Not enough information";
`justification: str`. Notebook-specific schemas (MCQs, evaluation verdicts,
debate stages) are defined in the notebooks where they are taught.

## Inputs / outputs / dependencies

- Inputs: environment variables (`OPENAI_API_KEY` / `SML_OPENAI_API_KEY`,
  `OPENROUTER_API_KEY`).
- Outputs: importable `toolkit` package; reproducible `.venv`; tracked
  parquet dataset.
- Depended on by every notebook (no sys.path hacks — the package is
  installed).

## Edge cases / failure modes

- Missing API keys → `RuntimeError` from `_keys.resolve_api_key` (OpenAI
  path) or `ValueError` from `config.get_openrouter_client`, at client
  construction rather than deep in a request stack.
- Playwright browser binary not installed → helper returns an error string;
  README covers `uv run playwright install chromium`.
- Sync scraping helper called inside Jupyter → would raise; docstring warns
  and notebooks use `await` exclusively.
