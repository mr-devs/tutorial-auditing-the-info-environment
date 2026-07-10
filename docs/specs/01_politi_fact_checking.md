# Spec 01 — Notebook 1: Closed-book vs. open-book baseline fact-checking

## Learning objective / role in the narrative

Establish the baseline the rest of the tutorial audits: how well does an LLM
fact-check real PolitiFact claims (a) from parametric memory alone
(closed-book, no tools) versus (b) with live **web search** attached
(open-book, uncurated retrieval)? Introduces the structured-output +
provider pattern used throughout, and the multi-granularity scoring from the
source project.

Instructional markdown covers parametric memory vs. retrieval-augmented
evaluation, and Matthew DeVerna's ACL 2026 findings that uncurated search
results and reasoning chains frequently fail to confirm complex political
claims — search access is not a guarantee of better verdicts.

## Implementation plan

1. Setup cell: chdir to repo root; imports from `toolkit.config`,
   `toolkit.metrics`, `toolkit.prompts`, `toolkit.providers.openai_provider`,
   `toolkit.text`.
2. Load `data/fact_checks/2024-10-10_factchecks_cleaned.parquet`; filter
   `verdict ∈ VERACITY_VERDICTS` (drops Flip-O-Meter rows); seeded sample
   (seed 303, n=10) — identical in notebook 5 so results join on
   `factcheck_analysis_link`.
3. `fact_check(row, open_book)` → `openai_provider.run_parsed(DEFAULT_LLM,
   SIMPLE_PROMPT | CLOSED_BOOK_PROMPT, build_user_text(originator,
   statement), use_web_search=open_book)`; returns validated
   `OpenAIFactCheckingResponse`.
4. Closed-book loop, then open-book loop (same claims, same model).
5. Scoring with `metrics.SCENARIOS`: accuracy at multi_class / ternary /
   binary granularity per condition, plus the "Not enough information" rate.
6. Persist long-format results to `data/nb01_results.csv`
   (`factcheck_analysis_link`, `statement`, `condition`, `gold`, `predicted`,
   `justification`); pivot cell showing claims where web search changed the
   verdict.

## Pydantic schemas

`toolkit.response_structure.OpenAIFactCheckingResponse` (copied from the
source project) — no notebook-local schema.

## Inputs / outputs / dependencies

- Inputs: `OPENAI_API_KEY` (or `SML_OPENAI_API_KEY`); the tracked parquet.
- Outputs: `data/nb01_results.csv` (both conditions; consumed by notebook 5).
- Depends on chunk 00.

## Edge cases / failure modes

- Label validity is enforced by the schema's `Literal` — no normalization
  needed.
- NEI predictions are never "correct" (gold is always a real verdict) but
  their rate is reported as an honesty signal.
- API errors mid-loop → try/except per claim (on top of the provider's
  tenacity retry); record `None` and continue.
- Open-book calls are slow (live retrieval) — sample kept at n=10.
