# Spec 05 — Notebook 5: Multi-agent debate (MADR)

## Learning objective / role in the narrative

Move beyond a single model's judgment: implement the Multi-Agent Debate
Refinement (MADR) framework, where *heterogeneous* models (different
providers/families, routed via OpenRouter) generate, critique, cross-examine,
adjudicate, and refine a fact-checking explanation. Compare the refined output
against notebook 1's closed-book and open-book single-model verdicts on the
same claims.

Five structured steps taught in markdown and implemented as functions:

1. **Explain** — zero-shot initial explanation of a claim's veracity.
2. **Debate** — two independent debaters critique the explanation; one is
   guided by a structured error typology (factual inconsistency, unsupported
   inference, missing context, logical fallacy). Both debaters have web
   search enabled (OpenRouter `web` plugin) so critiques can check facts
   against live evidence.
3. **Exchange** — debaters read each other's critiques and refine them,
   surfacing contradictions neither caught alone.
4. **Judge** — a judge reviews the exchanged critiques and establishes
   consensus on which are valid.
5. **Refine** — a refiner integrates consensus feedback into a final,
   verified explanation with a final veracity label and confidence.

Roster (`toolkit.config.DEBATE_MODEL_ROSTER`, second-best current model per
provider, OpenRouter ids): explainer/refiner `openai/gpt-5.6-terra`,
debater_general `anthropic/claude-opus-4.8`, debater_typology
`google/gemini-3.5-flash`, judge `x-ai/grok-4.3`. Verdicts use the same
7-label vocabulary as notebook 1 (`toolkit.metrics.MULTI_CLASS_ORDER`).

## Implementation plan

1. Setup cell; `get_openrouter_client()`; load the parquet and draw the
   **same seeded sample as notebook 1** (seed 303, n=10); load
   `data/nb01_results.csv` if present and pivot to closed-book/open-book
   columns keyed by `factcheck_analysis_link`.
2. Structured-output helper for OpenRouter: strict JSON schema from each
   Pydantic model (`model_json_schema()` + `additionalProperties: false`),
   passed as `response_format={"type": "json_schema", ...}` to
   `client.chat.send(..., stream=False)`; validate with
   `Model.model_validate_json`; fall back to extracting the first balanced
   JSON object. `web_search=True` roles add `plugins=OPENROUTER_WEB_PLUGIN`.
3. Define the four stage schemas (below), label descriptions built from
   `MULTI_CLASS_ORDER`.
4. Implement `generate_initial_explanation`, `run_debater` (typology flag
   switches the prompt; web search on), `exchange_feedback` (web search on),
   `judge_consensus`, `refine_explanation` — each pulls its model from the
   roster.
5. `run_madr_pipeline(row)` chains the five steps and prints each stage's
   intermediate output (role, model id, parsed content).
6. Run on the first 2 sampled claims (~9 API calls per claim; two with live
   search); save `data/nb05_madr_results.csv`.
7. Comparison cell: gold vs. closed-book vs. open-book vs. MADR labels, plus
   ternary-collapsed versions (`TO_TERNARY`) for readability.

## Pydantic schemas

`InitialExplanation` (veracity, explanation), `DebaterCritique`
(identified_errors, error_types [default []], proposed_revisions),
`JudgeConsensus` (agreed_errors, unresolved_disagreements [default []],
consensus_summary), `RefinedExplanation` (final_veracity, final_explanation,
confidence).

## Inputs / outputs / dependencies

- Inputs: `OPENROUTER_API_KEY`; the tracked parquet; optionally
  `data/nb01_results.csv` from chunk 01.
- Outputs: printed stage-by-stage debate transcripts; comparison table;
  `data/nb05_madr_results.csv`.
- Depends on chunks 00 and 01.

## Edge cases / failure modes

- A roster model is unavailable/rate-limited on OpenRouter → per-call
  try/except with the error surfaced; guidance to swap the roster entry in
  `toolkit/toolkit/config.py`.
- Provider ignores `json_schema` → Pydantic validation fails →
  JSON-extraction fallback; if that also fails the claim is skipped with a
  report.
- Missing `nb01_results.csv` → comparison degrades to gold vs. MADR only,
  with a note to run notebook 1 first.
- Long debate transcripts → prompts pass only the artifacts each stage needs
  (not full history) to control tokens.
