# Step 3 — LLM-as-judge analysis of the MCQs

**Status: scripts built; teaching notebook deferred.**

## Goal

Vet the Step 2 questions with LLM judges before humans and LLM contestants
see them. Three judge models each evaluate every question; questions passing
at least 2 of 3 judges are eligible, and a seeded random subset advances to
Steps 4–5.

## Pipeline

```
questions_<gen-model>.jsonl + guardian_articles.jsonl
        │  scripts/03-1_generate_judgments.py   (run once per judge model)
        ▼
data/judgments/judgments_<judge-model>.jsonl   (one judgment per line)
        │  scripts/03-2_combine_judgments.py --input-dir ... --glob 'judgments_*.jsonl'
        ▼
data/judgments/judgments_combined.csv   (tidy: one row per question x judge_model)
        │  scripts/03-3_select_questions.py --input ... --questions ...
        ▼
data/questions/selected_questions.jsonl   (seeded random n=100 passers)
```

## Judgment design

- Judge input: article headline + full body text, the question, its options
  lettered by list order (`zip("ABCD", options)`), and the marked correct
  letter. The generator's explanation is withheld so the judge assesses the
  question against the article alone.
- Structured output (`toolkit.judgments.Judgment`): two booleans + one
  short rationale —
  - `answerable` — well-formed, unambiguous, exactly one option defensible
    given the article;
  - `faithful` — the marked correct option is stated/supported by the
    article (not hallucinated).
- **Passing** (per model): `answerable & faithful` (both True). A question
  is selected only if ≥ `--min-passing` (default 2) judge models pass it.
- A third `guessable` dimension (answerable without the article) existed in
  the original design and was removed 2026-07-16 — judges marked nearly all
  widely-reported facts guessable, filtering too aggressively.
- Judge models (`toolkit.config.JUDGE_MODELS`, all OpenAI): `gpt-5.6-luna`,
  `gpt-5.5-2026-04-23`, `gpt-5.4-mini-2026-03-17`.
- Orchestration mirrors Step 2: `--parallel` ThreadPoolExecutor, single-writer
  appends, per-question error isolation, resume keyed on question id,
  fail-fast key check via `load_api_key`.

## Artifacts

| Artifact | Path |
|---|---|
| Judge one model | `scripts/03-1_generate_judgments.py` |
| Merge to CSV | `scripts/03-2_combine_judgments.py` |
| Select final set | `scripts/03-3_select_questions.py` |
| Toolkit module | `toolkit/toolkit/judgments.py` (+ judge prompts in `toolkit/toolkit/prompts.py`) |
| Teaching notebook | *deferred* |

## Notes

- `03-2_combine_judgments.py` validates every JSONL line against a Pydantic
  record schema; invalid lines are skipped with warnings.
- `03-3_select_questions.py` is deterministic given the same seed and inputs
  (default seed 42, n 100); if fewer than n questions pass it writes all
  passers with a warning (exit 0).
