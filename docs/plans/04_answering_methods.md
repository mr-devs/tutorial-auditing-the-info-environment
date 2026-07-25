# Step 4 — Testing LLMs on the MCQs, three ways

**Status: built.** Toolkit modules (`toolkit/answers.py`,
`toolkit/debate.py`), three CLI scripts, teaching notebook
(`notebooks/04_answering_methods.ipynb`).

## Pipeline

```
data/questions/selected_questions.jsonl     (Step 3 output, the vetted quiz)
        |
        +--> scripts/04-1_generate_answers.py         (per METHOD x MODEL)
        |        -> data/answers/answers_<method>_<model>.jsonl
        |
        +--> scripts/04-2_generate_debate_answers.py  (per GPT MODEL)
                 -> data/answers/answers_debate_<model>.jsonl
        |
        v
scripts/04-3_combine_answers.py             (glob -> validate -> tidy CSV)
        |
        v
data/answers/answers_combined.csv       (one row per question x model x method)
```

## The three conditions

Every contestant sees ONLY the question and its four lettered options —
never the article. The questions are about news published after the models'
training cutoffs, so the gap between conditions is the experiment:

1. **closed_book** — no tools. Measures what the weights know
   (1 call/question). All 6 models.
2. **web_search** — the identical prompt with the provider's built-in search
   tool enabled (OpenAI `web_search`, Gemini `google_search`). Measures what
   retrieval adds (1 call/question). The record's `search_used` field notes
   whether the raw response shows the tool was actually invoked. All 6
   models.
3. **debate** — society of minds (Du et al. 2023): `DEBATE_N_AGENTS` (3)
   copies of the same model answer independently, then revise over
   `DEBATE_N_ROUNDS` (2) rounds after reading the OTHER agents'
   previous-round answers. Measures what deliberation adds
   (3 x (1 + 2) = **9 calls/question**). **GPT models only** — the debate
   runs on the openai-agents SDK, which targets OpenAI's Responses API.

### Debate implementation (openai-agents SDK)

`toolkit/debate.py` builds each debater as an SDK `Agent` (same system
prompt, `output_type=Answer` for SDK-validated structured output) and runs
turns through `Runner.run`. Round structure stays a deterministic Python
loop — a debate's turn order is fixed, so LLM-driven handoffs have nothing
to decide (same reasoning as the legacy prototype
`scripts/mad-agents-fact-checking.py`). The SDK is async-native, so:

- the 3 agents within a round run concurrently (`asyncio.gather`);
- the orchestrator (`debate_questions`) is one asyncio event loop with a
  semaphore capping in-flight debates (`--parallel` / `--max-workers`,
  default 4) instead of a thread pool;
- notebooks `await debate_question_async(...)` (Jupyter already runs a
  loop); scripts use the sync `debate_question` / `debate_questions`.

The SDK's API key is routed through `load_api_key` (SML_ fallback) via
`set_default_openai_client`, with a loop-fresh `AsyncOpenAI` per run;
tracing uploads are disabled.

### Debate verdict (deterministic)

Majority vote over the final round. Ties (only 1-1-1 with K=3) go to the
letter with the highest mean self-reported confidence among its voters;
residual ties go to the alphabetically first letter. The record stores
`vote_counts` and which `tie_break` rule (if any) fired, plus the full
per-round `transcript`, so debates are auditable end to end.

## Contestant models

`config.ANSWER_MODELS` = `SUPPORTED_MODELS` ∪ `JUDGE_MODELS` (6 models);
`config.DEBATE_MODELS` = the OpenAI subset (4 models):

| Model | Provider | closed_book / web_search | debate |
|---|---|---|---|
| gpt-5.6-terra | openai | yes | yes |
| gpt-5.6-luna | openai | yes | yes |
| gpt-5.5-2026-04-23 | openai | yes | yes |
| gpt-5.4-mini-2026-03-17 | openai | yes | yes |
| gemini-3.5-flash | gemini | yes | — |
| gemini-3.1-flash-lite | gemini | yes | — |

## Record schemas

JSONL (per answer): `id` (`{method}__{model}__{question_id}`), `question_id`,
`article_id`, `method`, `model`, `provider`, `answer_letter`,
`correct_letter`, `is_correct` (graded on the spot), `confidence`,
`reasoning`, `search_used` (bool, web_search only), `debate` (nested dict,
debate only), `answered_at`.

CSV (from 04-3, one row per answer, schema-validated per line, `is_correct`
recomputed defensively): `question_id, article_id, method, model, provider,
answer_letter, correct_letter, is_correct, confidence, search_used,
reasoning, answered_at`. The nested `debate` transcripts are dropped — they
live only in the JSONL.

## Gemini web search + structured output

Gemini 3-series models accept the `google_search` tool together with
`response_mime_type="application/json"` + `response_schema` in a single
`generate_content` call (verified live with `gemini-3.1-flash-lite`; the
combination was rejected in the Gemini 2.x era). `gemini_provider.run_parsed`
therefore takes the same `use_web_search=False` flag as the OpenAI adapter.
If a future model 400s on the combination, the documented fallback is two
calls inside `_call`: a schema-free grounded call, then a structured
extraction call (merging `grounding_metadata` into the raw dict).

## Commands

```bash
# Smoke test (5 questions, one cheap model)
uv run python scripts/04-1_generate_answers.py \
    --model gpt-5.4-mini-2026-03-17 --method closed_book \
    --max-questions 5 --parallel

# Single-call sweep: 100 questions x 6 models x 2 methods = 1,200 calls
for METHOD in closed_book web_search; do
  for M in gpt-5.4-mini-2026-03-17 gpt-5.5-2026-04-23 gpt-5.6-luna \
           gpt-5.6-terra gemini-3.1-flash-lite gemini-3.5-flash; do
    uv run python scripts/04-1_generate_answers.py \
        --model $M --method $METHOD --parallel
  done
done

# Debate sweep: 100 questions x 4 GPT models x 9 calls = 3,600 calls
for M in gpt-5.4-mini-2026-03-17 gpt-5.5-2026-04-23 \
         gpt-5.6-luna gpt-5.6-terra; do
  uv run python scripts/04-2_generate_debate_answers.py --model $M --parallel
done

# Merge to the analysis CSV
uv run python scripts/04-3_combine_answers.py \
    --input-dir data/answers --glob 'answers_*.jsonl'
```

All runs share the Steps 2–3 contract: fail-fast key check, JSONL append
(crash-safe), resume keyed on `question_id` (safe because each file is one
method x model). 04-1 parallelizes with a ThreadPoolExecutor; 04-2 with an
asyncio semaphore (each debate already bursts 3 concurrent calls per
round, so keep `--max-workers` modest).

## Teaching notebook

`notebooks/04_answering_methods.ipynb`: the contestant prompt (and why the
article must never leak into it), one live closed-book vs. web-search call
on the same question, one live debate via the openai-agents SDK (top-level
`await` in Jupyter) with the transcript printed, the CLI sweep, then
figures from `answers_combined.csv` — accuracy by method x model, the
closed-book → web-search paired gain per model, and final-round vote
splits / outcome changes across the debates.

## Step 5 handoff

The horse-race site consumes `selected_questions.jsonl` (the quiz) and
`answers_combined.csv` (the LLM baselines per method).
