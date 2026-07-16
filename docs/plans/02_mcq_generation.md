# Step 2 — Generate multiple-choice questions from news articles

**Status: built.**

## Goal

Turn Step 1's articles into a machine-generated quiz: one LLM call per
article produces N validated multiple-choice questions. Teach the room
system/user message design, structured outputs on two providers (OpenAI +
Gemini), prompt craft for quiz questions, and `ThreadPoolExecutor` for
parallel I/O-bound API calls (no batch APIs).

## Artifacts

| Artifact | Path |
|---|---|
| Teaching notebook | `notebooks/02_question_generation.ipynb` |
| CLI script | `scripts/02_generate_questions.py` |
| Toolkit modules | `toolkit/toolkit/questions.py`, `toolkit/toolkit/prompts.py`, reworked `toolkit/toolkit/providers/` |

## Design

- **Keys** — `toolkit.providers.load_api_key(key_name)`: prefers the lab
  machines' `SML_`-prefixed env var (`SML_OPENAI_API_KEY`), falls back to the
  standard name, logs which was used, raises `ValueError` if neither is set.
  All LLM clients load keys through it.
- **Provider adapters** — `openai_provider` (Responses API,
  `responses.parse` with `text_format=`) and `gemini_provider` (`google-genai`,
  `system_instruction` + `response_schema`) share one interface:
  `run_parsed(model, system_prompt, user_text, response_format) -> (parsed, raw)`.
  Both retry only transient failures (429/5xx/network) with exponential
  backoff; both clients are thread-safe and cached per process.
- **Prompts** — `toolkit.prompts` holds `MCQ_SYSTEM_PROMPT` (quiz-writer role
  + rules: grounded in the article, stand-alone phrasing, plausible
  same-granularity distractors, no all/none-of-the-above or length/letter
  tells, explanation quotes the article) and `build_mcq_user_prompt(headline,
  body_text, n_questions)`. Every call uses both a system and a user message.
- **Schema** — `MCQuestion(question, options[4], correct_letter A–D
  (Literal→enum), explanation)` inside `ArticleQuestions`; field descriptions
  ride along in the JSON schema both SDKs send.
- **Records** — one JSONL line per question:
  `{id: "<provider>__<article_id>__q<i>", article_id, question_index,
  provider, model, question, options, correct_letter, explanation,
  generated_at}`.
- **Orchestration** — `generate_for_articles(articles, output_fp, provider,
  model, n_questions, parallel, max_workers, resume)`: fails fast on missing
  keys before spawning threads; sequential loop or
  `ThreadPoolExecutor`+`as_completed`; only the main thread writes the file
  (single-writer pattern, no locks); per-article error isolation (failures
  logged, retried on resume); append-per-article crash safety; summary dict.

## CLI

`--model` (required) selects the generator; the provider is inferred. Allowed
models (`toolkit.config.SUPPORTED_MODELS`): `gpt-5.4-mini-2026-03-17`,
`gpt-5.6-terra`, `gemini-3.1-flash-lite`, `gemini-3.5-flash`. Plus
`--input`, `--n-questions`, `--max-articles`, `--parallel`, `--max-workers`,
`--output` (default `data/questions/questions_<model>.jsonl`), `--no-resume`,
and the standard logging flags.

Note (2026-07-16): `gemini-2.5-flash-lite` was originally in the list but
returns 404 ("no longer available to new users"); it was replaced with
`gemini-3.5-flash`.

## Verification

Offline: `load_api_key` env permutations; threadpool integrity with mocked
provider (10 articles/5 workers → 30 lines, ~5× speedup); error isolation
(1 failure → 9 written); resume (re-run retries only the failure); input
dedup; CLI error paths. Live: 2 articles × both providers validated
(4 options, legal letters); notebook executed end-to-end (~14 calls,
3.8× measured speedup in the timing demo).
