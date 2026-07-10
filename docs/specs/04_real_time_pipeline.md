# Spec 04 — Notebook 4: Real-time pipeline audit

## Learning objective / role in the narrative

Close the loop: quiz the model on questions about *today's* news in a fresh,
unassisted context window. Central concept: **context leakage** — if the
evaluator sees the scraped article (or shares a conversation with the
generator), the audit measures reading comprehension, not real-time knowledge.
Each question is therefore an independent API request containing only the
question and options, never the article.

**Deliberately no web search** (`use_web_search=False` on every call): unlike
Notebook 1's open-book condition, this audit measures *parametric* real-time
awareness — what the model knows with no tools at all. The markdown makes the
contrast explicit and suggests rerunning with search enabled as an exercise
(which converts the audit into a measure of live-retrieval quality).

## Implementation plan

1. Setup cell; load every `data/questions/question_X.json`; flatten to a list
   of (article, question) records.
2. Define `EvaluationVerdict` (chosen_option_index, justification).
3. `answer_question(question, options)` via
   `toolkit.providers.openai_provider.run_parsed(DEFAULT_LLM, ...,
   use_web_search=False, text_format=EvaluationVerdict)` — system prompt says
   "answer from your own knowledge; if unsure, pick the most plausible
   option". One fresh call per question; no shared history, no tools.
4. Loop over all questions; record chosen vs. correct index, justification.
5. Report: overall accuracy vs. the 25 % random-guess floor, per-article
   accuracy table (pandas), and a qualitative look at failures (justifications
   reveal whether the model knew the event or guessed).
6. Persist `data/nb04_report.csv`; markdown discussion of interpretation
   (training-cutoff effects, day-of-run variance).

## Pydantic schemas

```python
class EvaluationVerdict(BaseModel):
    chosen_option_index: int
    justification: str
```

## Inputs / outputs / dependencies

- Inputs: `data/questions/question_{1..3}.json` (chunk 03), `OPENAI_API_KEY`.
- Outputs: `data/nb04_report.csv` + printed report summary.
- Depends on chunks 00 and 03.

## Edge cases / failure modes

- No question files found → actionable error pointing to notebook 3.
- `chosen_option_index` outside 0–3 → count as incorrect, flag in report.
- Per-question API failure → record as unanswered, exclude from accuracy
  denominator, report count.
- Small-N caveat (~9 questions) stated explicitly in the report markdown.
