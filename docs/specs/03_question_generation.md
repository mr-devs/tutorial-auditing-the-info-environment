# Spec 03 — Notebook 3: Question generation

## Learning objective / role in the narrative

Show how to automate synthetic test generation: turning arbitrary source text
into strict-JSON, machine-gradable evaluation items (multiple-choice
questions) without human annotators — the key to auditing models at scale and
on demand.

## Implementation plan

1. Setup cell; ensure `QUESTIONS_DIR` exists; load `article_{1..3}.txt`.
2. Define `MCQuestion` and a wrapper `MCQuestionSet` (`questions:
   list[MCQuestion]`) so one structured call returns several items per
   article.
3. `generate_questions(article_text, n=3)` via
   `toolkit.providers.openai_provider.run_parsed(DEFAULT_LLM, ...,
   use_web_search=False, text_format=MCQuestionSet)` — the article is the
   only source we want, so no tools. Prompt requires: fact-based, exactly 4
   options, single unambiguous answer, `evidence_excerpt` copied verbatim
   from the article, distractors plausible but wrong.
4. Truncate article text (~8k chars) to control tokens; note this in markdown.
5. Validation gate per question: exactly 4 options, `correct_index` in 0–3,
   evidence excerpt actually appears (case-normalized) in the article text —
   drop and report questions that fail.
6. Serialize per-article to `data/questions/question_1.json` … `_3.json`
   (article filename + list of validated questions), via
   `model_dump()`/`json.dump`.
7. Preview a formatted question.

## Pydantic schemas

```python
class MCQuestion(BaseModel):
    question: str
    options: list[str]        # exactly 4 (validated post-hoc)
    correct_index: int        # 0–3
    evidence_excerpt: str     # verbatim supporting snippet

class MCQuestionSet(BaseModel):
    questions: list[MCQuestion]
```

## Inputs / outputs / dependencies

- Inputs: `data/articles/article_{1..3}.txt` (chunk 02), `OPENAI_API_KEY`.
- Outputs: `data/questions/question_{1..3}.json`.
- Depends on chunks 00 and 02. Chunk 04 consumes outputs.

## Edge cases / failure modes

- Missing/empty article files → skip with a clear message pointing back to
  notebook 2.
- Model hallucinates an evidence excerpt → verbatim-containment check drops
  the question (this *is* the lesson: validate machine-generated tests).
- Fewer than `n` valid questions survive → proceed with what remains; the
  audit in notebook 4 handles variable counts.
- API failure on one article → try/except, continue with the others.
