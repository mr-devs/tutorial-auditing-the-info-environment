# Step 3 — LLM-as-judge analysis of the MCQs

**Status: not yet planned in detail.**

## General requirements (from tutorial design)

- Input: `data/questions/*.jsonl` from Step 2 (plus source articles from
  Step 1 for grounding).
- A judge LLM scores each MCQ on three dimensions:
  1. **Quality & answerability** — well-formed, unambiguous, exactly one
     correct answer supported by the article.
  2. **Faithfulness to source** — the correct answer is grounded in the
     article text (no hallucinated facts).
  3. **Difficulty / guessability** — distractors are plausible; the question
     is not answerable without the article (too easy / leaked from training
     data).
- The judge emits a pass/fail verdict (and per-dimension scores) used to
  filter which questions advance to Steps 4–5.
- Output: `data/questions_vetted/*.jsonl`.
- Artifacts: teaching notebook `notebooks/03_*.ipynb`, CLI script
  `scripts/03_*.py`, toolkit module(s).
