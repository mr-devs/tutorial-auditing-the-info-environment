# Step 4 — Testing LLMs on the MCQs, three ways

**Status: not yet planned in detail.**

## General requirements (from tutorial design)

- Input: vetted questions from Step 3.
- Evaluate LLM ability to answer the MCQs under three conditions:
  1. **Closed-book** — weights/training data only, no web search.
  2. **Web search** — same models with a search tool enabled.
  3. **Multi-agent debate** — a debate framework (see the prototype at
     `scripts/mad-agents-fact-checking.py`).
- Teaching focus: input parameters, structured responses, and batch input
  (batch APIs / concurrent requests).
- Output: `data/predictions/*.jsonl` per method, plus accuracy summaries.
- Artifacts: teaching notebook `notebooks/04_*.ipynb`, CLI script(s)
  `scripts/04_*.py`, toolkit module(s).
