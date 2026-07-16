# Step 2 — Generate multiple-choice questions from news articles

**Status: not yet planned in detail.**

## General requirements (from tutorial design)

- Input: `data/articles/*.jsonl` from Step 1.
- Use an LLM to convert each article into one or more multiple-choice
  questions (question, N options, correct answer, source article id).
- Use structured outputs (Pydantic/JSON schema) so responses are machine-
  readable and validated.
- Output: `data/questions/*.jsonl`, one MCQ per line.
- Artifacts: teaching notebook `notebooks/02_*.ipynb`, CLI script
  `scripts/02_*.py`, toolkit module(s).
- The notebook teaches prompting for question generation, structured
  responses, and validation; the script adds batching, resume, and retries.
