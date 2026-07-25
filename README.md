# Auditing the Information Environment with LLMs

Hands-on workshop tutorial. We build a pipeline that scrapes fresh news,
turns it into a multiple-choice quiz, vets the quiz with LLM judges, and
tests LLM "contestants" on it three different ways.

Each step ships a **demo notebook** (the step-by-step walkthrough) and a
**research-ready CLI script**, backed by the shared local **`toolkit`**
package. Steps 3–4 also ship analysis notebooks under `notebooks/analysis/`
that build figures from the script outputs.

1. **News collection** — scrape Guardian articles (full body text) into JSONL:
   - `notebooks/demos/01_guardian_news_collection.ipynb`
   - `scripts/01_collect_guardian_news.py`
2. **Question generation** — LLMs generate multiple-choice questions from the articles:
   - `notebooks/demos/02_question_generation.ipynb`
   - `scripts/02_generate_questions.py`
3. **LLM judge** — judge models vet each question for faithfulness to its article; a seeded random set of passers advances:
   - `notebooks/demos/03_llm_judge.ipynb` (analysis: `notebooks/analysis/03_judgment_analysis.ipynb`)
   - `scripts/03-1_generate_judgments.py` → `scripts/03-2_combine_judgments.py` → `scripts/03-3_select_questions.py`
4. **Answering methods** — LLMs answer the quiz: closed-book vs. web search vs. multi-agent debate:
   - `notebooks/demos/04_answering_methods.ipynb` (analysis: `notebooks/analysis/04_answer_analysis.ipynb`)
   - `scripts/04-1_generate_answers.py` + `scripts/04-2_generate_debate_answers.py` → `scripts/04-3_combine_answers.py`

## Setup

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A free [Guardian API key](https://open-platform.theguardian.com/access/) (Step 1)
- An [OpenAI API key](https://platform.openai.com/api-keys) (Steps 2–4)
- A [Gemini API key](https://aistudio.google.com/apikey) (Steps 2–4)

### Create the environment

From the repository root:

```bash
uv sync
```

This creates `.venv/`, installs every dependency, and installs the local
`toolkit/` package editable.

### Export your API keys

```bash
export GUARDIAN_API_KEY="..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
```

Keys are never stored in the repo. `SML_`-prefixed variants
(e.g. `SML_OPENAI_API_KEY`) are checked first and take precedence.
