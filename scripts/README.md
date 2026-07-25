# Running the scripts

All scripts run from the repository root with `uv run`, write to `data/`
(git-ignored), and support `--help` for the full option surface.

## Step 1: collect Guardian news

```bash
uv run python scripts/01_collect_guardian_news.py \
    --query "climate policy" "misinformation" \
    --from-date 2026-07-01 \
    --max-articles 200 \
    --output data/articles/guardian_articles.jsonl
```

The script paces itself under the free tier's 1 call/second, stops cleanly at
the 500 calls/day budget, retries transient failures with exponential
backoff, saves incrementally (crash-safe JSONL), and **resumes**: re-running
the same command skips articles already saved.

## Step 2: generate questions

Once per model you want to compare (the provider is inferred from the
model id):

```bash
uv run python scripts/02_generate_questions.py --model gpt-5.6-terra --parallel
uv run python scripts/02_generate_questions.py --model gemini-3.1-flash-lite --parallel
```

Each run writes `data/questions/questions_<model>.jsonl` (one question per
line, schema-validated). `--parallel` fans articles out to a thread pool;
resume works exactly like Step 1.

## Step 3: judge and select questions

Three judge models vet every question for faithfulness — is the marked
correct answer supported by the article, with no equally defensible
alternative? Questions marked faithful by at least 2 of 3 judges are
eligible, and a seeded random 100 advance:

```bash
# 1. Judge (once per judge model)
for M in gpt-5.6-luna gpt-5.5-2026-04-23 gpt-5.4-mini-2026-03-17; do
  uv run python scripts/03-1_generate_judgments.py \
      --questions data/questions/questions_gpt-5.6-terra.jsonl \
      --articles data/articles/guardian_articles.jsonl \
      --model $M --parallel
done

# 2. Merge the per-model judgment files into one tidy CSV
uv run python scripts/03-2_combine_judgments.py \
    --input-dir data/judgments --glob 'judgments_*.jsonl'

# 3. Seeded random selection of 100 passing questions
uv run python scripts/03-3_select_questions.py \
    --input data/judgments/judgments_combined.csv \
    --questions data/questions/questions_gpt-5.6-terra.jsonl
```

## Step 4: answer the quiz

The contestant models answer the selected questions under three
conditions — closed book (weights only), web search (provider search tool
on), and a 3-agent, 3-round **debate** on the openai-agents SDK (majority
vote; 9 LLM calls per question; GPT models only). One run = one method ×
one model:

```bash
# 1. Closed book + web search (once per method x model; 1,200 calls)
for METHOD in closed_book web_search; do
  for M in gpt-5.4-mini-2026-03-17 gpt-5.5-2026-04-23 gpt-5.6-luna \
           gpt-5.6-terra gemini-3.1-flash-lite gemini-3.5-flash; do
    uv run python scripts/04-1_generate_answers.py \
        --model $M --method $METHOD --parallel
  done
done

# 2. Debate (once per GPT model; 3,600 calls)
for M in gpt-5.4-mini-2026-03-17 gpt-5.5-2026-04-23 \
         gpt-5.6-luna gpt-5.6-terra; do
  uv run python scripts/04-2_generate_debate_answers.py --model $M --parallel
done

# 3. Merge the per-run answer files into one tidy, graded CSV
uv run python scripts/04-3_combine_answers.py \
    --input-dir data/answers --glob 'answers_*.jsonl'
```

Contestants see only the question and options — never the article. Each
answer is graded on the spot (`is_correct`), web-search runs record whether
the model actually searched, and debate runs keep the full transcript in
the JSONL.
