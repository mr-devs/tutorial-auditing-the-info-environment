# The Horse Race — humans vs. AI on this week's news

Step 5 of the tutorial: a small website where visitors answer multiple-choice
questions built from the past week's Guardian coverage of a topic, racing
three web-searching AI models (ChatGPT, Gemini, Grok). For every question
shown to a human, the three models answer the same question **at that
moment** in the background. Answers, confidence (0–100), and start/end times
are recorded for humans and machines alike.

Everything the site needs lives in this directory. `sitekit/` is a
copied-then-edited snapshot of the tutorial's `toolkit` package (plus an xAI
provider adapter) — the site never imports from, or modifies, the tutorial
tree.

```
site/
├── app/                # FastAPI app: config, db, quiz flow, AI runner, results
├── sitekit/            # pipeline code (copied from ../toolkit, self-contained)
├── templates/          # Jinja2 pages (landing, quiz, standings)
├── static/             # style.css, quiz.js, results.js
├── refresh_content.py  # CLI: build & load a new weekly question set
├── tests/              # pytest suite (AI calls stubbed)
├── deploy/             # systemd unit, nginx config, env template
└── data/               # runtime: horserace.db + refresh scratch (git-ignored)
```

## Run locally

```bash
cd site
uv sync --group dev
export GUARDIAN_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=... XAI_API_KEY=...

# Seed a cheap development question set (~10 articles, ~30 gen + ~90 judge calls)
uv run python refresh_content.py --max-articles 10 --n-select 5

# Start the app
uv run uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000. Missing AI keys don't break the quiz — that
model just shows as unavailable.

Tests (no API keys or network needed):

```bash
uv run pytest
```

## Weekly content refresh

```bash
uv run python refresh_content.py                  # topic "artificial intelligence", 100 articles, select 10
uv run python refresh_content.py --topic "climate crisis"
uv run python refresh_content.py --no-activate    # stage without going live
```

The pipeline: Guardian collect (past 7 days) → MCQ generation (3/article) →
3 LLM judges (a question needs ≥2 "faithful" verdicts) → seeded random
selection → new versioned `question_sets` row in SQLite. Old sets, sessions,
and answers are never modified; in-flight quizzes on the old set can finish.

**Cost per full refresh:** ~2–3 Guardian calls (budget: 500/day),
~100 generation calls, ~900 judge calls.
**Cost per visitor quiz:** 10 questions × 3 models = 30 web-search LLM calls
(a fresh AI run per question presentation, by design — page refreshes do
*not* re-trigger runs). Brakes: nginx rate limit on session creation and the
optional `SITE_DAILY_SESSION_CAP`.

## Deploy on a DigitalOcean droplet

One small droplet is plenty (single uvicorn worker + SQLite).

```bash
# as root on the droplet
adduser --system --group horserace
apt install nginx
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs to /usr/local/bin/uv

git clone <repo> /opt/horserace
cd /opt/horserace/site && uv sync

cp deploy/env.example /etc/horserace.env && chmod 600 /etc/horserace.env
$EDITOR /etc/horserace.env                        # fill in the API keys

cp deploy/horserace.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now horserace

cp deploy/nginx.conf /etc/nginx/sites-available/horserace
# edit server_name, then:
ln -s /etc/nginx/sites-available/horserace /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
certbot --nginx -d your.domain.example
```

Refresh content on the droplet (manual, by design):

```bash
cd /opt/horserace/site
sudo -u horserace bash -c 'set -a; source /etc/horserace.env; set +a; uv run python refresh_content.py'
```

Nightly database backup (crontab for the `horserace` user):

```cron
15 4 * * * sqlite3 /opt/horserace/site/data/horserace.db ".backup /opt/horserace/site/data/backup-$(date +\%w).db"
```

## Notes & gotchas

- **Single worker only.** Background AI tasks and the thread pool live in
  the web process; SQLite (WAL mode) wants one writer process. The systemd
  unit pins `--workers 1`.
- **Timing is server-side.** The visible timer is cosmetic; `elapsed_ms`
  is computed from the presentation row's timestamp on the server.
- **Answer leakage:** the question payload never contains the correct
  letter or explanation — they only appear in the grading response.
- **Standings gate:** `/results` shows a friendly message until
  `SITE_MIN_SESSIONS` (default 5) humans have finished, instead of noisy
  small-sample stats.
- **Model ids drift.** `SITE_OPENAI_MODEL` / `SITE_GEMINI_MODEL` /
  `SITE_XAI_MODEL` are env-overridable, so a provider rename is a config
  edit + restart, not a redeploy. Grok uses xAI's OpenAI-compatible
  Responses API with the server-side `web_search` tool (verified 2026-07-25).
