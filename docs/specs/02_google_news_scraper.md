# Spec 02 — Notebook 2: Google News scraper

## Learning objective / role in the narrative

Static benchmarks go stale the day they are published; auditing *real-time*
model knowledge requires acquiring information the model cannot have seen in
training. Participants scrape today's Google News Top Stories with Playwright
and persist article text for downstream question generation.

## Implementation plan

1. Setup cell (chdir, imports from `toolkit.config` /
   `toolkit.playwright_helper`, ensure `ARTICLES_DIR` exists).
2. Async cell (top-level `await` — works natively in Jupyter/VS Code):
   open `https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFZxYUdjU0FtVnVHZ0pWVXlnQVAB`
   headless, wait for content, collect `a[href^="./read"]` hrefs (current
   Google News markup; verified live). Keep only anchors with headline-like
   text (≥20 chars) to skip thumbnail/timestamp duplicates.
3. Convert relative hrefs to absolute (`https://news.google.com/read/...`);
   dedupe preserving order; keep headline text alongside each URL.
4. Loop over the top 3 links. Story URLs are JS redirect shells, so scrape
   in two hops: `resolve_publisher_url` (goto, then `wait_for_url` until the
   location leaves news.google.com) → `await scrape_article_text(publisher_url)`.
   A URL still on `news.google.com` or on `google.com/sorry` (rate-limit
   captcha) counts as blocked → fallback. Write raw text to
   `data/articles/article_1.txt` … `article_3.txt`.
5. Quality gate: if a scrape failed (`"Failed scraping URL"` prefix) or yields
   <500 characters, substitute a bundled fallback sample article (clearly
   labeled) so notebooks 3–4 always have usable input. Report which articles
   are live vs. fallback.
6. Preview cell: first ~600 chars of each saved article.

## Pydantic schemas

None — this chunk is pure acquisition; outputs are raw `.txt` files.

## Inputs / outputs / dependencies

- Inputs: live network access; Chromium installed via
  `uv run playwright install chromium`.
- Outputs: `data/articles/article_{1..3}.txt`.
- Depends on chunk 00 (`toolkit.playwright_helper`). Chunk 03 consumes
  outputs.

## Edge cases / failure modes

- Google News consent/interstitial pages or markup changes → selector returns
  nothing; notebook detects an empty link list and falls back to samples with
  a clear warning instead of crashing.
- Publisher blocks headless traffic or `networkidle` never fires (common on
  live-blog pages) → helper's 15 s timeout returns an error string; quality
  gate substitutes fallback.
- Google rate-limit captcha (`google.com/sorry`) under bursty automated
  access → detected by URL, treated as blocked.
- Boilerplate-heavy extraction (nav text, cookie banners) → acknowledged in
  markdown; stripping common chrome tags is "good enough" for MCQ generation.
- Sync Playwright API inside Jupyter → avoided entirely (async + `await`).
