"""
sitekit — the horse-race site's copy of the tutorial ``toolkit`` package.

Copied then edited (never imported from the tutorial tree) so the website
is fully self-contained under ``site/``. Differences from ``toolkit``:

- paths anchor to ``site/`` (see ``sitekit.config``)
- an xAI (Grok) provider adapter with server-side web search
- ``judgments.find_passing_question_ids`` — the >=2-of-3 judge pass rule
  as a reusable helper for the content-refresh pipeline
- tutorial-only modules (debate, plotting) are not included
"""
