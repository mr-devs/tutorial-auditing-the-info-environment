"""Guardian Content API collection.

Production version of the workflow taught in
``notebooks/01_guardian_news_collection.ipynb``: search the Guardian's
free Content API (https://open-platform.theguardian.com/) with pagination,
polite rate limiting, a daily call budget, exponential-backoff retries, and
incremental JSONL persistence with resume.

Typical use::

    from toolkit.guardian import collect

    summary = collect(
        "climate policy",
        output_fp="data/articles/climate.jsonl",
        max_articles=100,
        from_date="2026-06-01",
    )

Free-tier limits (developer key): 1 call/second, 500 calls/day.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from toolkit import config

logger = logging.getLogger(__name__)

GUARDIAN_SEARCH_URL = "https://content.guardianapis.com/search"
DEFAULT_FIELDS = "bodyText,headline,byline,trailText,wordcount"
DEFAULT_PAGE_SIZE = 50  # API maximum
MIN_INTERVAL_SECONDS = 1.05  # buffer under the 1 request/second limit
DEFAULT_DAILY_BUDGET = 500  # free-tier daily call cap
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT = 30


class GuardianAPIError(RuntimeError):
    """A non-retryable API failure (bad key, malformed parameters, ...)."""


class BudgetExhausted(RuntimeError):
    """Raised when the next call would exceed the daily call budget."""


def _is_retryable(exc: BaseException) -> bool:
    """Transient failures only: network errors and 429/5xx responses."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in RETRYABLE_STATUS
    return False


class RateLimiter:
    """Enforce a minimum interval between calls using a monotonic clock."""

    def __init__(self, min_interval: float = MIN_INTERVAL_SECONDS):
        self.min_interval = min_interval
        self._last_call: Optional[float] = None

    def wait(self) -> None:
        """Sleep just long enough to keep calls >= min_interval apart."""
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            remaining = self.min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


@dataclass
class GuardianClient:
    """Rate-limited, budget-aware, retrying Guardian Content API client.

    Parameters
    ----------
    api_key : str
        Guardian API key. Falls back to the ``GUARDIAN_API_KEY`` environment
        variable (via ``toolkit.config``). The shared public key ``"test"``
        works for light experimentation.
    daily_budget : int
        Maximum number of API calls this client will make (free tier allows
        500/day). ``BudgetExhausted`` is raised once it would be exceeded.
    """

    api_key: str = ""
    daily_budget: int = DEFAULT_DAILY_BUDGET
    calls_made: int = 0
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self):
        self.api_key = self.api_key or config.GUARDIAN_API_KEY
        if not self.api_key:
            raise GuardianAPIError(
                "No Guardian API key. Export GUARDIAN_API_KEY (register free at "
                "https://open-platform.theguardian.com/access/) or use api_key='test'."
            )

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _get(self, params: dict) -> dict:
        """One rate-limited, budget-counted GET. Returns the 'response' dict.

        Retries transient failures (network errors, 429/5xx) with exponential
        backoff; raises ``GuardianAPIError`` immediately on permanent 4xx and
        ``BudgetExhausted`` when the daily budget is spent.
        """
        if self.calls_made >= self.daily_budget:
            raise BudgetExhausted(
                f"Daily call budget of {self.daily_budget} exhausted."
            )
        self.rate_limiter.wait()
        self.calls_made += 1
        r = self.session.get(
            GUARDIAN_SEARCH_URL,
            params={**params, "api-key": self.api_key},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code in RETRYABLE_STATUS:
            r.raise_for_status()  # HTTPError -> tenacity retries
        if not r.ok:
            try:
                message = r.json()["response"]["message"]
            except Exception:
                message = r.text[:200]
            raise GuardianAPIError(
                f"Guardian API returned {r.status_code}: {message} "
                "(check your API key and query parameters)"
            )
        return r.json()["response"]

    def fetch_page(
        self,
        query: Optional[str] = None,
        *,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        section: Optional[str] = None,
        tag: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        order_by: str = "newest",
        show_fields: str = DEFAULT_FIELDS,
    ) -> dict:
        """Fetch one page of search results.

        Returns the raw API 'response' dict with keys such as ``total``,
        ``pages``, ``currentPage``, and ``results``. All filters are
        optional; at least one of query/section/tag is recommended.
        Dates are ``YYYY-MM-DD`` strings.
        """
        params = {
            "page": page,
            "page-size": page_size,
            "order-by": order_by,
            "show-fields": show_fields,
        }
        optional = {
            "q": query,
            "section": section,
            "tag": tag,
            "from-date": from_date,
            "to-date": to_date,
        }
        params.update({k: v for k, v in optional.items() if v is not None})
        return self._get(params)

    def search(
        self,
        query: Optional[str] = None,
        *,
        max_articles: Optional[int] = None,
        max_pages: Optional[int] = None,
        **filters,
    ) -> Iterator[dict]:
        """Yield flattened article records across pages.

        Stops at ``max_articles``, ``max_pages``, the last available page,
        or when the daily budget runs out (``BudgetExhausted`` propagates).
        Extra keyword arguments are passed to :meth:`fetch_page`
        (section, tag, from_date, to_date, order_by, page_size, ...).
        """
        yielded = 0
        page = 1
        while True:
            response = self.fetch_page(query, page=page, **filters)
            self.last_total = response["total"]
            total_pages = response["pages"]
            logger.info(
                "Fetched page %d/%d (%d results, %d total matches)",
                page,
                total_pages,
                len(response["results"]),
                response["total"],
            )
            for article in response["results"]:
                if max_articles is not None and yielded >= max_articles:
                    return
                yield to_record(article)
                yielded += 1
            if page >= total_pages:
                return
            if max_pages is not None and page >= max_pages:
                return
            if max_articles is not None and yielded >= max_articles:
                return
            page += 1


def _camel_to_snake(name: str) -> str:
    """Convert an API field name like ``bodyText`` to ``body_text``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def to_record(article: dict) -> dict:
    """Flatten one raw API result into a tidy, analysis-ready dict.

    The stable core keys (id, url, published, section) come from the article
    envelope; every requested ``show-fields`` field is merged in with its
    name converted to snake_case (``bodyText`` -> ``body_text``), so records
    automatically match whatever fields the search asked for.
    """
    record = {
        "id": article.get("id"),
        "url": article.get("webUrl"),
        "published": article.get("webPublicationDate"),
        "section": article.get("sectionName"),
    }
    for key, value in article.get("fields", {}).items():
        record[_camel_to_snake(key)] = value
    return record


def append_records(records, output_fp) -> int:
    """Append records to a JSONL file (creating parent dirs). Returns count."""
    output_fp = Path(output_fp)
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_fp, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_existing_ids(output_fp) -> set:
    """Return the set of article ids already saved in a JSONL file.

    Missing file -> empty set. Malformed lines are skipped with a warning.
    """
    output_fp = Path(output_fp)
    ids = set()
    if not output_fp.exists():
        return ids
    with open(output_fp, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("id"):
                    ids.add(record["id"])
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", lineno, output_fp)
    return ids


def collect(
    query: Optional[str] = None,
    *,
    output_fp,
    client: Optional[GuardianClient] = None,
    resume: bool = True,
    max_articles: Optional[int] = None,
    max_pages: Optional[int] = None,
    **filters,
) -> dict:
    """Search the Guardian and save results incrementally to JSONL.

    The high-level entry point used by the CLI script (and the notebook's
    closing one-liner). Records are appended one page at a time, so an
    interrupted run keeps everything fetched so far; with ``resume=True``
    (the default) re-running the same command skips articles whose ids are
    already in ``output_fp``.

    Returns a summary dict::

        {"new": int, "skipped": int, "calls_used": int,
         "total_available": int, "budget_exhausted": bool}
    """
    client = client or GuardianClient()
    existing = load_existing_ids(output_fp) if resume else set()
    if existing:
        logger.info("Resuming: %d article ids already in %s", len(existing), output_fp)

    new = skipped = 0
    budget_exhausted = False
    calls_before = client.calls_made
    buffer = []

    def flush():
        nonlocal buffer
        if buffer:
            append_records(buffer, output_fp)
            buffer = []

    try:
        for record in client.search(
            query, max_articles=max_articles, max_pages=max_pages, **filters
        ):
            if record["id"] in existing:
                skipped += 1
                continue
            existing.add(record["id"])
            buffer.append(record)
            new += 1
            if len(buffer) >= DEFAULT_PAGE_SIZE:
                flush()
    except BudgetExhausted:
        budget_exhausted = True
        logger.warning("Daily call budget exhausted; saving partial results.")
    finally:
        flush()

    return {
        "new": new,
        "skipped": skipped,
        "calls_used": client.calls_made - calls_before,
        "total_available": getattr(client, "last_total", 0),
        "budget_exhausted": budget_exhausted,
    }
