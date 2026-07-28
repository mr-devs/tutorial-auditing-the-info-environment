"""Utility functions for LLM search auditing."""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import requests
import tldextract
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from toolkit import config

# Gemini search results link sources through this redirect endpoint; a
# model can only have such a URL if the search tool actually returned it.
GROUNDING_REDIRECT_MARKER = "vertexaisearch.cloud.google.com/grounding-api-redirect"

REQUEST_TIMEOUT_SECONDS = 30

# requests.Session is not thread-safe, so the parallel resolver keeps one
# session per worker thread (each with its own keep-alive connection pool).
_thread_local = threading.local()


def _get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = _thread_local.session = requests.Session()
    return session


@retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(
        (requests.ConnectionError, requests.Timeout, requests.HTTPError)
    ),
    reraise=True,
)
def _follow_redirect(url, timeout):
    # First hop only: Google's redirect endpoint answers HEAD with a 302
    # whose Location header IS the cited URL, so the destination site is
    # never contacted (no bot-detection/paywall exposure) and each link
    # costs one round trip to a single keep-alive endpoint. raise_for_status
    # turns transient 5xx/429 from Google into retryable HTTPErrors; a
    # non-redirect success (no Location header) resolves to None.
    response = _get_session().head(url, allow_redirects=False, timeout=timeout)
    response.raise_for_status()
    return response.headers.get("Location")


def resolve_redirect_url(url, timeout=REQUEST_TIMEOUT_SECONDS) -> Optional[str]:
    """Resolve one Gemini grounding redirect to the URL it points at.

    Parameters
    ----------
    url : str
        A ``GROUNDING_REDIRECT_MARKER`` link as found in Gemini grounding
        chunks or self-reported ``citations``.
    timeout : float, default REQUEST_TIMEOUT_SECONDS
        Per-request timeout in seconds.

    Returns
    -------
    str or None
        The redirect's target URL (the 302 ``Location`` header), or None
        on hard failure (bad input, no redirect, or a request error that
        survives three retries).
    """
    if not url or not isinstance(url, str):
        return None
    try:
        return _follow_redirect(url, timeout)
    except requests.RequestException:
        return None


def resolve_redirect_urls(
    urls,
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_workers=8,
) -> dict:
    """Resolve many citation URLs at once, in parallel.

    Only ``GROUNDING_REDIRECT_MARKER`` links cost an HTTP round trip
    (deduplicated, spread across ``max_workers`` threads); any other URL
    is already its own destination and maps to itself untouched — so a
    raw ``citations`` list can be passed wholesale.

    Parameters
    ----------
    urls : iterable of str
        Citation URLs, mixed redirect and plain; duplicates are fine.
    timeout : float, default REQUEST_TIMEOUT_SECONDS
        Per-request timeout in seconds.
    max_workers : int, default 8
        Thread count for the redirect lookups.

    Returns
    -------
    dict
        Mapping of each unique input URL to its resolved URL (None where
        ``resolve_redirect_url`` hard-failed).
    """
    urls = list(dict.fromkeys(urls))  # dedupe, keep order
    redirects = [
        u for u in urls if isinstance(u, str) and GROUNDING_REDIRECT_MARKER in u
    ]
    resolved = {u: u for u in urls}
    if redirects:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(redirects))) as pool:
            targets = pool.map(lambda u: resolve_redirect_url(u, timeout), redirects)
        resolved.update(zip(redirects, targets))
    return resolved


def resolve_path(path) -> str:
    """Resolve a user-supplied path against the repository root.

    Absolute paths are returned unchanged; relative paths are interpreted
    relative to the repo root (NOT the current working directory), so
    ``data/articles/x.jsonl`` means the same thing no matter which directory
    a script is launched from.

    Parameters
    ----------
    path : str or Path
        The path to resolve.

    Returns
    -------
    str
        The resolved absolute path.
    """
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str(Path(config.REPO_ROOT) / path)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: Optional[bool] = None,
    append_mode: bool = False,
) -> logging.Logger:
    """Set up logging configuration with explicit output destination requirements.

    Parameters
    ----------
    log_level : str, default "INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str, optional
        Path to log file. If provided, logs will be written to file.
    console_output : bool, optional
        Whether to output logs to console/stdout. Must be explicitly set.
    append_mode : bool, default False
        Whether to append to existing log file (True) or overwrite (False).

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Raises
    ------
    ValueError
        If neither console_output nor log_file destination is clearly
        specified.
    """
    # Require explicit specification of output destination
    if console_output is None and log_file is None:
        raise ValueError(
            "Must specify logging destination: either set console_output=True/False "
            "or provide log_file path, or both"
        )

    # Default console_output to False if only log_file is provided
    if console_output is None and log_file is not None:
        console_output = False

    # Require at least one output destination
    if not console_output and log_file is None:
        raise ValueError(
            "Must specify at least one logging destination: "
            "either console_output=True or provide log_file path"
        )

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file is not None:
        # Ensure the directory exists
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)

        # Use append mode if requested
        file_mode = "a" if append_mode else "w"
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def extract_domain(url: str) -> Optional[str]:
    """Extract base domain from URL using tldextract (without subdomains).

    Parameters
    ----------
    url : str
        The URL to extract domain from.

    Returns
    -------
    str or None
        The extracted domain (e.g., 'example.com') or None if extraction
        fails.
    """
    try:
        extracted = tldextract.extract(url)
        # Combine domain and suffix (e.g., 'example' + 'co.uk' = 'example.co.uk')
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
        elif extracted.domain:
            return extracted.domain.lower()
        return None
    except Exception:
        return None


def load_jsonl(filepath: str) -> List[str]:
    """Load jsonl file contents into a list.

    Parameters
    ----------
    filepath : str
        Path to the jsonl file.

    Returns
    -------
    list
        Parsed JSON objects, one per line.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line]
