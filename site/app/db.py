"""
SQLite helpers for the horse-race site.

Connections are short-lived (open, use, close) so they are safe from any
thread — FastAPI request handlers and the AI runner's worker threads alike.
WAL mode plus a generous busy timeout keeps the single-file database happy
under the site's modest concurrency (one uvicorn worker).
"""

import sqlite3
from pathlib import Path

from app import config

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path=None) -> sqlite3.Connection:
    """Open a configured SQLite connection.

    Parameters
    ----------
    db_path : str or Path, optional
        Database file to open. Defaults to ``config.DB_PATH``.

    Returns
    -------
    sqlite3.Connection
        Connection with WAL journaling, a 5 s busy timeout, foreign keys
        enforced, and ``sqlite3.Row`` rows.
    """
    path = Path(db_path) if db_path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path=None) -> None:
    """Create all tables and indexes if they do not exist.

    Parameters
    ----------
    db_path : str or Path, optional
        Database file to initialize. Defaults to ``config.DB_PATH``.
    """
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text())


def fetch_one(sql: str, params=(), db_path=None):
    """Run a query and return the first row (or None)."""
    with connect(db_path) as conn:
        return conn.execute(sql, params).fetchone()


def fetch_all(sql: str, params=(), db_path=None):
    """Run a query and return all rows."""
    with connect(db_path) as conn:
        return conn.execute(sql, params).fetchall()


def execute(sql: str, params=(), db_path=None) -> int:
    """Run a write statement in its own transaction.

    Returns
    -------
    int
        ``rowcount`` from the executed statement.
    """
    with connect(db_path) as conn:
        cur = conn.execute(sql, params)
        return cur.rowcount
