"""Utility functions shared by the site's pipeline code."""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from sitekit import config


def resolve_path(path) -> str:
    """Resolve a user-supplied path against the site root.

    Absolute paths are returned unchanged; relative paths are interpreted
    relative to the site root (NOT the current working directory), so
    ``data/horserace.db`` means the same thing no matter which directory
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
    return str(Path(config.SITE_ROOT) / path)


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
