"""
File-IO helpers for fact-checking runners: claim loading, resume from
existing JSONL, output filename convention, JSONL append, and web-source
parquet loading.
"""

import glob
import json
import os
import shutil

import pandas as pd

# Probe-variant filename substrings to exclude from cleaning pipelines.
# These files use non-standard prompts and are analyzed separately
# (see code/data_collection/claude_websearch_probe.py).
PROBE_VARIANTS = ("__websearch_hard", "__websearch_soft")


def load_claims(path, limit=None):
    """Read the parquet of claims; optional `limit` truncates for smoke tests."""
    df = pd.read_parquet(path)
    return df.head(limit) if limit is not None else df


def get_completed_links(output_fp):
    """Return the set of `factcheck_analysis_link` values already in `output_fp`."""
    completed = set()
    try:
        with open(output_fp, "r") as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["factcheck_analysis_link"])
                except (json.JSONDecodeError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return completed


def filter_remaining(factchecks, completed_links):
    remaining = factchecks[~factchecks["factcheck_analysis_link"].isin(completed_links)]
    return remaining.reset_index(drop=True)


def build_output_filename(model):
    return f"fc_results__model_{model}.jsonl"


def append_jsonl(file_obj, record):
    """Append one record and flush so resume works on Ctrl-C.

    `default=str` coerces non-JSON-native values (e.g. datetime objects that
    some provider SDK responses embed in `model_response`) to their string
    form so a single odd field never aborts a whole run.
    """
    file_obj.write(json.dumps(record, default=str) + "\n")
    file_obj.flush()


def seed_local_working_file(local_fp, drive_fp):
    """
    Prepare the local working file so the end-of-run publish holds the full
    accumulated set of results (prior published history + this run).

    If `local_fp` already exists it is kept, and any rows from `drive_fp` whose
    `factcheck_analysis_link` is not already present are appended — never
    overwriting the local file. A lingering `local_fp` means a hard kill
    (SIGKILL/power loss) skipped the publish in run_fact_check.py's `finally`,
    so it holds rows not yet on Drive; overwriting it would drop those rows even
    though their links are already in the resume set (so they would never be
    recollected). When no local file exists, `drive_fp` is copied to seed it.
    No-op when neither file exists.

    Parameters
    ----------
    local_fp : str
        Path to the local working JSONL (streamed/flushed during the run).
    drive_fp : str
        Path to the published JSONL (the Drive copy / resume source).
    """
    if os.path.exists(local_fp):
        if not os.path.exists(drive_fp):
            return
        local_links = get_completed_links(local_fp)
        with open(drive_fp, "r") as src, open(local_fp, "a") as dst:
            for line in src:
                try:
                    link = json.loads(line)["factcheck_analysis_link"]
                except (json.JSONDecodeError, KeyError):
                    continue
                if link not in local_links:
                    dst.write(line if line.endswith("\n") else line + "\n")
    elif os.path.exists(drive_fp):
        shutil.copy2(drive_fp, local_fp)


def load_web_source_parquets(data_dir, glob_pattern):
    """
    Load all cleaned web-source parquets matching `glob_pattern` in `data_dir`.

    Parameters
    ----------
    data_dir : str or path-like
        Directory containing the parquet files.
    glob_pattern : str
        Filename glob, e.g. ``"cleaned__model_*.parquet"``.

    Returns
    -------
    pandas.DataFrame
        Concatenation of all matching parquets, reset index.

    Raises
    ------
    FileNotFoundError
        If no files match.
    """
    files = sorted(glob.glob(os.path.join(data_dir, glob_pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching '{glob_pattern}' in {data_dir}")
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)
