"""
Prompt-text assembly + tolerant JSON parsing.
"""

import json


def build_user_text(originator, statement):
    """Build the user-facing prompt body shared across all providers."""
    return f"STATEMENT ORIGINATOR: {originator}\nCLAIM: {statement}\n\n"


def extract_label_justification(text):
    """
    Best-effort extract `(label, justification)` from a free-text response.
    Tolerant of code fences, surrounding prose, single-element list wrappers,
    and multiple concatenated JSON objects. Opus sometimes emits draft objects
    before its final answer (e.g. `{"label": "True", "justification": ""}` run
    directly into the real object); the last object carrying a "label" is the
    final answer. Returns `(None, None)` when no parseable object is found.
    """
    if text is None:
        return None, None

    s = text.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip().rstrip("`").strip()

    # Fast path: the whole string is a single JSON value.
    try:
        d = json.loads(s)
        if isinstance(d, list) and d:
            d = d[0]
        if isinstance(d, dict):
            return d.get("label"), d.get("justification")
    except json.JSONDecodeError:
        pass

    # Walk every top-level JSON object and keep the last dict carrying a
    # "label". raw_decode parses one object at a time, so this handles
    # concatenated objects and surrounding prose that a single json.loads
    # cannot. Non-JSON characters between objects are skipped.
    decoder = json.JSONDecoder()
    i, n, last = 0, len(s), None
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(s, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and "label" in obj:
            last = obj
        i = end

    if last is not None:
        return last.get("label"), last.get("justification")
    return None, None
