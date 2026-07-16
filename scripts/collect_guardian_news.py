"""Guardian Content API collection: linear pacing, daily budget, incremental save."""

import os
import json
import math
import time

import requests

GUARDIAN_KEY = os.environ.get("GUARDIAN_API_KEY", "")
BASE = "https://content.guardianapis.com/search"
MIN_INTERVAL = 1.05  # seconds between calls, buffer under the 1/sec limit
PAGE_SIZE = 50


def fetch_page(query, page=1):
    params = {
        "q": query,
        "api-key": GUARDIAN_KEY,
        "show-fields": "bodyText,headline,byline",
        "page-size": PAGE_SIZE,
        "page": page,
        "order-by": "newest",
    }
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["response"]


def to_record(article):
    fields = article.get("fields", {})
    return {
        "id": article.get("id"),
        "url": article.get("webUrl"),
        "published": article.get("webPublicationDate"),
        "section": article.get("sectionName"),
        "headline": fields.get("headline"),
        "byline": fields.get("byline"),
        "body_text": fields.get("bodyText"),
    }


def append_records(records, output_fp):
    with open(output_fp, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def search_all(query, max_pages=10, call_budget=None,
               save_immediately=False, output_fp=None):
    if save_immediately and output_fp is None:
        raise ValueError("output_fp is required when save_immediately is True")

    first = fetch_page(query, 1)
    records = [to_record(a) for a in first["results"]]
    if save_immediately:
        append_records(records, output_fp)

    pages = min(first["pages"], max_pages)
    if call_budget is not None:
        pages = min(pages, call_budget)

    for page in range(2, pages + 1):
        time.sleep(MIN_INTERVAL)
        pg = fetch_page(query, page)
        page_records = [to_record(a) for a in pg["results"]]
        records.extend(page_records)
        if save_immediately:
            append_records(page_records, output_fp)

    return records


def run_batch(queries, per_query_pages=5, daily_budget=500,
              save_immediately=False, output_fp=None):
    if save_immediately and output_fp is None:
        raise ValueError("output_fp is required when save_immediately is True")

    calls_used = 0
    all_results = {}
    for q in queries:
        remaining = daily_budget - calls_used
        if remaining <= 0:
            print(f"Budget exhausted before '{q}'")
            break

        budget = min(per_query_pages, remaining)
        hits = search_all(
            q,
            max_pages=budget,
            call_budget=budget,
            save_immediately=save_immediately,
            output_fp=output_fp,
        )
        all_results[q] = hits

        calls_made = max(1, math.ceil(len(hits) / PAGE_SIZE))
        calls_used += calls_made
        print(f"'{q}': {len(hits)} articles, {calls_used}/{daily_budget} calls used")

    return all_results


if __name__ == "__main__":
    if not GUARDIAN_KEY:
        raise SystemExit("Set GUARDIAN_API_KEY in the environment first.")

    QUERIES = ["climate policy", "misinformation", "artificial intelligence"]

    results = run_batch(
        QUERIES,
        per_query_pages=5,
        daily_budget=500,
        save_immediately=True,
        output_fp="guardian_articles.jsonl",
    )

    total = sum(len(v) for v in results.values())
    print(f"Done: {total} articles across {len(results)} queries.")
