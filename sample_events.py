#!/usr/bin/env python3
"""
sample_events.py — Stage 6 sampling

Read extracted events from the private news_scrape_mar26 repo (sibling folder)
and write a stratified sample into due-diligence-agent/demo_data.json.

Skips any events already present in the current demo_data.json so the public
repo demo set is not duplicated when scaling up.

Run from due-diligence-agent repo root:
    python sample_events.py > sample_events_log.txt 2>&1

Environment overrides:
    NEWS_SCRAPE_DIR  — path to news_scrape_mar26 (default: ../news_scrape_mar26)
    TARGET_TOTAL     — number of events to sample (default: 1000)
    RANDOM_SEED      — sampling seed (default: 42)
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
NEWS_SCRAPE_DIR = Path(
    os.getenv("NEWS_SCRAPE_DIR", str(REPO_ROOT.parent / "news_scrape_mar26"))
)
SOURCE_FILE = NEWS_SCRAPE_DIR / "events_extracted.jsonl"
OUTPUT_FILE = REPO_ROOT / "demo_data.json"
EXISTING_FILE = REPO_ROOT / "demo_data.json"

TARGET_TOTAL = int(os.getenv("TARGET_TOTAL", "1000"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

CATEGORIES = [
    "CLOSURES",
    "EXPANSIONS",
    "PRODUCT_LAUNCHES",
    "NEW_BUILDS",
    "PACKAGING",
]


def _parse_date(published_at: str) -> str:
    if not published_at:
        return published_at
    return published_at[:10] if len(published_at) >= 10 else published_at


def record_fingerprint(record: dict) -> tuple[str, str, str, str]:
    """Stable identity for deduplication across demo and source corpora."""
    return (
        str(record.get("category", "")).strip().lower(),
        str(record.get("source", "")).strip().lower(),
        str(record.get("date", "")).strip(),
        str(record.get("text", "")).strip().lower(),
    )


def load_existing_fingerprints(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.is_file():
        return set()

    with path.open(encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array.")

    return {record_fingerprint(record) for record in records}


def flatten_events(source_path: Path) -> list[dict]:
    """Flatten events_extracted.jsonl into demo_data-compatible records."""
    records: list[dict] = []

    with source_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            article = json.loads(line)
            source = article.get("meta_site") or article.get("url", "")
            date = _parse_date(article.get("publishedAt", ""))
            url = article.get("url", "")

            for event in article.get("events", []):
                category = event.get("topic")
                if category not in CATEGORIES:
                    continue

                text = event.get("description", "").strip()
                if not text:
                    continue

                records.append(
                    {
                        "category": category,
                        "source": source,
                        "date": date,
                        "text": text,
                        "url": url,
                    }
                )

    return records


def stratified_sample(
    pool: list[dict],
    target_total: int,
    seed: int,
) -> list[dict]:
    """Sample up to target_total records, stratified by category."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for record in pool:
        by_category[record["category"]].append(record)

    if not by_category:
        return []

    rng = random.Random(seed)
    available_total = sum(len(v) for v in by_category.values())
    target = min(target_total, available_total)

    # Proportional allocation with at least one per non-empty category when possible
    categories = [c for c in CATEGORIES if by_category.get(c)]
    weights = {c: len(by_category[c]) for c in categories}
    weight_sum = sum(weights.values())

    allocations: dict[str, int] = {}
    remaining = target
    for i, category in enumerate(categories):
        if i == len(categories) - 1:
            allocations[category] = min(remaining, len(by_category[category]))
        else:
            share = max(1, round(target * weights[category] / weight_sum))
            share = min(share, len(by_category[category]), remaining)
            allocations[category] = share
            remaining -= share

    sampled: list[dict] = []
    for category, count in allocations.items():
        if count <= 0:
            continue
        sampled.extend(rng.sample(by_category[category], count))

    rng.shuffle(sampled)
    return sampled


def to_demo_schema(records: list[dict]) -> list[dict]:
    """Strip internal fields (url) for demo_data.json output schema."""
    return [
        {
            "category": r["category"],
            "source": r["source"],
            "date": r["date"],
            "text": r["text"],
        }
        for r in records
    ]


def main() -> int:
    if not SOURCE_FILE.is_file():
        print(f"ERROR: Source file not found: {SOURCE_FILE}")
        print("Clone news_scrape_mar26 alongside due-diligence-agent, or set NEWS_SCRAPE_DIR.")
        return 1

    existing = load_existing_fingerprints(EXISTING_FILE)
    print(f"Source file        : {SOURCE_FILE}")
    print(f"Existing fingerprints to exclude: {len(existing)}")

    all_records = flatten_events(SOURCE_FILE)
    print(f"Total source events: {len(all_records)}")

    pool = [r for r in all_records if record_fingerprint(r) not in existing]
    print(f"Pool after exclusion : {len(pool)}")

    if not pool:
        print("ERROR: No new events available after excluding existing demo_data.")
        return 1

    sampled = stratified_sample(pool, TARGET_TOTAL, RANDOM_SEED)
    output_records = to_demo_schema(sampled)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output_records, f, indent=2, ensure_ascii=False)
        f.write("\n")

    counts = {cat: sum(1 for r in output_records if r["category"] == cat) for cat in CATEGORIES}
    print(f"Wrote {len(output_records)} records to {OUTPUT_FILE}")
    for cat in CATEGORIES:
        print(f"  {cat}: {counts[cat]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
