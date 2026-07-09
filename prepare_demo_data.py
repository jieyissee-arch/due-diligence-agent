#!/usr/bin/env python3
from __future__ import annotations

"""
Build a small demo dataset from extracted news events.

DEPRECATED: prefer sample_events.py for scaling from the private corpus, or
copy demo_data.example.json for a public quickstart without real research data.

Reads ../news_scrape_mar26/events_extracted.jsonl, samples roughly 15–20
records per category, and writes demo_data.json in the repo root.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_FILE = REPO_ROOT.parent / "news_scrape_mar26" / "events_extracted.jsonl"
OUTPUT_FILE = REPO_ROOT / "demo_data.json"

CATEGORIES = [
    "CLOSURES",
    "EXPANSIONS",
    "PRODUCT_LAUNCHES",
    "NEW_BUILDS",
    "PACKAGING",
]

TARGET_PER_CATEGORY = 18
RANDOM_SEED = 42


def _parse_date(published_at: str) -> str:
    """Return YYYY-MM-DD from an ISO timestamp, or the original string."""
    if not published_at:
        return published_at
    return published_at[:10] if len(published_at) >= 10 else published_at


def load_records_by_category(source_path: Path) -> dict[str, list[dict]]:
    """Flatten JSONL articles into per-category demo records."""
    by_category: dict[str, list[dict]] = defaultdict(list)

    with source_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            article = json.loads(line)
            source = article.get("meta_site") or article.get("url", "")
            date = _parse_date(article.get("publishedAt", ""))

            for event in article.get("events", []):
                category = event.get("topic")
                if category not in CATEGORIES:
                    continue

                by_category[category].append(
                    {
                        "category": category,
                        "source": source,
                        "date": date,
                        "text": event.get("description", "").strip(),
                    }
                )

    return by_category


def sample_demo_records(
    by_category: dict[str, list[dict]],
    target_per_category: int = TARGET_PER_CATEGORY,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Sample up to target_per_category records from each category."""
    rng = random.Random(seed)
    sampled: list[dict] = []

    for category in CATEGORIES:
        pool = by_category.get(category, [])
        if not pool:
            continue

        count = min(target_per_category, len(pool))
        sampled.extend(rng.sample(pool, count))

    return sampled


def main() -> None:
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    by_category = load_records_by_category(SOURCE_FILE)
    demo_records = sample_demo_records(by_category)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(demo_records, f, indent=2, ensure_ascii=False)
        f.write("\n")

    counts = {cat: sum(1 for r in demo_records if r["category"] == cat) for cat in CATEGORIES}
    print(f"Wrote {len(demo_records)} records to {OUTPUT_FILE}")
    for cat in CATEGORIES:
        print(f"  {cat}: {counts[cat]}")


if __name__ == "__main__":
    main()
