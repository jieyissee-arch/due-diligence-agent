#!/usr/bin/env python3
"""
eval_chunking.py

Validate chunking invariants on demo_data.json.

Run from repo root:
    PYTHONPATH=src python src/eval_chunking.py > eval_chunking_log.txt 2>&1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from chunking import MAX_WORDS, MIN_WORDS, _word_count, chunk_records

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA = REPO_ROOT / "demo_data.json"


def run_eval(data_path: Path | None = None) -> dict[str, Any]:
    path = (data_path or DEMO_DATA).expanduser().resolve()
    if not path.is_file():
        return {
            "passed": False,
            "error": f"Demo data not found: {path}",
            "record_count": 0,
            "chunk_count": 0,
            "failures": [f"Demo data not found: {path}"],
        }

    with path.open(encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        return {
            "passed": False,
            "error": f"{path} must contain a non-empty JSON array.",
            "record_count": 0,
            "chunk_count": 0,
            "failures": [f"{path} must contain a non-empty JSON array."],
        }

    chunks = chunk_records(records)
    failures: list[str] = []
    multi_chunk_records = 0

    record_chunk_counts: dict[int, int] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        parts = chunk_id.rsplit("_", 2)
        if len(parts) >= 3:
            try:
                record_index = int(parts[-2])
                record_chunk_counts[record_index] = record_chunk_counts.get(record_index, 0) + 1
            except ValueError:
                pass

    for record_index, _record in enumerate(records):
        count = record_chunk_counts.get(record_index, 0)
        if count == 0:
            failures.append(f"Record {record_index} produced no chunks.")
        elif count > 1:
            multi_chunk_records += 1

    for i, chunk in enumerate(chunks):
        for field in ("chunk_id", "text", "category", "source", "date"):
            if field not in chunk or chunk[field] in (None, ""):
                failures.append(f"Chunk {i} missing or empty field: {field}")

        words = _word_count(chunk["text"])
        if words > MAX_WORDS:
            failures.append(
                f"Chunk {chunk['chunk_id']} exceeds MAX_WORDS ({words} > {MAX_WORDS})."
            )

    multi_chunk_rate = multi_chunk_records / len(records) if records else 0.0

    return {
        "passed": not failures,
        "record_count": len(records),
        "chunk_count": len(chunks),
        "multi_chunk_records": multi_chunk_records,
        "multi_chunk_rate": multi_chunk_rate,
        "word_bounds": {"min": MIN_WORDS, "max": MAX_WORDS},
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    _ = argv
    result = run_eval()

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return 1

    print(f"Input records     : {result['record_count']}")
    print(f"Output chunks     : {result['chunk_count']}")
    print(
        f"Chunk word bounds : {result['word_bounds']['min']}-"
        f"{result['word_bounds']['max']} (target for long text)"
    )
    print(
        f"Multi-chunk records : {result['multi_chunk_records']} "
        f"({result['multi_chunk_rate']:.1%})"
    )

    failures = result["failures"]
    if failures:
        print(f"\nFAILED — {len(failures)} issue(s):")
        for issue in failures[:20]:
            print(f"  - {issue}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1

    print("\nPASSED — all chunking invariants satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
