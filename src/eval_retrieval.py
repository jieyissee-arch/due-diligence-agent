#!/usr/bin/env python3
"""
eval_retrieval.py

Evaluate retrieval quality against labeled queries in eval/labeled_queries.json.

Metrics per query at k in eval_at_k (default [1, 3, 5]):
  - category_hit@k
  - source_hit@k
  - MRR (by first category match rank)

Run from repo root:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python src/eval_retrieval.py > eval_retrieval_log.txt 2>&1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from retrieval import DEFAULT_TOP_K, retrieve

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELED_QUERIES = REPO_ROOT / "eval" / "labeled_queries.json"
PRIMARY_K = DEFAULT_TOP_K


def _category_hit(results: list[dict[str, Any]], expected: list[str], k: int) -> bool:
    if not expected:
        return False
    expected_set = {c.upper() for c in expected}
    for row in results[:k]:
        if row.get("category", "").upper() in expected_set:
            return True
    return False


def _source_hit(results: list[dict[str, Any]], expected: list[str], k: int) -> bool:
    if not expected:
        return False
    expected_set = {s.lower() for s in expected}
    for row in results[:k]:
        if row.get("source", "").lower() in expected_set:
            return True
    return False


def _mrr(results: list[dict[str, Any]], expected_categories: list[str], k: int) -> float:
    if not expected_categories:
        return 0.0
    expected_set = {c.upper() for c in expected_categories}
    for rank, row in enumerate(results[:k], 1):
        if row.get("category", "").upper() in expected_set:
            return 1.0 / rank
    return 0.0


def _avg_similarity(results: list[dict[str, Any]], k: int) -> float:
    top = results[:k]
    if not top:
        return 0.0
    return sum(float(r.get("similarity", 0.0)) for r in top) / len(top)


def main() -> int:
    if not LABELED_QUERIES.is_file():
        print(f"ERROR: {LABELED_QUERIES} not found.")
        return 1

    with LABELED_QUERIES.open(encoding="utf-8") as f:
        queries = json.load(f)

    if not queries:
        print("ERROR: labeled_queries.json is empty.")
        return 1

    max_k = max(
        max(entry.get("eval_at_k", [PRIMARY_K])) for entry in queries
    )
    max_k = max(max_k, PRIMARY_K)

    print(f"Labeled queries : {len(queries)}")
    print(f"Max eval k      : {max_k}")
    print(f"Primary k       : {PRIMARY_K}")
    print()

    k_values = sorted({k for entry in queries for k in entry.get("eval_at_k", [1, 3, 5])})

    aggregates: dict[int, dict[str, list[float | bool]]] = {
        k: {"category_hit": [], "source_hit": [], "mrr": [], "avg_similarity": []}
        for k in k_values
    }

    for entry in queries:
        query_id = entry.get("id", "unknown")
        query = entry["query"]
        expected_categories = entry.get("expected_categories", [])
        expected_sources = entry.get("expected_sources_any", [])
        eval_at_k = entry.get("eval_at_k", [1, 3, 5])

        results = retrieve(query, top_k=max_k)

        print(f"[{query_id}] {query}")
        for k in eval_at_k:
            cat_hit = _category_hit(results, expected_categories, k)
            src_hit = _source_hit(results, expected_sources, k) if expected_sources else None
            mrr = _mrr(results, expected_categories, k) if expected_categories else 0.0
            avg_sim = _avg_similarity(results, k)

            aggregates[k]["category_hit"].append(cat_hit)
            if expected_sources:
                aggregates[k]["source_hit"].append(src_hit)
            if expected_categories:
                aggregates[k]["mrr"].append(mrr)
            aggregates[k]["avg_similarity"].append(avg_sim)

            src_part = f" source_hit@{k}={src_hit}" if expected_sources else ""
            print(
                f"  category_hit@{k}={cat_hit}{src_part} "
                f"mrr@{k}={mrr:.3f} avg_sim@{k}={avg_sim:.3f}"
            )
        print()

    print("=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)
    for k in k_values:
        cat_rate = sum(aggregates[k]["category_hit"]) / len(aggregates[k]["category_hit"])
        mrr_vals = aggregates[k]["mrr"]
        mrr_avg = sum(mrr_vals) / len(mrr_vals) if mrr_vals else 0.0
        sim_avg = sum(aggregates[k]["avg_similarity"]) / len(aggregates[k]["avg_similarity"])
        line = f"k={k}: category_hit_rate={cat_rate:.1%} mrr={mrr_avg:.3f} avg_similarity={sim_avg:.3f}"
        if aggregates[k]["source_hit"]:
            src_rate = sum(aggregates[k]["source_hit"]) / len(aggregates[k]["source_hit"])
            line += f" source_hit_rate={src_rate:.1%}"
        print(line)

    # Primary pass line
    primary_cat = aggregates.get(PRIMARY_K, {}).get("category_hit", [])
    if primary_cat:
        primary_rate = sum(primary_cat) / len(primary_cat)
        print()
        print(f"PRIMARY (k={PRIMARY_K}) category_hit_rate: {primary_rate:.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
