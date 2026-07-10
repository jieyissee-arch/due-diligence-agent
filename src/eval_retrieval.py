#!/usr/bin/env python3
"""
eval_retrieval.py

Evaluate retrieval quality against labeled queries in eval/labeled_queries.json.

Metrics per query at k in eval_at_k (default [1, 3, 5]):
  - category_hit@k
  - source_hit@k
  - MRR (by first category match rank)

Run from repo root:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python src/eval_retrieval.py --suite example
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval_common import (
    SUITE_CHOICES,
    check_retrieval_thresholds,
    filter_by_suite,
    load_thresholds,
)
from eval_scoring import rate
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


def run_eval(suite: str = "all") -> dict[str, Any]:
    if not LABELED_QUERIES.is_file():
        return {"passed": False, "error": f"{LABELED_QUERIES} not found.", "metrics": {}}

    with LABELED_QUERIES.open(encoding="utf-8") as f:
        all_queries = json.load(f)

    queries = filter_by_suite(all_queries, suite)
    if not queries:
        return {
            "passed": False,
            "error": f"No labeled queries for suite '{suite}'.",
            "metrics": {},
            "suite": suite,
            "query_count": 0,
        }

    max_k = max(max(entry.get("eval_at_k", [PRIMARY_K])) for entry in queries)
    max_k = max(max_k, PRIMARY_K)

    k_values = sorted({k for entry in queries for k in entry.get("eval_at_k", [1, 3, 5])})
    aggregates: dict[int, dict[str, list[float | bool]]] = {
        k: {"category_hit": [], "source_hit": [], "mrr": [], "avg_similarity": []}
        for k in k_values
    }

    for entry in queries:
        query = entry["query"]
        expected_categories = entry.get("expected_categories", [])
        expected_sources = entry.get("expected_sources_any", [])
        eval_at_k = entry.get("eval_at_k", [1, 3, 5])

        results = retrieve(query, top_k=max_k)

        for k in eval_at_k:
            if expected_categories:
                aggregates[k]["category_hit"].append(
                    _category_hit(results, expected_categories, k)
                )
                aggregates[k]["mrr"].append(_mrr(results, expected_categories, k))
            if expected_sources:
                aggregates[k]["source_hit"].append(
                    _source_hit(results, expected_sources, k)
                )
            aggregates[k]["avg_similarity"].append(_avg_similarity(results, k))

    metrics: dict[str, float] = {}
    for k in k_values:
        cat_vals = aggregates[k]["category_hit"]
        if cat_vals:
            metrics[f"category_hit_rate@{k}"] = rate(cat_vals)
        mrr_vals = aggregates[k]["mrr"]
        if mrr_vals:
            metrics[f"mrr@{k}"] = sum(mrr_vals) / len(mrr_vals)
        sim_vals = aggregates[k]["avg_similarity"]
        if sim_vals:
            metrics[f"avg_similarity@{k}"] = sum(sim_vals) / len(sim_vals)
        src_vals = aggregates[k]["source_hit"]
        if src_vals:
            metrics[f"source_hit_rate@{k}"] = rate(src_vals)

    thresholds = load_thresholds(suite)["retrieval"]
    primary_k = int(thresholds.get("primary_k", PRIMARY_K))
    passed, failures = check_retrieval_thresholds(metrics, thresholds)

    return {
        "passed": passed,
        "suite": suite,
        "query_count": len(queries),
        "primary_k": primary_k,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument(
        "--suite",
        choices=SUITE_CHOICES,
        default="all",
        help="Eval suite: example (public quickstart), private, or all.",
    )
    args = parser.parse_args(argv)

    result = run_eval(suite=args.suite)
    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return 1

    print(f"Suite           : {result['suite']}")
    print(f"Labeled queries : {result['query_count']}")
    print(f"Primary k       : {result['primary_k']}")
    print()

    metrics = result["metrics"]
    k_values = sorted(
        {int(key.split("@")[1]) for key in metrics if "@" in key and key.split("@")[1].isdigit()}
    )
    for k in k_values:
        line = (
            f"k={k}: category_hit_rate={metrics.get(f'category_hit_rate@{k}', 0.0):.1%} "
            f"mrr={metrics.get(f'mrr@{k}', 0.0):.3f} "
            f"avg_similarity={metrics.get(f'avg_similarity@{k}', 0.0):.3f}"
        )
        if f"source_hit_rate@{k}" in metrics:
            line += f" source_hit_rate={metrics[f'source_hit_rate@{k}']:.1%}"
        print(line)

    print()
    if result["passed"]:
        print("PASSED — retrieval metrics meet suite thresholds.")
        return 0

    print("FAILED — retrieval metrics below suite thresholds:")
    for failure in result["failures"]:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
