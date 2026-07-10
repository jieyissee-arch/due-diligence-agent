#!/usr/bin/env python3
"""
eval_generation.py

Evaluate RAG generation grounding against eval/labeled_answers.json.

Offline metrics (no API key required):
  - retrieval_category_hit
  - retrieval_term_hit
  - prompt_built

Live metrics (requires ANTHROPIC_API_KEY):
  - citation_present
  - citation_valid
  - sources_used_section
  - grounded_terms
  - abstention_ok

Run from repo root:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python src/eval_generation.py --suite example --offline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from eval_common import (
    SUITE_CHOICES,
    check_generation_thresholds,
    filter_by_suite,
    load_thresholds,
)
from eval_scoring import (
    citation_present,
    citation_valid,
    grounded_terms_in_cited_passages,
    has_sources_used_section,
    looks_like_abstention,
    max_chunk_similarity,
    rate,
    retrieval_category_hit,
    retrieval_term_hit,
)
from generate import build_prompt, generate_answer
from retrieval import retrieve

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELED_ANSWERS = REPO_ROOT / "eval" / "labeled_answers.json"


def evaluate_entry(
    entry: dict[str, Any],
    *,
    live: bool,
) -> dict[str, Any]:
    """Score one labeled answer entry. Returns per-metric booleans and notes."""
    query = entry["query"]
    top_k = int(entry.get("top_k", 3))
    expected_categories = entry.get("expected_categories", [])
    required_terms = entry.get("required_terms_any", [])
    require_citations = bool(entry.get("require_citations", True))
    expect_abstention = bool(entry.get("expect_abstention", False))

    chunks = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, chunks)

    scores: dict[str, bool | None] = {
        "retrieval_category_hit": retrieval_category_hit(chunks, expected_categories),
        "retrieval_term_hit": retrieval_term_hit(chunks, required_terms),
        "prompt_built": bool(prompt.strip()),
        "citation_present": None,
        "citation_valid": None,
        "sources_used_section": None,
        "grounded_terms": None,
        "abstention_ok": None,
    }

    answer = ""
    if live:
        result = generate_answer(query, top_k=top_k, chunks=chunks)
        answer = result.get("answer", "")

        if expect_abstention:
            scores["abstention_ok"] = looks_like_abstention(answer)
        else:
            scores["citation_present"] = (
                citation_present(answer) if require_citations else True
            )
            scores["citation_valid"] = (
                citation_valid(answer, len(chunks)) if require_citations else True
            )
            scores["sources_used_section"] = (
                has_sources_used_section(answer) if require_citations else True
            )
            scores["grounded_terms"] = grounded_terms_in_cited_passages(
                answer, chunks, required_terms
            )
            scores["abstention_ok"] = True

    offline_pass = all(
        scores[key]
        for key in ("retrieval_category_hit", "retrieval_term_hit", "prompt_built")
        if scores[key] is not None
    )

    if live:
        if expect_abstention:
            live_pass = bool(scores["abstention_ok"])
        else:
            live_keys = (
                "citation_present",
                "citation_valid",
                "sources_used_section",
                "grounded_terms",
                "abstention_ok",
            )
            live_pass = all(scores[key] for key in live_keys if scores[key] is not None)
    else:
        live_pass = None

    return {
        "id": entry.get("id", "unknown"),
        "query": query,
        "chunks_retrieved": len(chunks),
        "max_similarity": max_chunk_similarity(chunks),
        "scores": scores,
        "offline_pass": offline_pass,
        "live_pass": live_pass,
        "answer_preview": answer[:200].replace("\n", " ") if answer else "",
    }


def run_eval(
    suite: str = "all",
    *,
    offline: bool = False,
    live_required: bool = False,
) -> dict[str, Any]:
    if not LABELED_ANSWERS.is_file():
        return {"passed": False, "error": f"{LABELED_ANSWERS} not found.", "metrics": {}}

    with LABELED_ANSWERS.open(encoding="utf-8") as f:
        all_entries = json.load(f)

    entries = filter_by_suite(all_entries, suite)
    if not entries:
        return {
            "passed": False,
            "error": f"No labeled answers for suite '{suite}'.",
            "metrics": {},
            "suite": suite,
            "query_count": 0,
        }

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    has_key = bool(api_key) and api_key != "your_anthropic_api_key_here"
    live = (not offline) and has_key

    if live_required and not live:
        return {
            "passed": False,
            "error": "Live generation eval requested but ANTHROPIC_API_KEY is not set.",
            "metrics": {},
            "suite": suite,
            "query_count": len(entries),
        }

    results = [evaluate_entry(entry, live=live) for entry in entries]

    metrics: dict[str, float] = {
        "retrieval_category_hit_rate": rate(
            [r["scores"]["retrieval_category_hit"] for r in results]
        ),
        "retrieval_term_hit_rate": rate(
            [r["scores"]["retrieval_term_hit"] for r in results]
        ),
        "offline_pass_rate": rate([r["offline_pass"] for r in results]),
    }

    if live:
        live_results = [r for r in results if r["live_pass"] is not None]
        metrics.update(
            {
                "citation_present_rate": rate(
                    [r["scores"]["citation_present"] for r in live_results]
                ),
                "citation_valid_rate": rate(
                    [r["scores"]["citation_valid"] for r in live_results]
                ),
                "sources_used_rate": rate(
                    [r["scores"]["sources_used_section"] for r in live_results]
                ),
                "grounded_terms_rate": rate(
                    [r["scores"]["grounded_terms"] for r in live_results]
                ),
                "abstention_ok_rate": rate(
                    [r["scores"]["abstention_ok"] for r in live_results]
                ),
                "live_pass_rate": rate([r["live_pass"] for r in live_results]),
            }
        )

    thresholds = load_thresholds(suite)["generation"]
    passed, threshold_failures = check_generation_thresholds(
        metrics, thresholds, live=live
    )

    failed_offline = [r["id"] for r in results if not r["offline_pass"]]
    failed_live = [r["id"] for r in results if r["live_pass"] is False]

    if failed_offline or failed_live:
        passed = False

    return {
        "passed": passed,
        "suite": suite,
        "mode": "offline" if not live else "live",
        "query_count": len(entries),
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": threshold_failures,
        "failed_offline": failed_offline,
        "failed_live": failed_live,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG generation grounding.")
    parser.add_argument(
        "--suite",
        choices=SUITE_CHOICES,
        default="all",
        help="Eval suite: example (public quickstart), private, or all.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip Claude calls; run retrieval and prompt checks only.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Require live Claude generation (fail if API key missing).",
    )
    args = parser.parse_args(argv)

    result = run_eval(
        suite=args.suite,
        offline=args.offline,
        live_required=args.live,
    )

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return 1

    print(f"Suite           : {result['suite']}")
    print(f"Labeled answers : {result['query_count']}")
    print(f"Mode            : {result['mode']}")
    print()

    for outcome in result.get("results", []):
        scores = outcome["scores"]
        print(f"[{outcome['id']}] {outcome['query']}")
        print(
            f"  retrieval_category_hit={scores['retrieval_category_hit']} "
            f"retrieval_term_hit={scores['retrieval_term_hit']} "
            f"max_sim={outcome['max_similarity']:.3f}"
        )
        if result["mode"] == "live":
            print(
                f"  citation_present={scores['citation_present']} "
                f"citation_valid={scores['citation_valid']} "
                f"sources_used={scores['sources_used_section']} "
                f"grounded_terms={scores['grounded_terms']} "
                f"abstention_ok={scores['abstention_ok']}"
            )
            if outcome["answer_preview"]:
                print(f"  answer_preview: {outcome['answer_preview']}...")
        print(f"  offline_pass={outcome['offline_pass']} live_pass={outcome['live_pass']}")
        print()

    print("=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)
    metrics = result["metrics"]
    print(
        f"retrieval_category_hit_rate={metrics['retrieval_category_hit_rate']:.1%} "
        f"retrieval_term_hit_rate={metrics['retrieval_term_hit_rate']:.1%} "
        f"offline_pass_rate={metrics['offline_pass_rate']:.1%}"
    )
    if result["mode"] == "live":
        print(
            f"citation_present_rate={metrics['citation_present_rate']:.1%} "
            f"citation_valid_rate={metrics['citation_valid_rate']:.1%} "
            f"sources_used_rate={metrics['sources_used_rate']:.1%} "
            f"grounded_terms_rate={metrics['grounded_terms_rate']:.1%} "
            f"abstention_ok_rate={metrics['abstention_ok_rate']:.1%}"
        )
        print(f"live_pass_rate={metrics['live_pass_rate']:.1%}")

    if result["failed_offline"]:
        print()
        print(
            f"OFFLINE FAILURES ({len(result['failed_offline'])}): "
            f"{', '.join(result['failed_offline'])}"
        )
    if result["failed_live"]:
        print(
            f"LIVE FAILURES ({len(result['failed_live'])}): "
            f"{', '.join(result['failed_live'])}"
        )
    if result["failures"]:
        print("THRESHOLD FAILURES:")
        for failure in result["failures"]:
            print(f"  - {failure}")

    if result["passed"]:
        print()
        print(f"PASSED — all {result['mode']} generation eval checks satisfied.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
