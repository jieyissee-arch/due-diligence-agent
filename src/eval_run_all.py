#!/usr/bin/env python3
"""
eval_run_all.py

Run the full RAG evaluation suite:
  1. eval_chunking.py
  2. eval_retrieval.py
  3. eval_generation.py

Writes output/eval_report.json with aggregate metrics and pass/fail status.

Run from repo root:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python src/eval_run_all.py --suite example --offline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from eval_common import SUITE_CHOICES, new_report_header, write_eval_report
from eval_chunking import run_eval as run_chunking_eval
from eval_generation import run_eval as run_generation_eval
from eval_retrieval import run_eval as run_retrieval_eval

REPO_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = REPO_ROOT / "chroma_db"
DEMO_DATA = REPO_ROOT / "demo_data.json"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")


def _banner(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _preflight() -> int:
    if not DEMO_DATA.is_file():
        print(f"ERROR: {DEMO_DATA} not found.")
        print("Copy demo_data.example.json to demo_data.json for a quickstart.")
        return 1

    if not CHROMA_DIR.is_dir() or not any(CHROMA_DIR.iterdir()):
        print(f"ERROR: Chroma index not found under {CHROMA_DIR}.")
        print(
            "Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
            "PYTHONPATH=src python3 src/build_index.py"
        )
        return 1

    return 0


def _corpus_record_count() -> int:
    with DEMO_DATA.open(encoding="utf-8") as f:
        records = json.load(f)
    return len(records) if isinstance(records, list) else 0


def run_all(
    *,
    suite: str = "all",
    offline: bool = False,
    skip_chunking: bool = False,
    skip_retrieval: bool = False,
    skip_generation: bool = False,
) -> dict[str, Any]:
    report = new_report_header(suite, offline=offline)
    report["corpus_records"] = _corpus_record_count()
    report["embedding_provider"] = EMBEDDING_PROVIDER
    report["embedding_model"] = EMBEDDING_MODEL

    stages: dict[str, Any] = {}

    if not skip_chunking:
        _banner("STAGE 1 — CHUNKING EVAL")
        chunking = run_chunking_eval()
        stages["chunking"] = chunking
        if chunking.get("error"):
            print(f"ERROR: {chunking['error']}")
        elif chunking["passed"]:
            print("PASSED — chunking invariants satisfied.")
        else:
            print(f"FAILED — {len(chunking['failures'])} chunking issue(s).")

    if not skip_retrieval:
        _banner("STAGE 2 — RETRIEVAL EVAL")
        retrieval = run_retrieval_eval(suite=suite)
        stages["retrieval"] = retrieval
        if retrieval.get("error"):
            print(f"ERROR: {retrieval['error']}")
        elif retrieval["passed"]:
            print("PASSED — retrieval metrics meet suite thresholds.")
        else:
            print("FAILED — retrieval below thresholds:")
            for failure in retrieval.get("failures", []):
                print(f"  - {failure}")

    if not skip_generation:
        _banner("STAGE 3 — GENERATION EVAL")
        generation = run_generation_eval(suite=suite, offline=offline)
        stages["generation"] = generation
        if generation.get("error"):
            print(f"ERROR: {generation['error']}")
        elif generation["passed"]:
            print(f"PASSED — generation eval ({generation['mode']}).")
        else:
            print(f"FAILED — generation eval ({generation.get('mode', 'unknown')}).")
            for failure in generation.get("failures", []):
                print(f"  - {failure}")

    report["stages"] = {
        name: {
            "passed": stage.get("passed", False),
            "metrics": stage.get("metrics", {}),
            "failures": stage.get("failures", []),
            "query_count": stage.get("query_count"),
            "record_count": stage.get("record_count"),
            "chunk_count": stage.get("chunk_count"),
            "mode": stage.get("mode"),
        }
        for name, stage in stages.items()
    }
    report["passed"] = all(stage.get("passed", False) for stage in stages.values())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all RAG evaluation scripts.")
    parser.add_argument(
        "--suite",
        choices=SUITE_CHOICES,
        default="all",
        help="Eval suite: example (public quickstart), private, or all.",
    )
    parser.add_argument("--offline", action="store_true", help="Skip live Claude generation.")
    parser.add_argument("--skip-chunking", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args(argv)

    if _preflight() != 0:
        return 1

    started = time.time()
    report = run_all(
        suite=args.suite,
        offline=args.offline,
        skip_chunking=args.skip_chunking,
        skip_retrieval=args.skip_retrieval,
        skip_generation=args.skip_generation,
    )
    report["elapsed_seconds"] = round(time.time() - started, 1)

    report_path = write_eval_report(report)
    print()
    _banner("EVAL RUN ALL — SUMMARY")
    print(f"Suite  : {report['suite']}")
    print(f"Elapsed: {report['elapsed_seconds']}s")
    print(f"Report : {report_path}")
    print()

    for name, stage in report["stages"].items():
        status = "PASSED" if stage["passed"] else "FAILED"
        print(f"  {name:12s} {status}")

    if report["passed"]:
        print()
        print("ALL EVAL STAGES PASSED")
        return 0

    failed = [name for name, stage in report["stages"].items() if not stage["passed"]]
    print()
    print(f"FAILED stages: {', '.join(failed)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
