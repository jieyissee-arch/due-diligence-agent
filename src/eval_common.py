"""
eval_common.py

Shared utilities for RAG evaluation: suite filtering, thresholds, reports.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS_PATH = REPO_ROOT / "eval" / "thresholds.json"
EVAL_REPORT_PATH = REPO_ROOT / "output" / "eval_report.json"

SUITE_CHOICES = ("all", "example", "private")


def infer_suite(entry: dict[str, Any]) -> str:
    """Infer eval suite from explicit field or query id."""
    explicit = entry.get("suite")
    if explicit in SUITE_CHOICES:
        return explicit

    entry_id = entry.get("id", "")
    if entry_id.startswith("example_"):
        return "example"
    if entry_id in {"negative_irrelevant", "example_negative_irrelevant"}:
        return "all"
    return "private"


def filter_by_suite(entries: list[dict[str, Any]], suite: str) -> list[dict[str, Any]]:
    """Keep entries for the requested suite plus shared ``all`` entries."""
    if suite == "all":
        return entries
    return [entry for entry in entries if infer_suite(entry) in {suite, "all"}]


def load_thresholds(suite: str) -> dict[str, Any]:
    if not THRESHOLDS_PATH.is_file():
        raise FileNotFoundError(f"Thresholds file not found: {THRESHOLDS_PATH}")

    with THRESHOLDS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    key = suite if suite in data else "all"
    return data[key]


def check_retrieval_thresholds(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    primary_k = int(thresholds.get("primary_k", 5))

    cat_rate = float(metrics.get(f"category_hit_rate@{primary_k}", 0.0))
    min_cat = float(thresholds.get("min_category_hit_rate", 0.0))
    if cat_rate < min_cat:
        failures.append(
            f"category_hit_rate@{primary_k}={cat_rate:.1%} < {min_cat:.1%}"
        )

    mrr = float(metrics.get(f"mrr@{primary_k}", 0.0))
    min_mrr = float(thresholds.get("min_mrr", 0.0))
    if mrr < min_mrr:
        failures.append(f"mrr@{primary_k}={mrr:.3f} < {min_mrr:.3f}")

    min_src = thresholds.get("min_source_hit_rate")
    if min_src is not None:
        src_key = f"source_hit_rate@{primary_k}"
        if src_key in metrics:
            src_rate = float(metrics[src_key])
            if src_rate < float(min_src):
                failures.append(
                    f"source_hit_rate@{primary_k}={src_rate:.1%} < {float(min_src):.1%}"
                )

    return not failures, failures


def check_generation_thresholds(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
    *,
    live: bool,
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    offline_rate = float(metrics.get("offline_pass_rate", 0.0))
    min_offline = float(thresholds.get("min_offline_pass_rate", 1.0))
    if offline_rate < min_offline:
        failures.append(
            f"offline_pass_rate={offline_rate:.1%} < {min_offline:.1%}"
        )

    if live:
        live_rate = float(metrics.get("live_pass_rate", 0.0))
        min_live = float(thresholds.get("min_live_pass_rate", 1.0))
        if live_rate < min_live:
            failures.append(f"live_pass_rate={live_rate:.1%} < {min_live:.1%}")

    return not failures, failures


def write_eval_report(report: dict[str, Any], path: Path | None = None) -> Path:
    destination = path or EVAL_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return destination


def new_report_header(suite: str, *, offline: bool) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "offline": offline,
        "stages": {},
        "passed": False,
    }
