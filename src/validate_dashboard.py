#!/usr/bin/env python3
"""
validate_dashboard.py

Smoke validation for the FastAPI IC dashboard (api.py + static UI).

Does not modify dashboard code — exercises committed REST endpoints and UI assets.

Run from repo root:
    PYTHONPATH=src python src/validate_dashboard.py --offline > validate_dashboard_log.txt 2>&1
    PYTHONPATH=src python src/validate_dashboard.py --live > validate_dashboard_live_log.txt 2>&1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA = REPO_ROOT / "demo_data.json"
CHROMA_DIR = REPO_ROOT / "chroma_db"
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


def _preflight() -> list[str]:
    errors: list[str] = []
    if not DEMO_DATA.is_file():
        errors.append(f"Missing {DEMO_DATA} (copy demo_data.example.json for quickstart).")
    if not CHROMA_DIR.is_dir() or not any(CHROMA_DIR.iterdir()):
        errors.append(
            f"Missing Chroma index under {CHROMA_DIR}. "
            "Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python3 src/build_index.py"
        )
    if not STATIC_INDEX.is_file():
        errors.append(f"Missing dashboard UI: {STATIC_INDEX}")
    return errors


def _get_client():
    from fastapi.testclient import TestClient
    from api import app

    return TestClient(app)


def validate_dashboard_payload(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    kpi = payload.get("kpi") or {}
    for key in ("total_events", "date_range", "net_signal", "net_label"):
        if key not in kpi:
            failures.append(f"dashboard.kpi missing '{key}'")

    if kpi.get("total_events", 0) < 1:
        failures.append("dashboard.kpi.total_events must be >= 1")

    narratives = payload.get("narratives") or {}
    for key in ("capacity", "packaging", "launches", "market"):
        if not narratives.get(key):
            failures.append(f"dashboard.narratives missing '{key}'")

    charts = payload.get("charts") or {}
    for key in ("capacity", "packaging", "launches", "market"):
        chart = charts.get(key) or {}
        if not chart.get("data"):
            failures.append(f"dashboard.charts.{key} missing Plotly data traces")

    return failures


def validate_offline(client) -> tuple[bool, list[str]]:
    failures: list[str] = []

    response = client.get("/api/dashboard")
    if response.status_code != 200:
        failures.append(f"GET /api/dashboard -> {response.status_code}")
    else:
        failures.extend(validate_dashboard_payload(response.json()))

    response = client.get("/api/briefing")
    if response.status_code != 200:
        failures.append(f"GET /api/briefing -> {response.status_code}")
    elif "briefing" not in response.json():
        failures.append("GET /api/briefing missing 'briefing' key")

    response = client.get("/")
    if response.status_code != 200:
        failures.append(f"GET / -> {response.status_code}")
    elif "html" not in response.headers.get("content-type", "").lower():
        failures.append("GET / did not return HTML")
    elif "Due Diligence" not in response.text and "dashboard" not in response.text.lower():
        failures.append("GET / HTML does not look like the IC dashboard shell")

    return not failures, failures


def validate_live(client) -> tuple[bool, list[str]]:
    failures: list[str] = []

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_anthropic_api_key_here":
        failures.append("ANTHROPIC_API_KEY not set — cannot run live chat validation.")
        return False, failures

    response = client.post(
        "/api/chat",
        json={
            "message": "What packaging sustainability changes are mentioned?",
            "history": [],
            "briefing": None,
        },
    )
    if response.status_code != 200:
        failures.append(f"POST /api/chat -> {response.status_code}: {response.text[:200]}")
        return False, failures

    body = response.json()
    for key in ("answer", "history", "sources_html"):
        if key not in body:
            failures.append(f"POST /api/chat missing '{key}'")

    answer = body.get("answer", "")
    if not answer.strip():
        failures.append("POST /api/chat returned empty answer")

    if "[Passage" not in answer and "passage" not in answer.lower():
        failures.append("POST /api/chat answer missing passage citation")

    sources = body.get("sources_html", "")
    if not sources.strip():
        failures.append("POST /api/chat returned empty sources_html")

    return not failures, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-validate the IC dashboard API.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Validate dashboard, briefing, and static UI only (default).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also validate POST /api/chat with Claude (requires API key).",
    )
    args = parser.parse_args(argv)

    offline = args.offline or not args.live

    preflight_errors = _preflight()
    if preflight_errors:
        for err in preflight_errors:
            print(f"ERROR: {err}")
        return 1

    client = _get_client()
    all_failures: list[str] = []

    if offline:
        print("MODE: offline")
        passed, failures = validate_offline(client)
        all_failures.extend(failures)
        print(f"offline dashboard smoke: {'PASSED' if passed else 'FAILED'}")
        for failure in failures:
            print(f"  - {failure}")

    if args.live:
        print("MODE: live chat")
        passed, failures = validate_live(client)
        all_failures.extend(failures)
        print(f"live chat smoke: {'PASSED' if passed else 'FAILED'}")
        for failure in failures:
            print(f"  - {failure}")

    if all_failures:
        print()
        print(f"FAILED — {len(all_failures)} issue(s)")
        return 1

    print()
    print("PASSED — dashboard smoke validation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
