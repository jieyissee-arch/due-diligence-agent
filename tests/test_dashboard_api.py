"""Offline smoke tests for the FastAPI IC dashboard API."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA = REPO_ROOT / "demo_data.json"
CHROMA_DIR = REPO_ROOT / "chroma_db"


pytestmark = pytest.mark.skipif(
    not DEMO_DATA.is_file() or not CHROMA_DIR.is_dir(),
    reason="Requires demo_data.json and built chroma_db index",
)


@pytest.fixture()
def client() -> TestClient:
    from api import app

    return TestClient(app)


def test_dashboard_returns_kpis_narratives_and_charts(client: TestClient) -> None:
    response = client.get("/api/dashboard")
    assert response.status_code == 200

    payload = response.json()
    assert payload["kpi"]["total_events"] >= 1
    assert payload["kpi"]["date_range"]

    for key in ("capacity", "packaging", "launches", "market"):
        assert payload["narratives"][key]
        assert payload["charts"][key]["data"]


def test_briefing_endpoint_returns_wrapper(client: TestClient) -> None:
    response = client.get("/api/briefing")
    assert response.status_code == 200
    assert "briefing" in response.json()


def test_static_dashboard_shell_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    body = response.text.lower()
    assert "plotly" in body or "dashboard" in body or "ask me" in body


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY")
    or os.getenv("ANTHROPIC_API_KEY") == "your_anthropic_api_key_here",
    reason="Live chat requires ANTHROPIC_API_KEY",
)
def test_chat_endpoint_returns_grounded_answer(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        json={
            "message": "What packaging changes reduce plastic use?",
            "history": [],
            "briefing": None,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].strip()
    assert body["sources_html"].strip()
    assert isinstance(body["history"], list)
