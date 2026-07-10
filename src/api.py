"""
api.py

FastAPI backend for the PE due diligence dashboard.
Replaces Gradio (app.py) with clean REST endpoints + static SPA.

Run from repo root:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src ./ragdemo/bin/python src/api.py
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_dda")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analysis import _CAT_LABELS, run_analysis
from chat import ChatError, chat_turn
from insights import (
    InsightsError,
    get_or_load_briefing,
    refresh_briefing,
)

load_dotenv()

# ── Proxy bypass (same logic as generate.py) ─────────────────────────────────

def _configure_proxy_bypass() -> None:
    for env_name in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(env_name, "")
        parts = [p.strip() for p in existing.split(",") if p.strip()]
        for host in ("localhost", "127.0.0.1", "::1"):
            if host not in parts:
                parts.append(host)
        os.environ[env_name] = ",".join(parts)

_configure_proxy_bypass()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Due Diligence Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Design colours ────────────────────────────────────────────────────────────

_C = {
    "PACKAGING":        "#004A7C",
    "CLOSURES":         "#A93226",
    "EXPANSIONS":       "#1A7A4A",
    "NEW_BUILDS":       "#2E86C1",
    "PRODUCT_LAUNCHES": "#6C3483",
    "trend":            "#CA6F1E",
    "bg":               "#FFFFFF",
    "panel":            "#F4F6F9",
    "grid":             "#E0E4EC",
    "text_dark":        "#12263A",
}

_STACK_ORDER = ["CLOSURES", "NEW_BUILDS", "EXPANSIONS", "PRODUCT_LAUNCHES", "PACKAGING"]

# ── Narrative helpers (ported from app.py) ────────────────────────────────────

def _sign(v: float) -> str:
    return f"+{v}" if v >= 0 else str(v)


def _kpi_strip(ins: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_events": s["total_events"],
        "date_range": f"{s['date_min'][:4]}–{s['date_max'][:4]}",
        "net_signal": ins["net_signal"],
        "net_label": ins["net_label"],
        "pkg_yoy": ins["pkg_chg_pct"],
        "launches_yoy": ins["launches_chg_pct"],
        "window_label": ins["window_label"],
    }


def _growth_narrative(ins: dict[str, Any]) -> str:
    net = ins["net_signal"]
    g = ins["growth_recent"]
    c = ins["closures_recent"]
    label = ins["net_label"]
    verb = "outpacing" if net >= 0 else "lagging behind"
    thesis = (
        "Each facility expansion or new build represents a sustained procurement "
        "relationship — the most direct volume driver for a packaging supplier."
        if net >= 0
        else
        "Closures signal customer attrition risk; monitor whether specific customers "
        "of the target are among those exiting capacity."
    )
    return (
        f"In the trailing 12 months ({ins['window_label']}), "
        f"expansions and new builds totalled <strong>{g} events</strong>, "
        f"{verb} closures at <strong>{c}</strong> — a "
        f"<strong>{_sign(net)} net signal</strong> ({label}). "
        f"{thesis}"
    )


def _packaging_narrative(ins: dict[str, Any]) -> str:
    pkg = ins["pkg_recent"]
    pct = ins["pkg_chg_pct"]
    direction = ins["pkg_direction"]
    qualifier = (
        "signals a growing addressable market with more active buyers"
        if direction == "up"
        else "may indicate sector consolidation or quieter capex cycles"
        if direction == "down"
        else "suggests a mature, steady competitive environment"
    )
    return (
        f"Packaging-specific events — covering new line investments, format changes, "
        f"and sustainability pivots — registered <strong>{pkg} events</strong> in the last 12 months "
        f"(<strong>{_sign(pct)}% YoY</strong>, trending {direction}). "
        f"Elevated activity {qualifier}. "
        f"This chart is the most direct read on the target's sector competitiveness."
    )


def _launches_narrative(ins: dict[str, Any]) -> str:
    l = ins["launches_recent"]
    pct = ins["launches_chg_pct"]
    direction = ins["launches_direction"]
    read = (
        "a growing pipeline that will require new and adapted packaging formats"
        if direction == "up"
        else "a contracting innovation cycle — monitor whether this reflects category maturity or cost pressure"
        if direction == "down"
        else "a steady innovation cadence that sustains baseline packaging demand"
    )
    return (
        f"Every new food or beverage SKU demands new or revised packaging specifications. "
        f"Food manufacturers recorded <strong>{l} product launch events</strong> in the trailing 12 months "
        f"(<strong>{_sign(pct)}% YoY</strong>, trending {direction}) — indicating {read}."
    )


def _market_narrative(ins: dict[str, Any], s: dict[str, Any]) -> str:
    counts = s["category_counts"]
    dom = max(counts, key=counts.get)
    dom_n = counts[dom]
    dom_pct = round(dom_n / s["total_events"] * 100)
    return (
        f"Across <strong>{s['total_events']:,} tracked events</strong> from "
        f"{s['date_min'][:7]} to {s['date_max'][:7]}, "
        f"<strong>{_CAT_LABELS[dom]}</strong> is the dominant signal at "
        f"{dom_n} events ({dom_pct}% of total) — reflecting sustained innovation "
        f"pressure from food manufacturers, the primary customer base for packaging suppliers. "
        f"Activity peaked at <strong>{s['peak_month_count']} events in {s['peak_month']}</strong>. "
        f"The stacked view below shows how the composition of activity has shifted "
        f"across the five categories over time."
    )


# ── Plotly chart spec builders ────────────────────────────────────────────────

_PE_LAYOUT = {
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#F4F6F9",
    "font": {"family": "Inter, sans-serif", "color": "#12263A"},
    "margin": {"t": 40, "r": 20, "b": 60, "l": 50},
    "legend": {"orientation": "h", "y": -0.2},
    "xaxis": {"gridcolor": "#E0E4EC", "linecolor": "#E0E4EC"},
    "yaxis": {"gridcolor": "#E0E4EC", "linecolor": "#E0E4EC", "zeroline": False},
}


def _ts(series: pd.Series) -> list[str]:
    """Convert a datetime Series to ISO strings."""
    return [str(v)[:10] if hasattr(v, "strftime") else str(v) for v in series.tolist()]


def _build_capacity_chart(gs: pd.DataFrame) -> dict[str, Any]:
    xs = _ts(gs["month"])
    growth = gs["growth_rolling"].tolist()
    closures = gs["closures_rolling"].tolist()

    traces = [
        {
            "x": xs,
            "y": growth,
            "name": "Expansions + New Builds (3M avg)",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": "#1A7A4A", "width": 2.5},
        },
        {
            "x": xs,
            "y": closures,
            "name": "Closures (3M avg)",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": "#A93226", "width": 2.5, "dash": "dash"},
        },
        {
            "x": xs + xs[::-1],
            "y": [max(g, c) for g, c in zip(growth, closures)]
                + [min(g, c) for g, c in zip(growth, closures)][::-1],
            "fill": "toself",
            "fillcolor": "rgba(26,122,74,0.10)",
            "line": {"color": "rgba(255,255,255,0)"},
            "showlegend": False,
            "type": "scatter",
            "hoverinfo": "skip",
            "name": "_fill",
        },
    ]
    layout = {
        **_PE_LAYOUT,
        "title": {"text": "Customer Capacity — Expansions & New Builds vs Closures", "x": 0.0, "font": {"size": 14}},
        "yaxis": {**_PE_LAYOUT["yaxis"], "title": "Events per month"},
    }
    return {"data": traces, "layout": layout}


def _build_packaging_chart(ps: pd.DataFrame) -> dict[str, Any]:
    xs = _ts(ps["month"])
    traces = [
        {
            "x": xs,
            "y": ps["packaging"].tolist(),
            "name": "Packaging events (monthly)",
            "type": "bar",
            "marker": {"color": "#004A7C", "opacity": 0.75},
        },
        {
            "x": xs,
            "y": ps["packaging_rolling"].tolist(),
            "name": "3-month rolling average",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": "#CA6F1E", "width": 2.5},
        },
    ]
    layout = {
        **_PE_LAYOUT,
        "title": {"text": "Packaging Sector Activity — Your Target's Competitive Arena", "x": 0.0, "font": {"size": 14}},
        "yaxis": {**_PE_LAYOUT["yaxis"], "title": "Events per month"},
        "barmode": "overlay",
    }
    return {"data": traces, "layout": layout}


def _build_launches_chart(qpivot: pd.DataFrame) -> dict[str, Any]:
    xs = _ts(pd.Series(qpivot.index))
    ys = qpivot.get("PRODUCT_LAUNCHES", pd.Series(0, index=qpivot.index)).tolist()
    rolling = pd.Series(ys, dtype=float).rolling(3, min_periods=1).mean().tolist()
    traces = [
        {
            "x": xs,
            "y": ys,
            "name": "Product launches (quarterly)",
            "type": "bar",
            "marker": {"color": "#6C3483", "opacity": 0.78},
        },
        {
            "x": xs,
            "y": rolling,
            "name": "3-quarter rolling average",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": "#CA6F1E", "width": 2.5},
        },
    ]
    layout = {
        **_PE_LAYOUT,
        "title": {"text": "Innovation Pipeline — Product Launches as a Packaging Demand Proxy", "x": 0.0, "font": {"size": 14}},
        "yaxis": {**_PE_LAYOUT["yaxis"], "title": "Events per quarter"},
        "barmode": "overlay",
    }
    return {"data": traces, "layout": layout}


def _build_market_chart(qpivot: pd.DataFrame) -> dict[str, Any]:
    xs = _ts(pd.Series(qpivot.index))
    colors = {
        "CLOSURES":         "#A93226",
        "NEW_BUILDS":       "#2E86C1",
        "EXPANSIONS":       "#1A7A4A",
        "PRODUCT_LAUNCHES": "#6C3483",
        "PACKAGING":        "#004A7C",
    }
    traces = []
    for cat in _STACK_ORDER:
        ys = qpivot.get(cat, pd.Series(0, index=qpivot.index)).tolist()
        traces.append({
            "x": xs,
            "y": ys,
            "name": _CAT_LABELS[cat],
            "type": "scatter",
            "mode": "none",
            "fill": "tonexty",
            "fillcolor": colors[cat],
            "stackgroup": "one",
            "line": {"color": colors[cat]},
        })
    layout = {
        **_PE_LAYOUT,
        "title": {"text": "Total Market Activity — All Five Categories, Quarterly", "x": 0.0, "font": {"size": 14}},
        "yaxis": {**_PE_LAYOUT["yaxis"], "title": "Events per quarter"},
    }
    return {"data": traces, "layout": layout}


# ── Cached analysis (loaded once at startup) ──────────────────────────────────

_ANALYSIS: dict[str, Any] = {}


def _get_analysis() -> dict[str, Any]:
    global _ANALYSIS
    if not _ANALYSIS:
        _ANALYSIS = run_analysis()
    return _ANALYSIS


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/dashboard")
async def dashboard() -> JSONResponse:
    try:
        data = _get_analysis()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    ins = data["insights"]
    s   = data["stat_summary"]
    gs  = data["growth_signals"]
    ps  = data["packaging_signal"]
    qp  = data["quarterly_pivot"]

    payload = {
        "kpi": _kpi_strip(ins, s),
        "narratives": {
            "capacity":  _growth_narrative(ins),
            "packaging": _packaging_narrative(ins),
            "launches":  _launches_narrative(ins),
            "market":    _market_narrative(ins, s),
        },
        "charts": {
            "capacity":  _build_capacity_chart(gs),
            "packaging": _build_packaging_chart(ps),
            "launches":  _build_launches_chart(qp),
            "market":    _build_market_chart(qp),
        },
    }
    return JSONResponse(content=payload)


@app.get("/api/briefing")
async def briefing() -> JSONResponse:
    data = get_or_load_briefing()
    return JSONResponse(content={"briefing": data})


@app.post("/api/refresh-briefing")
async def refresh_briefing_endpoint() -> JSONResponse:
    try:
        data = refresh_briefing()
        return JSONResponse(content={"briefing": data})
    except InsightsError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class ChatRequest(BaseModel):
    message: str
    history: List = []
    briefing: Optional[Dict] = None


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest) -> JSONResponse:
    try:
        result = chat_turn(
            message=req.message,
            history=req.history,
            briefing=req.briefing,
        )
        chunks = result.get("chunks_used", [])
        sources_html = _format_sources_html(chunks)
        return JSONResponse(content={
            "history": result["history"],
            "sources_html": sources_html,
            "answer": result["answer"],
        })
    except ChatError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _format_sources_html(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "<p style='color:#999;font-style:italic'>No source passages retrieved.</p>"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        preview = (chunk.get("text") or "")[:200].replace("<", "&lt;").replace(">", "&gt;")
        parts.append(
            f"<div style='margin-bottom:10px;padding:10px;background:#F4F6F9;"
            f"border-left:3px solid #004A7C;border-radius:4px;font-size:0.85em'>"
            f"<strong>[{i}] {chunk.get('category', '')}</strong> — "
            f"{chunk.get('source', '')}, {chunk.get('date', '')} "
            f"<span style='color:#888'>(sim: {chunk.get('similarity', 0):.3f})</span><br>"
            f"<span style='color:#445566'>{preview}…</span>"
            f"</div>"
        )
    return "\n".join(parts)


# ── Static files ──────────────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)

app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "7860"))
    uvicorn.run("api:app", host="127.0.0.1", port=port, reload=False)
