"""
analysis.py

Pandas aggregations over demo_data.json for the insights dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "demo_data.json"

CATEGORY_ORDER = [
    "CLOSURES",
    "EXPANSIONS",
    "NEW_BUILDS",
    "PRODUCT_LAUNCHES",
    "PACKAGING",
]

_CAT_LABELS = {
    "CLOSURES":         "Closures",
    "EXPANSIONS":       "Expansions",
    "NEW_BUILDS":       "New Builds",
    "PRODUCT_LAUNCHES": "Product Launches",
    "PACKAGING":        "Packaging",
}


class AnalysisError(Exception):
    """Raised when demo data cannot be loaded or aggregated."""


def load_records(data_path: Path | None = None) -> list[dict[str, Any]]:
    path = (data_path or DEFAULT_DATA_PATH).expanduser().resolve()
    if not path.is_file():
        raise AnalysisError(
            f"Demo data not found: {path}. "
            "Copy demo_data.example.json to demo_data.json for a quickstart."
        )
    with path.open(encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise AnalysisError(f"{path} must contain a non-empty JSON array.")
    return records


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    required = {"category", "source", "date", "text"}
    missing = required - set(df.columns)
    if missing:
        raise AnalysisError(f"Records missing required fields: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def timeseries_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly event counts by category (long format)."""
    grouped = (
        df.dropna(subset=["date"])
        .groupby(["month", "category"], observed=True)
        .size()
        .reset_index(name="count")
    )
    if grouped.empty:
        return pd.DataFrame(columns=["month", "category", "count"])
    grouped["category"] = pd.Categorical(
        grouped["category"], categories=CATEGORY_ORDER, ordered=True
    )
    return grouped.sort_values(["month", "category"]).reset_index(drop=True)


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Total events per category."""
    grouped = df.groupby("category", observed=True).size().reset_index(name="count")
    grouped["category"] = pd.Categorical(
        grouped["category"], categories=CATEGORY_ORDER, ordered=True
    )
    return grouped.sort_values("category").reset_index(drop=True)


def source_totals(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top sources by event count."""
    return (
        df.groupby("source", observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def prepare_chart_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Categorical columns to str for Gradio native plots."""
    if df.empty:
        return df
    chart_df = df.copy()
    for col in chart_df.columns:
        if pd.api.types.is_categorical_dtype(chart_df[col]):
            chart_df[col] = chart_df[col].astype(str)
    return chart_df


def quarterly_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quarterly event counts by category — wide format with datetime index.
    Rows = first day of each quarter; columns = the five categories.
    """
    df2 = df.dropna(subset=["date"]).copy()
    df2["quarter"] = df2["date"].dt.to_period("Q").dt.to_timestamp()
    pivot = (
        df2.groupby(["quarter", "category"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    pivot.columns.name = None
    for cat in CATEGORY_ORDER:
        if cat not in pivot.columns:
            pivot[cat] = 0
    return pivot[CATEGORY_ORDER].sort_index()


def growth_signals(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Monthly capacity growth (EXPANSIONS + NEW_BUILDS) vs CLOSURES,
    plus rolling averages.
    """
    ts = timeseries_by_category(df)
    growth = (
        ts[ts["category"].isin(["EXPANSIONS", "NEW_BUILDS"])]
        .groupby("month", observed=True)["count"]
        .sum()
        .reset_index(name="growth")
    )
    closures = (
        ts[ts["category"] == "CLOSURES"]
        .groupby("month", observed=True)["count"]
        .sum()
        .reset_index(name="closures")
    )
    merged = (
        growth.merge(closures, on="month", how="outer")
        .fillna(0)
        .sort_values("month")
        .reset_index(drop=True)
    )
    merged["growth_rolling"] = merged["growth"].rolling(window, min_periods=1).mean()
    merged["closures_rolling"] = merged["closures"].rolling(window, min_periods=1).mean()
    return merged


def packaging_signal(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Monthly PACKAGING events and PRODUCT_LAUNCHES counts, with rolling averages.
    Used for the packaging spotlight and innovation pipeline charts.
    """
    ts = timeseries_by_category(df)
    pkg = (
        ts[ts["category"] == "PACKAGING"]
        .groupby("month", observed=True)["count"]
        .sum()
        .reset_index(name="packaging")
    )
    launches = (
        ts[ts["category"] == "PRODUCT_LAUNCHES"]
        .groupby("month", observed=True)["count"]
        .sum()
        .reset_index(name="launches")
    )
    merged = (
        pkg.merge(launches, on="month", how="outer")
        .fillna(0)
        .sort_values("month")
        .reset_index(drop=True)
    )
    merged["packaging_rolling"] = merged["packaging"].rolling(window, min_periods=1).mean()
    merged["launches_rolling"] = merged["launches"].rolling(window, min_periods=1).mean()
    return merged


def stat_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compact summary stats for narrative generation."""
    dated = df.dropna(subset=["date"])
    category_counts = (
        df.groupby("category", observed=True)
        .size()
        .sort_values(ascending=False)
        .astype(int)
        .to_dict()
    )
    monthly = dated.groupby("month", observed=True).size().sort_values(ascending=False)
    peak_month = monthly.index[0] if not monthly.empty else None
    peak_month_count = int(monthly.iloc[0]) if not monthly.empty else 0
    if peak_month is not None and hasattr(peak_month, "strftime"):
        peak_month = peak_month.strftime("%Y-%m")

    top_sources = (
        df.groupby("source", observed=True)
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="count")
    )
    return {
        "total_events": int(len(df)),
        "date_min": dated["date"].min().strftime("%Y-%m-%d") if not dated.empty else None,
        "date_max": dated["date"].max().strftime("%Y-%m-%d") if not dated.empty else None,
        "unique_sources": int(df["source"].nunique()),
        "category_counts": category_counts,
        "peak_month": peak_month,
        "peak_month_count": peak_month_count,
        "avg_events_per_month": round(float(monthly.mean()), 1) if not monthly.empty else 0.0,
        "top_sources": [
            {"source": row["source"], "count": int(row["count"])}
            for _, row in top_sources.iterrows()
        ],
    }


def derive_insights(df: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    """
    Compute narrative-ready insights for the IC dashboard.
    Compares the trailing 12 months against the prior 12 months.
    """
    dated = df.dropna(subset=["date"])
    max_date = dated["date"].max()
    cutoff_recent = max_date - pd.DateOffset(months=12)
    cutoff_prior = max_date - pd.DateOffset(months=24)

    recent = dated[dated["date"] > cutoff_recent]
    prior = dated[(dated["date"] > cutoff_prior) & (dated["date"] <= cutoff_recent)]

    def _count(frame: pd.DataFrame, cats) -> int:
        if isinstance(cats, str):
            cats = [cats]
        return int(len(frame[frame["category"].isin(cats)]))

    r_pkg = _count(recent, "PACKAGING")
    p_pkg = _count(prior, "PACKAGING")
    pkg_chg = round((r_pkg - p_pkg) / max(p_pkg, 1) * 100, 1)

    r_growth = _count(recent, ["EXPANSIONS", "NEW_BUILDS"])
    r_close = _count(recent, "CLOSURES")
    net = r_growth - r_close

    r_launches = _count(recent, "PRODUCT_LAUNCHES")
    p_launches = _count(prior, "PRODUCT_LAUNCHES")
    l_chg = round((r_launches - p_launches) / max(p_launches, 1) * 100, 1)

    r_exp = _count(recent, "EXPANSIONS")
    r_nb = _count(recent, "NEW_BUILDS")

    def _direction(chg: float) -> str:
        return "up" if chg > 5 else "down" if chg < -5 else "stable"

    def _signal(n: int) -> str:
        return "net positive" if n > 0 else "net negative" if n < 0 else "balanced"

    return {
        "pkg_recent": r_pkg,
        "pkg_prior": p_pkg,
        "pkg_chg_pct": pkg_chg,
        "pkg_direction": _direction(pkg_chg),
        "growth_recent": r_growth,
        "expansions_recent": r_exp,
        "new_builds_recent": r_nb,
        "closures_recent": r_close,
        "net_signal": net,
        "net_label": _signal(net),
        "launches_recent": r_launches,
        "launches_chg_pct": l_chg,
        "launches_direction": _direction(l_chg),
        "window_label": f"{cutoff_recent.strftime('%b %Y')} – {max_date.strftime('%b %Y')}",
    }


def run_analysis(data_path: Path | None = None) -> dict[str, Any]:
    """
    Load demo data and return all aggregations for the dashboard.
    """
    records = load_records(data_path)
    df = records_to_dataframe(records)
    summary = stat_summary(df)
    return {
        "records": records,
        "dataframe": df,
        "timeseries": prepare_chart_frame(timeseries_by_category(df)),
        "category_breakdown": prepare_chart_frame(category_breakdown(df)),
        "source_totals": prepare_chart_frame(source_totals(df)),
        "stat_summary": summary,
        "quarterly_pivot": quarterly_pivot(df),
        "growth_signals": growth_signals(df),
        "packaging_signal": packaging_signal(df),
        "insights": derive_insights(df, summary),
    }


if __name__ == "__main__":
    result = run_analysis()
    s = result["stat_summary"]
    ins = result["insights"]
    print(f"Total events      : {s['total_events']}")
    print(f"Date range        : {s['date_min']} → {s['date_max']}")
    print(f"Peak month        : {s['peak_month']} ({s['peak_month_count']} events)")
    print(f"Category counts   : {s['category_counts']}")
    print(f"Quarterly pivot   : {result['quarterly_pivot'].shape}")
    print(f"Growth signals    : {len(result['growth_signals'])} months")
    print(f"Packaging signal  : {len(result['packaging_signal'])} months")
    print(f"Insights window   : {ins['window_label']}")
    print(f"Net capacity      : {ins['net_signal']:+d} ({ins['net_label']})")
    print(f"Packaging YoY     : {ins['pkg_chg_pct']:+.1f}%")
    print(f"Launches YoY      : {ins['launches_chg_pct']:+.1f}%")
