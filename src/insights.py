"""
insights.py

Generate Claude intelligence findings from corpus events and cache the
briefing to output/insights_briefing.json.
"""

from __future__ import annotations

import json
import os
import re
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from analysis import run_analysis
from generate import MODEL, _get_client

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIEFING_PATH = REPO_ROOT / "output" / "insights_briefing.json"

SAMPLE_SIZE      = int(os.getenv("INSIGHTS_SAMPLE_SIZE", "30"))
INSIGHTS_MAX_TOKENS = int(os.getenv("INSIGHTS_MAX_TOKENS", "2000"))
MAX_RETRIES      = 3
SLEEP_SECS       = 2
RATE_LIMIT_WAIT_SECS = 30

# ── Groups that structure the output ──────────────────────────────────────────

GROUPS = ["URGENT", "RISKS", "OPPORTUNITIES", "GOOD TO KNOW"]

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior analyst at a private equity fund writing a
due diligence intelligence report for the Investment Committee (IC).

Context: The fund is evaluating an acquisition of a packaging manufacturer that
supplies food and beverage producers. The corpus covers five-plus years of
industry news across five event categories: CLOSURES, EXPANSIONS, NEW_BUILDS,
PRODUCT_LAUNCHES, and PACKAGING.

Your report must surface what aggregate charts cannot show: specific companies,
geographies, sub-themes, and time windows grounded in the provided sample events.

──────────────────────────────────────────────────────────────────────────────
OUTPUT FORMAT — copy these exact theme headers, then add findings beneath each:
──────────────────────────────────────────────────────────────────────────────

## CUSTOMER CAPACITY

Findings about customer expansions, new builds, and closures — the direct driver
of packaging volume demand. Write 1–3 findings.

## PACKAGING SECTOR

Findings about competitive activity within the packaging industry itself —
new lines, format changes, sustainability pivots, new entrants. Write 1–2 findings.

## INNOVATION PIPELINE

Findings about product launches — each new SKU demands new or adapted packaging.
Write 1–2 findings.

## MARKET MOMENTUM

Findings about broader dynamics: geographic patterns, cross-category signals,
or structural shifts that cut across the other themes. Write 1–2 findings.

──────────────────────────────────────────────────────────────────────────────
Each finding must use this exact sub-format:

### [TAG] Finding title here

Where TAG is exactly one of: URGENT | RISK | OPPORTUNITY | NOTE

- URGENT  : directly affects the investment thesis or near-term valuation
- RISK    : downside signal requiring active monitoring
- OPPORTUNITY : upside potential for a packaging supplier
- NOTE    : useful context, not decision-critical

Then three mandatory sub-sections:

**What the data shows:** One sentence. Name specific companies or geographies
if the evidence supports it.

**Evidence from corpus:** 2–3 bullets. Each must cite at least one event from
the sample — company name, approximate date, what they announced or did.

**Why this matters for a packaging supplier:** 1–2 sentences. State the direct
investment implication: upside, risk, volume impact, or strategic consideration.

──────────────────────────────────────────────────────────────────────────────
RULES:
- Use these EXACT four ## headers; do not rename or add others.
- Total findings across all four themes: 5–7.
- Do NOT repeat aggregate statistics already visible in the dashboard charts.
- Do NOT invent companies, dates, or events not in the sample.
- Use UK English spelling.
- Total length: 600–900 words."""


class InsightsError(Exception):
    """Raised when insight generation or cache I/O fails."""


# ── Claude call ────────────────────────────────────────────────────────────────

def _call_claude(
    prompt: str,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
    max_tokens: int = INSIGHTS_MAX_TOKENS,
) -> str:
    api_client = client or _get_client()
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = api_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        except anthropic.RateLimitError as exc:
            last_error = exc
            wait = RATE_LIMIT_WAIT_SECS * (attempt + 1)
            print(f"    Rate limit — waiting {wait}s before retry...")
            time.sleep(wait)

        except anthropic.APIError as exc:
            last_error = InsightsError(f"Anthropic API error: {exc}")
            if attempt < MAX_RETRIES - 1:
                print(f"    API error (attempt {attempt + 1}): {exc}")
                time.sleep(SLEEP_SECS)

        except Exception as exc:
            last_error = InsightsError(f"Insight generation failed: {exc}")
            if attempt < MAX_RETRIES - 1:
                print(f"    Failed (attempt {attempt + 1}): {exc}")
                time.sleep(SLEEP_SECS)

    raise InsightsError(
        f"Claude insight generation failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ── Sampling ───────────────────────────────────────────────────────────────────

def _sample_records(
    records: list[dict[str, Any]],
    sample_size: int = SAMPLE_SIZE,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Stratified sample: equal slots per category, most-recent events first.
    Ensures every category is represented even if one dominates by volume.
    """
    rng = random.Random(seed)

    by_cat: dict[str, list] = defaultdict(list)
    for r in records:
        by_cat[r.get("category", "OTHER")].append(r)

    cats = sorted(by_cat.keys())
    per_cat = max(2, sample_size // len(cats))

    selected: list[dict[str, Any]] = []
    for cat in cats:
        items = sorted(by_cat[cat], key=lambda x: x.get("date", ""), reverse=True)
        selected.extend(items[:per_cat])

    # Fill remaining slots from unselected pool
    selected_ids = {id(r) for r in selected}
    pool = [r for r in records if id(r) not in selected_ids]
    remaining = sample_size - len(selected)
    if remaining > 0 and pool:
        selected.extend(rng.sample(pool, min(remaining, len(pool))))

    selected = selected[:sample_size]
    rng.shuffle(selected)
    return selected


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_insights_prompt(
    stat_summary: dict[str, Any],
    sample_records: list[dict[str, Any]],
    insights: dict[str, Any] | None = None,
) -> str:
    sample_lines: list[str] = []
    for i, r in enumerate(sample_records, 1):
        sample_lines.append(
            "\n".join([
                f"[Event {i}]",
                f"  category : {r.get('category', '')}",
                f"  source   : {r.get('source', '')}",
                f"  date     : {r.get('date', '')}",
                f"  text     : {r.get('text', '')}",
            ])
        )

    quant_ctx = ""
    if insights:
        net = insights["net_signal"]
        quant_ctx = (
            "\n\nQuantitative context the charts already show "
            "(do NOT repeat — your findings must go beyond this):\n"
            f"  • Total events: {stat_summary.get('total_events', '?')}\n"
            f"  • Date range: {stat_summary.get('date_min', '')} → "
            f"{stat_summary.get('date_max', '')}\n"
            f"  • Trailing 12M net capacity: {net:+d} "
            f"({insights['growth_recent']} expansions/new-builds vs "
            f"{insights['closures_recent']} closures)\n"
            f"  • Packaging events YoY: {insights['pkg_chg_pct']:+.1f}%\n"
            f"  • Product launches YoY: {insights['launches_chg_pct']:+.1f}%\n"
            f"  • Window: {insights['window_label']}"
        )

    return (
        f"Produce the intelligence report from the sample events below."
        f"{quant_ctx}\n\n"
        f"Sample events ({len(sample_records)} of "
        f"{stat_summary.get('total_events', '?')} total):\n\n"
        + "\n\n".join(sample_lines)
    )


# ── Parsing ────────────────────────────────────────────────────────────────────

_THEME_KEYWORDS: dict[str, str] = {
    "CUSTOMER": "CUSTOMER CAPACITY",
    "PACKAGING": "PACKAGING SECTOR",
    "INNOVATION": "INNOVATION PIPELINE",
    "MARKET": "MARKET MOMENTUM",
}

_URGENCY_TAGS = {"URGENT", "RISK", "OPPORTUNITY", "NOTE"}
_URGENCY_RE   = re.compile(r"^\[([A-Z]+)\]\s*(.+)$")


def _normalize_theme(raw: str) -> str:
    """Map Claude's free-text theme header to one of our 4 canonical keys."""
    upper = raw.strip().upper()
    for keyword, canonical in _THEME_KEYWORDS.items():
        if keyword in upper:
            return canonical
    return upper  # fallback


def _parse_themed_findings(narrative: str) -> dict[str, list[dict[str, str]]]:
    """
    Parse Claude's theme-structured output into a dict keyed by canonical theme.

    Returns: {
        "CUSTOMER CAPACITY": [{"urgency": "OPPORTUNITY", "title": "...", "content": "..."}, ...],
        "PACKAGING SECTOR":  [...],
        ...
    }
    """
    themed: dict[str, list[dict[str, str]]] = {}
    text = narrative.replace("\r\n", "\n").strip()

    group_blocks = re.split(r"\n(?=## )", text)
    for block in group_blocks:
        block = block.strip()
        if not block.startswith("## "):
            continue
        first_line, _, rest = block.partition("\n")
        theme_key = _normalize_theme(first_line[3:])

        findings: list[dict[str, str]] = []
        for fb in re.split(r"\n(?=### )", rest):
            fb = fb.strip()
            if not fb.startswith("### "):
                continue
            fb_first, _, fb_rest = fb.partition("\n")
            title_raw = fb_first[4:].strip()
            m = _URGENCY_RE.match(title_raw)
            if m and m.group(1) in _URGENCY_TAGS:
                urgency = m.group(1)
                title   = m.group(2).strip()
            else:
                urgency = "NOTE"
                title   = title_raw
            findings.append({"urgency": urgency, "title": title, "content": fb_rest.strip()})

        if findings:
            themed.setdefault(theme_key, []).extend(findings)

    return themed


def _parse_grouped_findings(narrative: str) -> list[dict[str, str]]:
    """Legacy parser — kept for backward compatibility with old cached briefings."""
    findings: list[dict[str, str]] = []
    text = narrative.replace("\r\n", "\n").strip()
    for block in re.split(r"\n(?=## )", text):
        block = block.strip()
        if not block.startswith("## "):
            continue
        first_line, _, rest = block.partition("\n")
        group_name = first_line[3:].strip().upper()
        for fb in re.split(r"\n(?=### )", rest):
            fb = fb.strip()
            if not fb.startswith("### "):
                continue
            fb_first, _, fb_rest = fb.partition("\n")
            title = fb_first[4:].strip()
            if title:
                findings.append({"group": group_name, "title": title, "content": fb_rest.strip()})
    return findings


# ── Markdown → plain HTML ──────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    """Minimal markdown→HTML converter: bold, italic, bullets, paragraphs."""
    lines = text.split("\n")
    out: list[str] = []
    in_list = False

    for line in lines:
        stripped = line.rstrip()

        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                out.append("<ul style='margin:6px 0 6px 20px;padding:0'>")
                in_list = True
            item = stripped[2:]
            item = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", item)
            item = re.sub(r"\*(.+?)\*",    r"<em>\1</em>",          item)
            out.append(f"<li style='margin:3px 0'>{item}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if stripped:
                content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", stripped)
                content = re.sub(r"\*(.+?)\*",    r"<em>\1</em>",          content)
                out.append(f"<p style='margin:5px 0'>{content}</p>")

    if in_list:
        out.append("</ul>")

    return "\n".join(out)


# ── Briefing generation ────────────────────────────────────────────────────────

def generate_briefing(
    stat_summary: dict[str, Any] | None = None,
    records: list[dict[str, Any]] | None = None,
    insights: dict[str, Any] | None = None,
    sample_size: int = SAMPLE_SIZE,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
) -> dict[str, Any]:
    """
    Generate a Claude intelligence report and return the structured payload.
    Runs run_analysis() to fetch data if not provided.
    """
    if stat_summary is None or records is None:
        analysis = run_analysis()
        stat_summary = stat_summary or analysis["stat_summary"]
        records = records or analysis["records"]
        if insights is None:
            insights = analysis.get("insights")

    sample = _sample_records(records, sample_size=sample_size)
    prompt = build_insights_prompt(stat_summary, sample, insights=insights)

    if not os.getenv("ANTHROPIC_API_KEY"):
        narrative = (
            "_No `ANTHROPIC_API_KEY` set — click **Refresh Intelligence Findings** "
            "once the key is configured in `.env`._"
        )
        themed: dict[str, list[dict[str, str]]] = {}
    else:
        narrative = _call_claude(prompt, client=client, model=model)
        themed    = _parse_themed_findings(narrative)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "sample_size": len(sample),
        "stat_summary": stat_summary,
        "narrative": narrative,
        "themed_findings": themed,
        # Legacy field — kept so old cached briefings still render
        "findings": _parse_grouped_findings(narrative) if os.getenv("ANTHROPIC_API_KEY") else [],
    }


def save_briefing(briefing: dict[str, Any], path: Path = BRIEFING_PATH) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(briefing, f, indent=2, ensure_ascii=False)
    return path


def load_briefing(path: Path = BRIEFING_PATH) -> dict[str, Any] | None:
    path = path.expanduser().resolve()
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def refresh_briefing(path: Path = BRIEFING_PATH) -> dict[str, Any]:
    """Regenerate intelligence findings and write them to the cache file."""
    briefing = generate_briefing()
    save_briefing(briefing, path=path)
    return briefing


def get_or_load_briefing(path: Path = BRIEFING_PATH) -> dict[str, Any] | None:
    """Return cached briefing if present, otherwise None."""
    return load_briefing(path=path)


if __name__ == "__main__":
    briefing = refresh_briefing()
    print(f"Saved to   : {BRIEFING_PATH}")
    print(f"Generated  : {briefing['generated_at']}")
    print(f"Sample     : {briefing['sample_size']} events")
    print(f"Findings   : {len(briefing.get('findings', []))}")
    for f in briefing.get("findings", []):
        print(f"  [{f['group']}] {f['title']}")
