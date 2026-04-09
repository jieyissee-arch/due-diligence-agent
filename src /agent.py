"""
agent.py

Core agent loop for the due diligence news agent.

Reads article URLs from inputs/urls.json, fetches each article,
extracts structured industry events using Claude, validates output
against the schema, and writes results to output/report.json.

Retry logic: if Claude's output fails validation, the agent retries
with a reinforced prompt up to MAX_RETRIES times before logging an
error result and moving on.

Usage:
    python src/agent.py

Environment variables:
    ANTHROPIC_API_KEY  — required
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from schema import validate_extraction, build_extraction_result, build_error_result
from tools import fetch_article, get_article_text_for_claude

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

MODEL       = "claude-haiku-4-5-20251001"
MAX_TOKENS  = 300
MAX_RETRIES = 3
SLEEP_SECS  = 2

INPUT_FILE  = Path("inputs/urls.json")
OUTPUT_DIR  = Path("output")
OUTPUT_FILE = OUTPUT_DIR / "report.json"

SYSTEM_PROMPT = """You are a due diligence analyst extracting factual corporate
events from trade press articles about the food manufacturing and packaging industry.
Respond with raw JSON only. No markdown, no code fences, no preamble."""

EXTRACTION_PROMPT = """Read the article below and identify any events that fall
into these five categories:

1. CLOSURES: A company closing or shutting down a manufacturing site,
   factory, or production facility
2. EXPANSIONS: A company renovating, expanding, or investing in an
   existing manufacturing site or production line
3. PRODUCT_LAUNCHES: A company launching a new product, product range,
   or brand
4. NEW_BUILDS: A company building, constructing, or opening a brand
   new manufacturing site or facility
5. PACKAGING: A company changing, updating, or announcing new packaging,
   particularly moves toward sustainable, recyclable, or eco-friendly
   packaging materials

For each event found extract:
- topic: one of the five categories above (must be exact)
- company: the company involved
- description: one sentence describing what happened
- location: city and country if mentioned, otherwise null
- scale: investment amount, capacity, or jobs if mentioned, otherwise null

If no relevant events are found return {{"events": []}}

Respond in raw JSON only using this exact structure:
{{"events": [{{"topic": "...", "company": "...", "description": "...",
"location": "...", "scale": "..."}}]}}

Article URL: {url}
Article text: {text}"""

RETRY_PROMPT = """Your previous response did not match the required format.

Reason: {reason}

Please try again. You must respond with raw JSON only using this exact structure:
{{"events": [{{"topic": "...", "company": "...", "description": "...",
"location": "...", "scale": "..."}}]}}

Valid topics are: CLOSURES, EXPANSIONS, PRODUCT_LAUNCHES, NEW_BUILDS, PACKAGING

If no relevant events are found return {{"events": []}}

Article URL: {url}
Article text: {text}"""


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_events_from_article(
    client: anthropic.Anthropic,
    url: str,
    text: str
) -> dict:
    """
    Calls Claude to extract events from article text.
    Validates output and retries with a reinforced prompt if validation fails.

    Returns a completed ExtractionResult dict.
    """
    prompt = EXTRACTION_PROMPT.format(url=url, text=text)
    retries = 0

    for attempt in range(MAX_RETRIES):

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"    Rate limit hit. Waiting {wait}s before retry...")
            time.sleep(wait)
            continue

        except anthropic.APIError as e:
            return build_error_result(url, f"API error: {e}", retries)

        raw = response.content[0].text
        is_valid, events, reason = validate_extraction(raw)

        if is_valid:
            return build_extraction_result(url, events, retries)

        # Validation failed — build retry prompt and try again
        retries += 1
        print(f"    Validation failed (attempt {attempt + 1}): {reason}")

        if attempt < MAX_RETRIES - 1:
            prompt = RETRY_PROMPT.format(
                reason=reason,
                url=url,
                text=text
            )
            time.sleep(SLEEP_SECS)

    # All retries exhausted
    return build_error_result(
        url,
        f"Failed validation after {MAX_RETRIES} attempts. Last reason: {reason}",
        retries
    )


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent():
    """
    Main agent loop.

    Loads URLs from inputs/urls.json, fetches each article,
    extracts events, and writes results to output/report.json.
    """
    # Validate environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. "
            "Copy .env.example to .env and add your key."
        )

    # Load input URLs
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with open(INPUT_FILE) as f:
        urls = json.load(f)

    if not isinstance(urls, list) or len(urls) == 0:
        raise ValueError("urls.json must be a non-empty JSON array of URL strings")

    # Prepare output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\nDue Diligence News Agent")
    print(f"{'=' * 40}")
    print(f"Articles to process : {len(urls)}")
    print(f"Model               : {MODEL}")
    print(f"Max retries         : {MAX_RETRIES}")
    print(f"Output              : {OUTPUT_FILE}")
    print(f"{'=' * 40}\n")

    client = anthropic.Anthropic(api_key=api_key)

    results = []
    total_cost = 0.0
    errors = 0
    events_found = 0

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url[:80]}")

        # Step 1: Fetch article
        article = fetch_article(url)

        if not article["success"]:
            print(f"    Fetch failed: {article['error']}")
            results.append(build_error_result(url, article["error"]))
            errors += 1
            continue

        # Step 2: Prepare text for Claude
        text = get_article_text_for_claude(article, max_words=300)

        # Step 3: Extract events via Claude
        result = extract_events_from_article(client, url, text)

        # Step 4: Log progress
        n_events = len(result.get("events", []))
        events_found += n_events
        status = "validated" if result["validated"] else "error"
        print(f"    {status} | events: {n_events} | retries: {result['retries']}")

        if not result["validated"]:
            errors += 1

        results.append(result)
        time.sleep(SLEEP_SECS)

    # Write report
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": MODEL,
        "articles_processed": len(urls),
        "events_extracted": events_found,
        "errors": errors,
        "results": results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    # Summary
    print(f"\n{'=' * 40}")
    print(f"Complete")
    print(f"Articles processed  : {len(urls)}")
    print(f"Events extracted    : {events_found}")
    print(f"Errors              : {errors}")
    print(f"Report written to   : {OUTPUT_FILE}")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    run_agent()
