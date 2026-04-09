"""
schema.py

Defines the structure of an extracted industry event and validates Claude's output against it before accepting results.

The five event categories reflect due diligence research focus areas in food manufacturing and packaging sectors.
"""

import json
import re


# The five extraction categories
VALID_TOPICS = {
    "CLOSURES",        # company closing a manufacturing site or facility
    "EXPANSIONS",      # company expanding or investing in an existing site
    "PRODUCT_LAUNCHES",# company launching a new product or brand
    "NEW_BUILDS",      # company building or opening a new facility
    "PACKAGING"        # company changing packaging, especially sustainability moves
}


def validate_event(event: dict) -> tuple[bool, str]:
    """
    Validates a single extracted event against the schema.
    Returns (is_valid, reason) tuple.
    """
    required_fields = {"topic", "company", "description", "location", "scale"}

    # Check all required fields are present
    missing = required_fields - event.keys()
    if missing:
        return False, f"Missing required fields: {missing}"

    # topic must be one of the five valid categories
    if event["topic"] not in VALID_TOPICS:
        return False, (
            f"Invalid topic '{event['topic']}'. "
            f"Must be one of: {VALID_TOPICS}"
        )

    # company must be present and non-empty
    if not event.get("company", "").strip():
        return False, "Company name is empty"

    # description must be meaningful
    if len(event.get("description", "").strip()) < 20:
        return False, "Description is too short or empty"

    # location and scale can be null — that is valid
    return True, "valid"


def parse_response(raw: str) -> dict:
    """
    Strips markdown fences from Claude's response and parses JSON.
    Claude occasionally wraps output in ```json blocks despite instructions.
    """
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned.strip())
    return json.loads(cleaned)


def validate_extraction(raw_output: str) -> tuple[bool, list[dict], str]:
    """
    Parses and validates Claude's raw JSON output.
    Returns (is_valid, events_list, reason) tuple.

    Expects Claude to return:
    {"events": [{"topic": ..., "company": ..., ...}, ...]}
    """
    # Attempt to parse JSON
    try:
        parsed = parse_response(raw_output)
    except json.JSONDecodeError as e:
        return False, [], f"JSON parse error: {e}"

    # Expect a dict with an "events" key
    if not isinstance(parsed, dict) or "events" not in parsed:
        return False, [], "Output must contain an 'events' key"

    events = parsed["events"]

    # Empty list is valid — article may have no relevant events
    if len(events) == 0:
        return True, [], "valid — no events found"

    # Validate each event
    validated_events = []
    for i, event in enumerate(events):
        is_valid, reason = validate_event(event)
        if not is_valid:
            return False, [], f"Event {i} failed validation: {reason}"
        validated_events.append(event)

    return True, validated_events, "valid"


def build_extraction_result(
    url: str,
    validated_events: list[dict],
    retries: int = 0
) -> dict:
    """
    Builds the final output dict for a single article.
    """
    return {
        "url": url,
        "events": validated_events,
        "validated": True,
        "retries": retries,
        "error": None
    }


def build_error_result(url: str, error: str, retries: int = 0) -> dict:
    """
    Builds an error output dict when extraction fails after all retries.
    """
    return {
        "url": url,
        "events": [],
        "validated": False,
        "retries": retries,
        "error": error
    }
