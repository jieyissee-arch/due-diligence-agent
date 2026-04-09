"""
tools.py

Tool definitions for the due diligence news agent.
Each function represents a discrete action Claude can request
during the agent loop.

Tools:
    fetch_article — retrieves and cleans article text from a URL
"""

import re
import httpx
from bs4 import BeautifulSoup


# Request timeout in seconds
REQUEST_TIMEOUT = 15

# Minimum number of words for article text to be considered usable
MIN_WORD_COUNT = 50

# Headers to reduce likelihood of being blocked
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DueDiligenceAgent/1.0)"
    )
}


def fetch_article(url: str) -> dict:
    """
    Fetches article content from a URL and returns cleaned text.

    Returns a dict with:
        success  (bool)   — whether fetch and parse succeeded
        url      (str)    — the original URL
        text     (str)    — cleaned article body text
        error    (str)    — error message if success is False
    """
    try:
        response = httpx.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True
        )
        response.raise_for_status()

    except httpx.TimeoutException:
        return _error_result(url, f"Request timed out after {REQUEST_TIMEOUT}s")

    except httpx.HTTPStatusError as e:
        return _error_result(
            url, f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
        )

    except httpx.RequestError as e:
        return _error_result(url, f"Request error: {e}")

    # Parse HTML and extract body text
    text = _extract_text(response.text)

    if not text:
        return _error_result(url, "No article text found after parsing")

    word_count = len(text.split())
    if word_count < MIN_WORD_COUNT:
        return _error_result(
            url,
            f"Article text too short ({word_count} words, minimum {MIN_WORD_COUNT})"
        )

    return {
        "success": True,
        "url": url,
        "text": text,
        "error": None
    }


def _extract_text(html: str) -> str:
    """
    Extracts and cleans body text from raw HTML.

    Removes navigation, headers, footers, scripts, and ads.
    Returns plain text suitable for passing to Claude.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "header",
                     "footer", "aside", "form", "figure"]):
        tag.decompose()

    # Try to find the main article body
    # Trade news sites commonly use <article> or role="main"
    article = (
        soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.find("main")
        or soup.body
    )

    if not article:
        return ""

    # Extract text and normalise whitespace
    raw_text = article.get_text(separator=" ")
    cleaned = re.sub(r'\s+', ' ', raw_text).strip()

    return cleaned


def get_article_text_for_claude(article: dict, max_words: int = 300) -> str:
    """
    Truncates article text to max_words for passing to Claude.

    Claude does not need the full article — the first 300 words
    typically contain the key facts for event extraction.
    Returns truncated text as a string.
    """
    text = article.get("text", "")
    words = text.split()

    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words])


def _error_result(url: str, error: str) -> dict:
    """
    Builds a failed fetch result dict.
    """
    return {
        "success": False,
        "url": url,
        "text": "",
        "error": error
    }
