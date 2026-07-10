"""
eval_scoring.py

Pure scoring helpers for RAG evaluation (no Chroma / Claude imports).
"""

from __future__ import annotations

import re
from typing import Any

_PASSAGE_CITE_RE = re.compile(r"\[Passage\s+(\d+)\]", re.IGNORECASE)
_ABSTENTION_RE = re.compile(
    r"\b(insufficient|not enough|no (relevant )?evidence|cannot answer|"
    r"don't have|do not have|missing|not (mentioned|provided|available)|"
    r"unable to|no information)\b",
    re.IGNORECASE,
)
_SOURCES_USED_RE = re.compile(r"sources used\s*:", re.IGNORECASE)


def extract_cited_passage_indices(answer: str) -> set[int]:
    """Return 1-based passage indices cited in the answer."""
    return {int(match) for match in _PASSAGE_CITE_RE.findall(answer)}


def has_sources_used_section(answer: str) -> bool:
    return bool(_SOURCES_USED_RE.search(answer))


def looks_like_abstention(answer: str) -> bool:
    return bool(_ABSTENTION_RE.search(answer))


def citation_valid(answer: str, num_chunks: int) -> bool:
    cited = extract_cited_passage_indices(answer)
    if not cited:
        return False
    return all(1 <= index <= num_chunks for index in cited)


def citation_present(answer: str) -> bool:
    return bool(extract_cited_passage_indices(answer))


def _normalize(text: str) -> str:
    return text.lower().strip()


def terms_in_texts(texts: list[str], terms: list[str]) -> bool:
    """True if any term appears as a substring in any text (case-insensitive)."""
    if not terms:
        return True
    haystack = _normalize(" ".join(texts))
    return any(_normalize(term) in haystack for term in terms)


def retrieval_category_hit(
    chunks: list[dict[str, Any]],
    expected_categories: list[str],
) -> bool:
    if not expected_categories:
        return True
    expected = {c.upper() for c in expected_categories}
    return any(chunk.get("category", "").upper() in expected for chunk in chunks)


def retrieval_term_hit(
    chunks: list[dict[str, Any]],
    required_terms_any: list[str],
) -> bool:
    texts = [str(chunk.get("text", "")) for chunk in chunks]
    return terms_in_texts(texts, required_terms_any)


def grounded_terms_in_cited_passages(
    answer: str,
    chunks: list[dict[str, Any]],
    required_terms_any: list[str],
) -> bool:
    """True if required terms appear in at least one cited passage."""
    if not required_terms_any:
        return True

    cited = extract_cited_passage_indices(answer)
    if not cited:
        return False

    cited_texts = [
        str(chunks[index - 1].get("text", ""))
        for index in sorted(cited)
        if 1 <= index <= len(chunks)
    ]
    return terms_in_texts(cited_texts, required_terms_any)


def max_chunk_similarity(chunks: list[dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    return max(float(chunk.get("similarity", 0.0)) for chunk in chunks)


def rate(values: list[bool | None]) -> float:
    scored = [v for v in values if v is not None]
    if not scored:
        return 0.0
    return sum(scored) / len(scored)
