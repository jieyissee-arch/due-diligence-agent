"""
chunking.py

Split extracted article records into retrieval-sized text passages for RAG.
"""

from __future__ import annotations

import re
from typing import Any


MIN_WORDS = 200
MAX_WORDS = 400
TARGET_WORDS = 300

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _word_count(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_BOUNDARY.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_long_sentence(sentence: str, max_words: int) -> list[str]:
    """Hard-split an oversized sentence at word boundaries."""
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]

    segments: list[str] = []
    for start in range(0, len(words), max_words):
        segments.append(" ".join(words[start : start + max_words]))
    return segments


def _split_text(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> list[str]:
    """
    Split text into passages of roughly min_words to max_words.

    Short texts are returned as a single passage. Longer texts are grouped
    sentence-by-sentence, falling back to word-boundary splits when needed.
    """
    text = text.strip()
    if not text:
        return []

    if _word_count(text) <= max_words:
        return [text]

    sentences = _split_sentences(text)
    if not sentences:
        return [text]

    units: list[str] = []
    for sentence in sentences:
        if _word_count(sentence) > max_words:
            units.extend(_split_long_sentence(sentence, max_words))
        else:
            units.append(sentence)

    chunks: list[str] = []
    current_units: list[str] = []
    current_words = 0

    for unit in units:
        unit_words = _word_count(unit)

        if current_units and current_words + unit_words > max_words and current_words >= min_words:
            chunks.append(" ".join(current_units))
            current_units = [unit]
            current_words = unit_words
        elif current_units and current_words + unit_words > max_words:
            chunks.append(" ".join(current_units))
            current_units = [unit]
            current_words = unit_words
        else:
            current_units.append(unit)
            current_words += unit_words

    if current_units:
        chunks.append(" ".join(current_units))

    # Merge a tiny trailing chunk into the previous passage when possible.
    if len(chunks) >= 2 and _word_count(chunks[-1]) < min_words:
        merged = f"{chunks[-2]} {chunks[-1]}"
        if _word_count(merged) <= max_words:
            chunks[-2] = merged
            chunks.pop()

    return chunks


def _make_chunk_id(source: str, date: str, record_index: int, chunk_index: int) -> str:
    safe_source = re.sub(r"[^a-zA-Z0-9]+", "-", source).strip("-").lower() or "unknown"
    safe_date = re.sub(r"[^0-9-]", "", date) or "unknown-date"
    return f"{safe_source}_{safe_date}_{record_index:04d}_{chunk_index:02d}"


def chunk_records(
    records: list[dict[str, Any]],
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> list[dict[str, Any]]:
    """
    Split article records into word-bounded passages for retrieval.

    Each input record must include ``text``, ``category``, ``source``, and
    ``date``. The ``text`` field is divided into passages of roughly 200 to
    400 words (configurable via ``min_words`` and ``max_words``). Very short
    texts are kept as a single passage rather than padded.

    Parameters
    ----------
    records:
        Extracted article records, e.g. rows from ``demo_data.json``.
    min_words:
        Minimum target size for a passage when splitting long text.
    max_words:
        Maximum size for any passage.

    Returns
    -------
    list[dict]
        Chunk dictionaries with keys ``chunk_id``, ``text``, ``category``,
        ``source``, and ``date``.
    """
    chunks: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        text = record.get("text", "")
        category = record.get("category", "")
        source = record.get("source", "")
        date = record.get("date", "")

        passages = _split_text(text, min_words=min_words, max_words=max_words)
        for chunk_index, passage in enumerate(passages):
            chunks.append(
                {
                    "chunk_id": _make_chunk_id(source, date, record_index, chunk_index),
                    "text": passage,
                    "category": category,
                    "source": source,
                    "date": date,
                }
            )

    return chunks


if __name__ == "__main__":
    sample_records = [
        {
            "category": "CLOSURES",
            "source": "foodbusinessnews.net",
            "date": "2022-06-05",
            "text": (
                "Lancaster Colony decided to exit the Bantam Bagels business, "
                "which it had acquired for $34 million in October 2018."
            ),
        },
        {
            "category": "EXPANSIONS",
            "source": "fooddive.com",
            "date": "2023-04-12",
            "text": " ".join(
                [
                    (
                        "The company announced a multi-phase expansion of its "
                        "Midwest production network, adding new lines for beverages "
                        "and frozen meals."
                    )
                ]
                * 80
            ),
        },
        {
            "category": "PRODUCT_LAUNCHES",
            "source": "foodmanufacture.co.uk",
            "date": "2024-01-15",
            "text": (
                "Danone launched HiPRO Expert, a yogurt containing protein, "
                "vitamins and minerals developed for the Paris 2024 Olympic Games."
            ),
        },
    ]

    result = chunk_records(sample_records)

    print(f"Input records: {len(sample_records)}")
    print(f"Output chunks: {len(result)}")
    for chunk in result:
        print(
            f"- {chunk['chunk_id']}: "
            f"{_word_count(chunk['text'])} words, "
            f"{chunk['category']}, {chunk['source']}, {chunk['date']}"
        )
