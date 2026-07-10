"""Unit tests for eval_scoring helpers."""

from __future__ import annotations

from eval_scoring import (
    citation_present,
    citation_valid,
    extract_cited_passage_indices,
    grounded_terms_in_cited_passages,
    has_sources_used_section,
    looks_like_abstention,
    retrieval_category_hit,
    retrieval_term_hit,
    terms_in_texts,
)


def test_extract_cited_passage_indices():
    answer = "A closure occurred [Passage 1] and expansion [Passage 3]."
    assert extract_cited_passage_indices(answer) == {1, 3}


def test_citation_valid_accepts_in_range():
    answer = "Claim [Passage 1] and [Passage 2]."
    assert citation_valid(answer, 2) is True


def test_citation_valid_rejects_out_of_range():
    answer = "Claim [Passage 4]."
    assert citation_valid(answer, 3) is False


def test_citation_present_requires_label():
    assert citation_present("No labels here.") is False
    assert citation_present("Cited [Passage 1].") is True


def test_sources_used_section():
    assert has_sources_used_section("Sources used: [Passage 1]") is True
    assert has_sources_used_section("No footer.") is False


def test_looks_like_abstention():
    assert looks_like_abstention("The passages are insufficient to answer.") is True
    assert looks_like_abstention("Two closures were reported [Passage 1].") is False


def test_terms_in_texts_any_match():
    assert terms_in_texts(["A bakery closure in Ohio"], ["bakery", "dividend"]) is True
    assert terms_in_texts(["Quarterly EPS guidance"], ["bakery"]) is False


def test_retrieval_category_hit():
    chunks = [{"category": "CLOSURES", "text": "shutdown"}]
    assert retrieval_category_hit(chunks, ["CLOSURES"]) is True
    assert retrieval_category_hit(chunks, ["EXPANSIONS"]) is False
    assert retrieval_category_hit(chunks, []) is True


def test_retrieval_term_hit():
    chunks = [{"category": "CLOSURES", "text": "bakery plant shutdown"}]
    assert retrieval_term_hit(chunks, ["bakery"]) is True
    assert retrieval_term_hit(chunks, ["dividend"]) is False


def test_grounded_terms_in_cited_passages():
    chunks = [
        {"text": "A bakery closed in Manchester."},
        {"text": "An unrelated expansion."},
    ]
    answer = "A bakery closed [Passage 1] foodmanufacture.co.uk, 2025-01-01."
    assert grounded_terms_in_cited_passages(answer, chunks, ["bakery"]) is True
    assert grounded_terms_in_cited_passages(answer, chunks, ["dividend"]) is False
