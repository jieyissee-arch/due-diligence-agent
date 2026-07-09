#!/usr/bin/env python3
"""
build_index.py

Build the local RAG index end-to-end:
  demo_data.json -> chunk_records -> embed_chunks -> save_embeddings -> Chroma

Run from the repo root:
    python3 src/build_index.py > build_index_log.txt 2>&1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from chunking import chunk_records
from embeddings import (
    DEFAULT_OUTPUT_FILE,
    embed_chunks,
    get_embedding_provider,
    save_embeddings,
)
from retrieval import load_chunks_into_chroma

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA = REPO_ROOT / "demo_data.json"


def main() -> None:
    if not DEMO_DATA.is_file():
        raise FileNotFoundError(f"Demo data not found: {DEMO_DATA}")

    with DEMO_DATA.open(encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise ValueError(f"{DEMO_DATA} must contain a non-empty JSON array.")

    print(f"Input records       : {len(records)}")

    chunks = chunk_records(records)
    print(f"Chunks created      : {len(chunks)}")

    provider = get_embedding_provider()
    print(f"Embedding provider  : {provider.name}")
    print(f"Embedding model     : {provider.model}")

    embedded = embed_chunks(chunks, provider=provider)
    output_path = save_embeddings(embedded, output_path=DEFAULT_OUTPUT_FILE, provider=provider)
    print(f"Embeddings saved to : {output_path}")

    summary = load_chunks_into_chroma(embedded_records=embedded, reset=True)
    print(f"Chroma collection   : {summary['collection']}")
    print(f"Chroma directory    : {summary['persist_directory']}")
    print(f"Documents loaded    : {summary['count']}")
    print("Index build complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
