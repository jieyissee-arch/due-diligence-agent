"""
retrieval.py

Local Chroma-backed retrieval for embedded article chunks.

Initialises a persistent Chroma collection under the repo, loads embedding
records produced by embeddings.py, and exposes ``retrieve(query, top_k=5)``.
Queries are embedded with the same provider used at index time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from embeddings import (
    EmbeddingProvider,
    get_embedding_provider,
    load_embeddings,
    to_chroma_payload,
)

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(REPO_ROOT / "chroma_db")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "due_diligence_chunks")
DEFAULT_TOP_K = 5

# Prefer repo-root output; fall back to src/output from earlier test runs.
_CANDIDATE_EMBEDDING_FILES = (
    REPO_ROOT / "output" / "embedded_chunks.json",
    SRC_DIR / "output" / "embedded_chunks.json",
)


class RetrievalError(Exception):
    """Raised when the local vector store cannot be initialised or queried."""


def _resolve_embedding_path(path: Path | None = None) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise RetrievalError(f"Embedding file not found: {resolved}")
        return resolved

    for candidate in _CANDIDATE_EMBEDDING_FILES:
        if candidate.is_file():
            return candidate.resolve()

    raise RetrievalError(
        "No embedded_chunks.json found. Run embeddings.py first, or pass "
        "embedded_records / embeddings_path explicitly."
    )


def _cosine_similarity(distance: float) -> float:
    """Convert Chroma cosine distance (1 - cos) into a similarity score."""
    return 1.0 - float(distance)


def validate_embedding_config(
    meta: dict[str, Any],
    provider: EmbeddingProvider | None = None,
) -> None:
    """
    Ensure the active embedding provider matches the saved index metadata.

    Raises RetrievalError when provider or model differs from index time.
    """
    backend = provider or get_embedding_provider()
    saved_provider = meta.get("provider")
    saved_model = meta.get("model")

    if saved_provider and saved_provider != backend.name:
        raise RetrievalError(
            f"Embedding provider mismatch: index was built with "
            f"'{saved_provider}' but EMBEDDING_PROVIDER is '{backend.name}'. "
            "Rebuild the index or update your environment."
        )

    if saved_model and saved_model != backend.model:
        raise RetrievalError(
            f"Embedding model mismatch: index was built with "
            f"'{saved_model}' but EMBEDDING_MODEL is '{backend.model}'. "
            "Rebuild the index or update your environment."
        )


def _ensure_sqlite() -> None:
    """
    On older macOS/Python builds, Chroma needs sqlite3 >= 3.35.

    Prefer the system sqlite3 when new enough; otherwise swap in
    pysqlite3-binary (see Chroma troubleshooting docs).
    """
    import sqlite3
    import sys

    try:
        major_minor = tuple(int(part) for part in sqlite3.sqlite_version.split(".")[:2])
    except ValueError:
        major_minor = (0, 0)

    if major_minor >= (3, 35):
        return

    try:
        import pysqlite3  # type: ignore
    except ImportError as exc:
        raise RetrievalError(
            "Chroma requires sqlite3 >= 3.35.0. Install a workaround with:\n"
            "  pip install pysqlite3-binary\n"
            "See https://docs.trychroma.com/troubleshooting#sqlite"
        ) from exc

    sys.modules["sqlite3"] = pysqlite3


def _get_chroma_client(persist_directory: Path = CHROMA_DIR):
    _ensure_sqlite()
    try:
        import chromadb
    except ImportError as exc:
        raise RetrievalError(
            "chromadb is not installed. Run: pip install chromadb"
        ) from exc

    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_directory))


def init_collection(
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
):
    """
    Initialise (or open) the local Chroma collection.

    Uses cosine space so retrieved distances can be mapped to similarity scores.
    """
    client = _get_chroma_client(persist_directory)

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def load_chunks_into_chroma(
    embedded_records: list[dict[str, Any]] | None = None,
    embeddings_path: Path | None = None,
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    reset: bool = True,
) -> dict[str, Any]:
    """
    Load Chroma-ready embedding records into a local persistent collection.

    Parameters
    ----------
    embedded_records:
        Records from ``embed_chunks`` / ``save_embeddings``. If omitted, loads
        from ``embeddings_path`` or the default embedded_chunks.json location.
    embeddings_path:
        Optional path to a saved embedding JSON file.
    persist_directory:
        Local folder for the Chroma database (inside the repo by default).
    collection_name:
        Chroma collection name.
    reset:
        If True, replace any existing collection contents.

    Returns
    -------
    dict
        Summary with collection name, path, and number of documents loaded.
    """
    if embedded_records is None:
        loaded = load_embeddings(_resolve_embedding_path(embeddings_path))
        validate_embedding_config(loaded["meta"])
        embedded_records = loaded["records"]

    if not embedded_records:
        raise RetrievalError("No embedded records to load into Chroma.")

    collection = init_collection(
        persist_directory=persist_directory,
        collection_name=collection_name,
        reset=reset,
    )
    payload = to_chroma_payload(embedded_records)

    # upsert keeps ids unique if reset=False and records are reloaded
    collection.upsert(
        ids=payload["ids"],
        embeddings=payload["embeddings"],
        documents=payload["documents"],
        metadatas=payload["metadatas"],
    )

    return {
        "collection": collection_name,
        "persist_directory": str(persist_directory.resolve()),
        "count": len(payload["ids"]),
    }


class Retriever:
    """Stateful helper that holds a Chroma collection and embedding provider."""

    def __init__(
        self,
        collection=None,
        provider: EmbeddingProvider | None = None,
        persist_directory: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.provider = provider or get_embedding_provider()
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.collection = collection or init_collection(
            persist_directory=persist_directory,
            collection_name=collection_name,
            reset=False,
        )

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """
        Embed ``query`` and return the top_k nearest chunks from Chroma.

        Each result includes ``text``, ``category``, ``source``, ``date``,
        ``chunk_id``, ``distance``, and ``similarity``.
        """
        return retrieve(
            query,
            top_k=top_k,
            collection=self.collection,
            provider=self.provider,
        )


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    collection=None,
    provider: EmbeddingProvider | None = None,
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, Any]]:
    """
    Search the local Chroma collection for passages nearest to ``query``.

    The query is embedded with the same embedding provider used for document
    chunks. Returned items include category, source, date, and similarity score.

    Parameters
    ----------
    query:
        Natural-language search string.
    top_k:
        Maximum number of matches to return.
    collection:
        Optional existing Chroma collection. Opens the local DB if omitted.
    provider:
        Embedding backend. Defaults to ``get_embedding_provider()``.
    persist_directory:
        Local Chroma persistence folder.
    collection_name:
        Collection to query.

    Returns
    -------
    list[dict]
        Ranked matches with keys ``chunk_id``, ``text``, ``category``,
        ``source``, ``date``, ``distance``, and ``similarity``.
    """
    if not query or not query.strip():
        raise RetrievalError("Query must be a non-empty string.")

    if top_k < 1:
        raise RetrievalError("top_k must be >= 1.")

    backend = provider or get_embedding_provider()
    coll = collection or init_collection(
        persist_directory=persist_directory,
        collection_name=collection_name,
        reset=False,
    )

    if coll.count() == 0:
        raise RetrievalError(
            f"Collection '{collection_name}' is empty. "
            "Call load_chunks_into_chroma() first."
        )

    n_results = min(top_k, coll.count())
    query_vector = backend.embed_texts([query.strip()])[0]

    raw = coll.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    results: list[dict[str, Any]] = []
    for chunk_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        meta = metadata or {}
        results.append(
            {
                "chunk_id": chunk_id,
                "text": document,
                "category": meta.get("category", ""),
                "source": meta.get("source", ""),
                "date": meta.get("date", ""),
                "distance": float(distance),
                "similarity": _cosine_similarity(distance),
            }
        )

    return results


def build_retriever(
    embeddings_path: Path | None = None,
    embedded_records: list[dict[str, Any]] | None = None,
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    provider: EmbeddingProvider | None = None,
    reset: bool = True,
) -> Retriever:
    """Load embeddings into Chroma and return a ready-to-query Retriever."""
    load_chunks_into_chroma(
        embedded_records=embedded_records,
        embeddings_path=embeddings_path,
        persist_directory=persist_directory,
        collection_name=collection_name,
        reset=reset,
    )
    return Retriever(
        provider=provider,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


if __name__ == "__main__":
    summary = load_chunks_into_chroma(reset=True)
    print(f"Loaded chunks     : {summary['count']}")
    print(f"Collection        : {summary['collection']}")
    print(f"Persist directory : {summary['persist_directory']}")

    matches = retrieve("factory closure bakery shutdown", top_k=3)
    print(f"Query matches     : {len(matches)}")
    for i, match in enumerate(matches, 1):
        preview = (match["text"] or "")[:80].replace("\n", " ")
        print(
            f"{i}. similarity={match['similarity']:.4f} "
            f"category={match['category']} "
            f"source={match['source']} "
            f"date={match['date']}"
        )
        print(f"   {preview}...")
