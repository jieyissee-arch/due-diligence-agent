"""
embeddings.py

Generate embedding vectors for chunked article passages and store them in a
Chroma-ready format.

Default provider: sentence-transformers (local, free). Swap providers via
``get_embedding_provider`` or by passing a custom ``EmbeddingProvider``.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 3
SLEEP_SECS = 2
RATE_LIMIT_WAIT_SECS = 30
DEFAULT_BATCH_SIZE = 32

DEFAULT_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
DEFAULT_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)
OUTPUT_DIR = Path("output")
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "embedded_chunks.json"


class EmbeddingError(Exception):
    """Raised when embedding generation fails after all retries."""


class RateLimitError(EmbeddingError):
    """Raised when an embedding API returns a rate-limit response."""


class _call_with_retry:
    """Retry helper aligned with the backoff pattern used in agent.py."""

    def __init__(
        self,
        max_retries: int = MAX_RETRIES,
        sleep_secs: int = SLEEP_SECS,
        rate_limit_wait_secs: int = RATE_LIMIT_WAIT_SECS,
    ) -> None:
        self.max_retries = max_retries
        self.sleep_secs = sleep_secs
        self.rate_limit_wait_secs = rate_limit_wait_secs

    def run(self, fn: Callable[[], Any], description: str = "embedding call") -> Any:
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return fn()
            except RateLimitError as exc:
                last_error = exc
                wait = self.rate_limit_wait_secs * (attempt + 1)
                print(f"    Rate limit hit on {description}. Waiting {wait}s before retry...")
                time.sleep(wait)
            except EmbeddingError as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    print(f"    {description} failed (attempt {attempt + 1}): {exc}")
                    time.sleep(self.sleep_secs)
            except Exception as exc:
                last_error = EmbeddingError(f"{description} failed: {exc}")
                if attempt < self.max_retries - 1:
                    print(f"    {description} failed (attempt {attempt + 1}): {exc}")
                    time.sleep(self.sleep_secs)

        raise EmbeddingError(
            f"{description} failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )


class EmbeddingProvider(ABC):
    """Interface for swappable embedding backends."""

    name: str
    model: str
    dimension: int | None = None

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Free to run on your machine; no API key required. Good default for local
    RAG demos with Chroma.
    """

    name = "sentence-transformers"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        self.model = model
        self._model = SentenceTransformer(model)
        sample = self._model.encode(["dimension probe"], show_progress_bar=False)
        self.dimension = int(len(sample[0]))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        retry = _call_with_retry()

        def _encode_batch(batch: list[str]) -> list[list[float]]:
            vectors = self._model.encode(batch, show_progress_bar=False)
            return [vector.tolist() for vector in vectors]

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), DEFAULT_BATCH_SIZE):
            batch = texts[start : start + DEFAULT_BATCH_SIZE]
            batch_vectors = retry.run(
                lambda batch=batch: _encode_batch(batch),
                description=f"sentence-transformers batch {start // DEFAULT_BATCH_SIZE + 1}",
            )
            embeddings.extend(batch_vectors)

        return embeddings


class VoyageEmbeddingProvider(EmbeddingProvider):
    """
    Optional API provider for Voyage AI embeddings.

    Low-cost hosted option; requires VOYAGE_API_KEY in the environment.
    """

    name = "voyage"

    def __init__(
        self,
        model: str = "voyage-3-lite",
        api_key: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        import httpx

        self.model = model
        self.batch_size = batch_size
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise EmbeddingError(
                "VOYAGE_API_KEY not set. Add it to .env or choose "
                "EMBEDDING_PROVIDER=sentence-transformers for local embeddings."
            )

        self._client = httpx.Client(
            base_url="https://api.voyageai.com/v1",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        self.dimension = None

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import httpx

        response = self._client.post(
            "/embeddings",
            json={"input": texts, "model": self.model},
        )

        if response.status_code == 429:
            raise RateLimitError("Voyage AI rate limit reached")
        if response.status_code >= 500:
            raise EmbeddingError(f"Voyage AI server error: {response.status_code}")
        if response.status_code >= 400:
            raise EmbeddingError(
                f"Voyage AI request failed ({response.status_code}): {response.text}"
            )

        payload = response.json()
        data = sorted(payload["data"], key=lambda row: row["index"])
        vectors = [row["embedding"] for row in data]

        if self.dimension is None and vectors:
            self.dimension = len(vectors[0])

        return vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        retry = _call_with_retry()
        embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batch_vectors = retry.run(
                lambda batch=batch: self._embed_batch(batch),
                description=f"voyage batch {start // self.batch_size + 1}",
            )
            embeddings.extend(batch_vectors)

        return embeddings


def get_embedding_provider(
    provider_name: str | None = None,
    model: str | None = None,
) -> EmbeddingProvider:
    """
    Factory for embedding providers.

    Supported values:
    - ``sentence-transformers`` (default, local/free)
    - ``voyage`` (hosted API, requires VOYAGE_API_KEY)
    """
    selected = (provider_name or DEFAULT_PROVIDER).strip().lower()
    selected_model = model or DEFAULT_MODEL

    if selected in {"sentence-transformers", "local", "minilm"}:
        return SentenceTransformerProvider(model=selected_model)

    if selected == "voyage":
        return VoyageEmbeddingProvider(model=selected_model or "voyage-3-lite")

    raise EmbeddingError(
        f"Unknown embedding provider '{selected}'. "
        "Use 'sentence-transformers' or 'voyage'."
    )


def _chunk_metadata(chunk: dict[str, Any]) -> dict[str, str]:
    return {
        "category": str(chunk.get("category", "")),
        "source": str(chunk.get("source", "")),
        "date": str(chunk.get("date", "")),
    }


def embed_chunks(
    chunks: list[dict[str, Any]],
    provider: EmbeddingProvider | None = None,
) -> list[dict[str, Any]]:
    """
    Generate embeddings for chunked article passages.

    Parameters
    ----------
    chunks:
        Output from ``chunk_records`` in chunking.py. Each chunk must include
        ``chunk_id`` and ``text`` plus metadata fields.
    provider:
        Optional embedding backend. Defaults to ``get_embedding_provider()``.

    Returns
    -------
    list[dict]
        Records with ``id``, ``document``, ``embedding``, and ``metadata`` —
        ready to load into Chroma.
    """
    if not chunks:
        return []

    backend = provider or get_embedding_provider()
    texts = [chunk["text"] for chunk in chunks]
    vectors = backend.embed_texts(texts)

    if len(vectors) != len(chunks):
        raise EmbeddingError(
            f"Expected {len(chunks)} embeddings, received {len(vectors)}."
        )

    embedded_records: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors):
        embedded_records.append(
            {
                "id": chunk["chunk_id"],
                "document": chunk["text"],
                "embedding": vector,
                "metadata": _chunk_metadata(chunk),
            }
        )

    return embedded_records


def to_chroma_payload(records: Iterable[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Convert embedded records into the keyword arguments expected by
    ``collection.add(...)`` in Chroma.
    """
    records = list(records)
    return {
        "ids": [record["id"] for record in records],
        "embeddings": [record["embedding"] for record in records],
        "documents": [record["document"] for record in records],
        "metadatas": [record["metadata"] for record in records],
    }


def save_embeddings(
    records: list[dict[str, Any]],
    output_path: Path = DEFAULT_OUTPUT_FILE,
    provider: EmbeddingProvider | None = None,
) -> Path:
    """
    Persist embedded chunks as JSON for later loading into Chroma.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backend = provider or get_embedding_provider()

    payload = {
        "provider": backend.name,
        "model": backend.model,
        "embedding_dimension": backend.dimension,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "count": len(records),
        "records": records,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    return output_path


def load_embeddings(path: Path = DEFAULT_OUTPUT_FILE) -> dict[str, Any]:
    """Load a saved embedding file and return records plus Chroma payload."""
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    records = payload["records"]
    return {
        "meta": {
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "embedding_dimension": payload.get("embedding_dimension"),
            "created_at": payload.get("created_at"),
            "count": payload.get("count", len(records)),
        },
        "records": records,
        "chroma": to_chroma_payload(records),
    }


if __name__ == "__main__":
    from chunking import chunk_records

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
            "category": "PRODUCT_LAUNCHES",
            "source": "foodmanufacture.co.uk",
            "date": "2024-01-15",
            "text": (
                "Danone launched HiPRO Expert, a yogurt containing protein, "
                "vitamins and minerals developed for the Paris 2024 Olympic Games."
            ),
        },
    ]

    chunks = chunk_records(sample_records)
    provider = get_embedding_provider()
    embedded = embed_chunks(chunks, provider=provider)
    output_path = save_embeddings(embedded, provider=provider)
    chroma_payload = to_chroma_payload(embedded)

    print(f"Provider            : {provider.name}")
    print(f"Model               : {provider.model}")
    print(f"Embedding dimension : {provider.dimension}")
    print(f"Embedded chunks     : {len(embedded)}")
    print(f"Saved to            : {output_path}")
    print(f"Chroma ids          : {chroma_payload['ids']}")
