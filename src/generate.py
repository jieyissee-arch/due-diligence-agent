"""
generate.py

RAG generation layer: retrieve supporting chunks, prompt Claude, and return a
traceable answer with the source passages used.
"""

from __future__ import annotations

import os
import time
from typing import Any

import anthropic
import httpx
from dotenv import load_dotenv

from retrieval import DEFAULT_TOP_K, retrieve

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = int(os.getenv("GENERATE_MAX_TOKENS", "700"))
MAX_RETRIES = 3
SLEEP_SECS = 2
RATE_LIMIT_WAIT_SECS = 30

SYSTEM_PROMPT = """You are a due diligence analyst answering questions about
food manufacturing and packaging industry events.

Use only the provided source passages. Do not invent facts that are not
supported by those passages. If the passages are insufficient, say what is
missing.

In your answer, cite each claim with the passage label from the prompt
(for example: [Passage 1]), together with the source name and date
(for example: [Passage 1] foodbusinessnews.net, 2022-06-05).

At the end, add a short "Sources used:" section listing every passage
label you relied on (for example: [Passage 1], [Passage 3]).

Prefer concise, factual prose suitable for an analyst briefing."""


class GenerationError(Exception):
    """Raised when Claude generation fails after retries."""


def _format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No source passages were retrieved."

    blocks: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        blocks.append(
            "\n".join(
                [
                    f"[Passage {i}]",
                    f"source: {chunk.get('source', '')}",
                    f"date: {chunk.get('date', '')}",
                    f"category: {chunk.get('category', '')}",
                    f"similarity: {chunk.get('similarity', '')}",
                    f"text: {chunk.get('text', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Build the user prompt with the question and retrieved chunk metadata."""
    return (
        "Answer the question using only the source passages below.\n"
        "Cite each claim with the passage label, source, and date "
        "(for example: [Passage 1] foodbusinessnews.net, 2022-06-05).\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Source passages:\n{_format_chunks_for_prompt(chunks)}"
    )


def _get_client(api_key: str | None = None) -> anthropic.Anthropic:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise GenerationError(
            "ANTHROPIC_API_KEY not set. "
            "Copy .env.example to .env and add your key."
        )
    # Cursor's sandboxed terminal injects a local proxy (127.0.0.1:507xx) that
    # returns 403 for api.anthropic.com. Bypass it with trust_env=False so we
    # connect directly. For real corporate proxies (non-localhost) we keep the
    # proxy in the loop by leaving trust_env at its default (True).
    proxy_url = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
        or ""
    )
    is_local_proxy = "127.0.0.1" in proxy_url or "localhost" in proxy_url
    return anthropic.Anthropic(
        api_key=key,
        http_client=httpx.Client(
            trust_env=not is_local_proxy,
            timeout=httpx.Timeout(30.0),
        ),
    )


def _call_claude(
    prompt: str,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Call Claude with the same retry/backoff pattern as agent.py."""
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
            print(f"    Rate limit hit. Waiting {wait}s before retry...")
            time.sleep(wait)

        except anthropic.APIError as exc:
            last_error = GenerationError(f"Anthropic API error: {exc}")
            if attempt < MAX_RETRIES - 1:
                print(f"    API error (attempt {attempt + 1}): {exc}")
                time.sleep(SLEEP_SECS)

        except Exception as exc:
            last_error = GenerationError(f"Generation failed: {exc}")
            if attempt < MAX_RETRIES - 1:
                print(f"    Generation failed (attempt {attempt + 1}): {exc}")
                time.sleep(SLEEP_SECS)

    raise GenerationError(
        f"Claude generation failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def generate_answer(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Retrieve supporting chunks and generate a Claude answer.

    Parameters
    ----------
    question:
        User question to answer from the local vector store.
    top_k:
        Number of passages to retrieve (ignored if ``chunks`` is provided).
    client:
        Optional Anthropic client. Uses ``ANTHROPIC_API_KEY`` if omitted.
    model:
        Claude model id.
    max_tokens:
        Maximum completion tokens.
    chunks:
        Optional pre-retrieved chunks. If omitted, calls ``retrieve()``.

    Returns
    -------
    dict
        ``answer``: generated text, and ``chunks_used``: retrieved passages
        with ``category``, ``source``, ``date``, ``text``, and similarity.
    """
    if not question or not question.strip():
        raise GenerationError("Question must be a non-empty string.")

    used_chunks = chunks if chunks is not None else retrieve(question, top_k=top_k)
    prompt = build_prompt(question, used_chunks)
    answer = _call_claude(
        prompt,
        client=client,
        model=model,
        max_tokens=max_tokens,
    )

    return {
        "answer": answer,
        "chunks_used": used_chunks,
        "model": model,
        "prompt": prompt,
    }


if __name__ == "__main__":
    sample_question = (
        "What bakery or food facility closures are mentioned in the sources?"
    )

    # Offline smoke path when ANTHROPIC_API_KEY is unset: exercise retrieval +
    # prompt assembly without calling Claude.
    if not os.getenv("ANTHROPIC_API_KEY"):
        chunks = retrieve(sample_question, top_k=3)
        prompt = build_prompt(sample_question, chunks)
        print("Mode         : offline (no ANTHROPIC_API_KEY)")
        print(f"Chunks used  : {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(
                f"  {i}. {chunk.get('source')} | {chunk.get('date')} | "
                f"{chunk.get('category')} | similarity={chunk.get('similarity', 0):.4f}"
            )
        print(f"Prompt chars : {len(prompt)}")
        print("Set ANTHROPIC_API_KEY in .env to run live Claude generation.")
    else:
        result = generate_answer(sample_question, top_k=3)
        print(f"Model        : {result['model']}")
        print(f"Chunks used  : {len(result['chunks_used'])}")
        for i, chunk in enumerate(result["chunks_used"], 1):
            print(
                f"  {i}. {chunk.get('source')} | {chunk.get('date')} | "
                f"{chunk.get('category')} | similarity={chunk.get('similarity', 0):.4f}"
            )
        print("Answer preview:")
        preview = result["answer"][:400].replace("\n", " ")
        print(f"  {preview}...")
