"""
chat.py

History-aware RAG drill-down chat grounded in retrieved chunks and the
cached insights briefing.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import anthropic
from dotenv import load_dotenv

from generate import MODEL, MAX_TOKENS, _get_client
from insights import load_briefing
from retrieval import DEFAULT_TOP_K, retrieve

load_dotenv()

MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "6"))
MAX_RETRIES = 3
SLEEP_SECS = 2
RATE_LIMIT_WAIT_SECS = 30

ChatHistory = list[list[Optional[str]]]


class ChatError(Exception):
    """Raised when the analyst chat session fails."""


def build_system_prompt(briefing: dict[str, Any] | None = None) -> str:
    briefing = briefing or load_briefing() or {}
    trends = briefing.get("trends") or briefing.get("narrative") or "_No briefing loaded._"
    patterns = briefing.get("patterns") or ""

    return f"""You are a due diligence analyst in a drill-down chat session about
food manufacturing and packaging industry events.

A prior corpus analysis produced this briefing:

{trends}

{patterns}

Use the retrieved source passages provided with each user question to answer.
Cite claims with passage labels, source, and date
(for example: [Passage 1] foodbusinessnews.net, 2022-06-05).

Rules:
- Use only the retrieved passages and conversation context; do not invent facts.
- If passages are insufficient, say what is missing and suggest a narrower question.
- Keep answers concise and suitable for an analyst workflow.
- Reference the briefing themes when they help frame the answer."""


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


def _history_to_messages(history: ChatHistory) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    trimmed = history[-MAX_HISTORY_TURNS:] if history else []

    for turn in trimmed:
        if not turn or len(turn) != 2:
            continue
        user_msg, assistant_msg = turn
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    return messages


def build_user_message(question: str, chunks: list[dict[str, Any]]) -> str:
    return (
        "Answer the question using only the source passages below.\n"
        "Cite each claim with the passage label, source, and date.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Source passages:\n{_format_chunks_for_prompt(chunks)}"
    )


def _call_claude_chat(
    messages: list[dict[str, str]],
    system: str,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
) -> str:
    api_client = client or _get_client()
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = api_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            return response.content[0].text.strip()

        except anthropic.RateLimitError as exc:
            last_error = exc
            wait = RATE_LIMIT_WAIT_SECS * (attempt + 1)
            print(f"    Rate limit hit. Waiting {wait}s before retry...")
            time.sleep(wait)

        except anthropic.APIError as exc:
            last_error = ChatError(f"Anthropic API error: {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(SLEEP_SECS)

        except Exception as exc:
            last_error = ChatError(f"Chat generation failed: {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(SLEEP_SECS)

    raise ChatError(
        f"Claude chat failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def chat_turn(
    message: str,
    history: ChatHistory | None = None,
    briefing: dict[str, Any] | None = None,
    top_k: int = DEFAULT_TOP_K,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
) -> dict[str, Any]:
    """
    Run one analyst chat turn with retrieval and optional conversation history.

    Returns
    -------
    dict
        ``answer``, ``chunks_used``, ``history`` (updated Gradio tuple history),
        and ``model``.
    """
    if not message or not message.strip():
        raise ChatError("Message must be a non-empty string.")

    history = list(history or [])
    chunks = retrieve(message.strip(), top_k=top_k)
    system = build_system_prompt(briefing)

    messages = _history_to_messages(history)
    messages.append(
        {
            "role": "user",
            "content": build_user_message(message.strip(), chunks),
        }
    )

    if not os.getenv("ANTHROPIC_API_KEY"):
        answer = (
            "_Chat unavailable offline._ Set `ANTHROPIC_API_KEY` to run live "
            "drill-down answers. Retrieved passages are still shown below."
        )
    else:
        answer = _call_claude_chat(
            messages,
            system=system,
            client=client,
            model=model,
        )

    updated_history = history + [[message.strip(), answer]]
    return {
        "answer": answer,
        "chunks_used": chunks,
        "history": updated_history,
        "model": model,
    }


if __name__ == "__main__":
    from retrieval import retrieve

    sample_question = "What packaging sustainability moves appear in the sources?"
    chunks = retrieve(sample_question, top_k=3)
    print(f"Chunks used  : {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(
            f"  {i}. {chunk.get('source')} | {chunk.get('date')} | "
            f"{chunk.get('category')} | similarity={chunk.get('similarity', 0):.4f}"
        )

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Mode         : offline (no ANTHROPIC_API_KEY)")
        print("Set ANTHROPIC_API_KEY in .env to run live chat.")
    else:
        try:
            result = chat_turn(sample_question, history=[])
            print(f"Model        : {result['model']}")
            print("Answer preview:")
            print(result["answer"][:400])
        except ChatError as exc:
            print(f"Chat error   : {exc}")
