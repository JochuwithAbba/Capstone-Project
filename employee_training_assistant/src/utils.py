"""Small utility helpers used across the chatbot."""

from __future__ import annotations

from typing import Iterable


def normalize_text(text: str) -> str:
    """Normalize user text for keyword matching."""
    return " ".join((text or "").lower().strip().split())


def format_context_chunks(chunks: Iterable[str]) -> str:
    """Convert retrieved chunks into a prompt-ready context block."""
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    if not cleaned_chunks:
        return ""

    return "\n\n".join(
        f"[Context {index}]\n{chunk}" for index, chunk in enumerate(cleaned_chunks, start=1)
    )


def is_fallback_context(chunks: list[str]) -> bool:
    """Detect retrieval fallback messages so prompts can treat them as no source context."""
    if not chunks:
        return True

    fallback_prefixes = (
        "No vectorstore found",
        "No vectorstore mapping",
        "No relevant context",
        "Retrieval skipped",
        "The vectorstore exists, but retrieval could not be completed",
    )
    return all(chunk.startswith(fallback_prefixes) for chunk in chunks)
