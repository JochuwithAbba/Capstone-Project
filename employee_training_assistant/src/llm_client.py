"""LLM client wrapper using Groq with safe fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.config import get_groq_api_key, get_groq_model_name, missing_api_key_message
from src.utils import normalize_text


FALLBACK_NO_CONTEXT_MESSAGE = (
    "The assistant could not access the external LLM at the moment. "
    "Please configure a valid API key or try again later."
)


@dataclass(frozen=True)
class LLMResult:
    """Structured LLM result used by the orchestration layer."""

    answer: str
    status: str
    debug_error: str | None = None


class GroqLLMClient:
    """Small adapter around Groq so the rest of the app stays provider-neutral."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        self.api_key = api_key or get_groq_api_key()
        self.model_name = model_name or get_groq_model_name()
        self._client = self._create_client() if self.api_key else None

    @property
    def is_configured(self) -> bool:
        """Return True when an API key is available."""
        return self._client is not None

    def generate_text(self, prompt: str) -> str:
        """Generate text using the configured Groq chat model."""
        if not self._client:
            raise RuntimeError(missing_api_key_message())

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a clear, professional AI training assistant for new employees.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        return (response.choices[0].message.content or "").strip() or "The model returned an empty response."

    def _create_client(self):
        """Create the Groq client lazily so missing packages do not crash the UI."""
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "groq is not installed. Run 'pip install -r requirements.txt'."
            ) from exc

        return Groq(api_key=self.api_key)


def generate_llm_response(
    prompt: str,
    query: str,
    retrieved_context: list[str] | None = None,
) -> LLMResult:
    """
    Generate a response using Groq.

    Any Groq error, including missing key, permission errors, model errors,
    quota errors, and transport exceptions, falls back to a clean answer.
    Error details are returned only in debug_error.
    """
    context_chunks = retrieved_context or []
    try:
        answer = GroqLLMClient().generate_text(prompt)
        return LLMResult(answer=answer, status="groq_success")
    except Exception as exc:
        debug_error = _format_debug_error(exc)
        fallback_answer = _build_context_fallback_answer(query, context_chunks)
        if fallback_answer:
            return LLMResult(
                answer=fallback_answer,
                status="fallback_context_answer",
                debug_error=debug_error,
            )

        return LLMResult(
            answer=FALLBACK_NO_CONTEXT_MESSAGE,
            status="fallback_no_context",
            debug_error=debug_error,
        )


def _format_debug_error(error: Exception) -> str:
    """Create a compact debug-only error string."""
    message = str(error).strip() or error.__class__.__name__
    return f"{error.__class__.__name__}: {message}"


def _build_context_fallback_answer(query: str, context_chunks: list[str]) -> str | None:
    """
    Build a concise answer from retrieved context without calling an external LLM.

    This is intentionally simple and explainable for a capstone prototype: split
    context into sentences, score them by query-term overlap, and return the most
    relevant lines.
    """
    cleaned_context = [chunk.strip() for chunk in context_chunks if chunk and chunk.strip()]
    if not cleaned_context:
        return None

    sentences = _split_context_into_sentences(cleaned_context)
    if not sentences:
        return None

    query_terms = _important_terms(query)
    ranked_sentences = sorted(
        sentences,
        key=lambda sentence: _sentence_score(sentence, query_terms),
        reverse=True,
    )
    selected = [sentence for sentence in ranked_sentences[:3] if sentence]
    if not selected:
        return None

    answer_body = " ".join(selected)
    return (
        "Based on the retrieved company documents, here is the most relevant information I found: "
        f"{answer_body}"
    )


def _split_context_into_sentences(context_chunks: list[str]) -> list[str]:
    """Split retrieved chunks into readable candidate sentences."""
    text = " ".join(context_chunks)
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip(" -") for part in parts if len(part.strip()) > 20]


def _important_terms(query: str) -> set[str]:
    """Extract lightweight keywords from the employee question."""
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "can",
        "do",
        "does",
        "for",
        "how",
        "i",
        "if",
        "in",
        "is",
        "me",
        "my",
        "of",
        "on",
        "or",
        "should",
        "the",
        "to",
        "what",
        "when",
        "where",
        "who",
        "why",
    }
    normalized = normalize_text(query)
    return {term for term in re.findall(r"[a-z0-9]+", normalized) if len(term) > 2 and term not in stopwords}


def _sentence_score(sentence: str, query_terms: set[str]) -> int:
    """Score sentence relevance using keyword overlap."""
    if not query_terms:
        return 0

    sentence_terms = set(re.findall(r"[a-z0-9]+", normalize_text(sentence)))
    return len(query_terms.intersection(sentence_terms))
