"""Flexible retrieval adapter for FAISS/vectorstore-backed RAG."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.config import (
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_TOP_K,
    ROUTE_DIRECT_LLM,
    ROUTE_LABELS,
    VECTORSTORE_PATHS,
)


SEMANTIC_ROUTE_SCORE_THRESHOLD = 1.35


def retrieve_context(query: str, route: str) -> list[str]:
    """
    Retrieve relevant context for a query and route.

    The function expects an existing LangChain FAISS store in the mapped
    vectorstore folder. If the store is missing or cannot be loaded, it returns
    a helpful fallback message instead of raising an exception.
    """
    if route == ROUTE_DIRECT_LLM:
        return ["Retrieval skipped because the query was routed to direct LLM response."]

    vectorstore_path = VECTORSTORE_PATHS.get(route)
    if not vectorstore_path:
        return [f"No vectorstore mapping is configured for route: {route}."]

    if not _looks_like_faiss_store(vectorstore_path):
        return [
            f"No vectorstore found for route '{route}' at '{vectorstore_path}'. "
            "Add a FAISS index to this folder to enable document-grounded answers."
        ]

    try:
        vectorstore = _load_vectorstore(str(vectorstore_path))
        documents = vectorstore.similarity_search(query, k=RETRIEVAL_TOP_K)
        if not documents:
            return ["No relevant context was found in the selected knowledge base."]

        return [_format_document_context(document) for document in documents]
    except Exception as exc:
        return [
            "The vectorstore exists, but retrieval could not be completed. "
            f"Reason: {exc}"
        ]


def retrieve_best_context_across_routes(query: str) -> tuple[str | None, list[str], str]:
    """
    Search every available vectorstore and return the best document route.

    This is used when keyword routing would otherwise send a question directly
    to the LLM. It gives uploaded PDFs a chance to answer ambiguous questions.
    """
    best_route: str | None = None
    best_documents = []
    best_score: float | None = None

    for route, vectorstore_path in VECTORSTORE_PATHS.items():
        if not _looks_like_faiss_store(vectorstore_path):
            continue

        documents_with_scores = _similarity_search_with_scores(query, vectorstore_path)
        if not documents_with_scores:
            continue

        top_document, top_score = documents_with_scores[0]
        if best_score is None or top_score < best_score:
            best_route = route
            best_score = top_score
            best_documents = [document for document, _score in documents_with_scores]

    if best_route is None or best_score is None:
        return None, [], "No FAISS vectorstores were available for semantic document search."

    if best_score > SEMANTIC_ROUTE_SCORE_THRESHOLD:
        return (
            None,
            [],
            f"Semantic document search found no close match. Best score was {best_score:.3f}.",
        )

    context_chunks = [_format_document_context(document) for document in best_documents]
    route_label = ROUTE_LABELS.get(best_route, best_route)
    reason = (
        f"Keyword routing did not find a document route, but semantic search found "
        f"a close match in {route_label} with score {best_score:.3f}."
    )
    return best_route, context_chunks, reason


def _similarity_search_with_scores(query: str, vectorstore_path: Path):
    """Run FAISS similarity search with scores for one vectorstore."""
    try:
        vectorstore = _load_vectorstore(str(vectorstore_path))
        return vectorstore.similarity_search_with_score(query, k=RETRIEVAL_TOP_K)
    except Exception:
        return []


@lru_cache(maxsize=1)
def _get_embeddings():
    """Load the embedding model once per process."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=8)
def _load_vectorstore(vectorstore_path: str):
    """Load a FAISS vectorstore once per process."""
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(
        vectorstore_path,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def _format_document_context(document) -> str:
    """Attach source metadata to a retrieved document chunk."""
    source = document.metadata.get("source", "Unknown source")
    page = document.metadata.get("page")
    source_label = f"{source}, page {page + 1}" if isinstance(page, int) else source
    return f"Source: {source_label}\n{document.page_content}"


def _looks_like_faiss_store(path: Path) -> bool:
    """Check for files normally created by LangChain FAISS.save_local()."""
    return path.exists() and (path / "index.faiss").exists() and (path / "index.pkl").exists()
