"""Main orchestration layer for routing, retrieval, prompting, and generation."""

from __future__ import annotations

from src.config import ROUTE_DIRECT_LLM, ROUTE_LABELS
from src.evaluation import calculate_answer_scores
from src.llm_client import generate_llm_response
from src.prompts import build_direct_prompt, build_rag_prompt
from src.retriever import retrieve_best_context_across_routes, retrieve_context
from src.router import route_query
from src.utils import format_context_chunks, is_fallback_context


def generate_answer(query: str) -> dict:
    """
    Generate a complete chatbot response payload.

    Returns route metadata, retrieved context, and the final answer so the UI can
    transparently show how the system made its decision.
    """
    routing_result = route_query(query)
    route = routing_result.route

    retrieved_context: list[str] = []
    if route != ROUTE_DIRECT_LLM:
        retrieved_context = retrieve_context(query, route)
    else:
        semantic_route, semantic_context, semantic_reason = retrieve_best_context_across_routes(query)
        if semantic_route and semantic_context:
            route = semantic_route
            retrieved_context = semantic_context
            routing_result = type(routing_result)(
                route=route,
                reason=f"{routing_result.reason} {semantic_reason}",
            )

    route_label = ROUTE_LABELS.get(route, route)
    context_text = format_context_chunks(retrieved_context)
    answer_mode = "direct_llm"

    if route == ROUTE_DIRECT_LLM or is_fallback_context(retrieved_context):
        prompt = build_direct_prompt(query)
    else:
        prompt = build_rag_prompt(query=query, context=context_text, route_label=route_label)
        answer_mode = "document_grounded_rag"

    llm_context = [] if is_fallback_context(retrieved_context) else retrieved_context
    llm_result = generate_llm_response(
        prompt=prompt,
        query=query,
        retrieved_context=llm_context,
    )
    evaluation_scores = calculate_answer_scores(llm_result.answer, llm_context)

    return {
        "query": query,
        "route": route,
        "routing_reason": routing_result.reason,
        "retrieved_context": retrieved_context,
        "final_answer": llm_result.answer,
        "LLM_STATUS": llm_result.status,
        "ANSWER_MODE": answer_mode,
        "EVALUATION_SCORES": evaluation_scores,
        "llm_debug_error": llm_result.debug_error,
    }
