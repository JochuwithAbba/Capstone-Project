"""Explainable query router for selecting the right answer path."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    get_groq_api_key,
    get_groq_model_name,
    ROUTE_ADMIN_POLICIES,
    ROUTE_COMPANY_OVERVIEW,
    ROUTE_DIRECT_LLM,
    ROUTE_ROLE_DOCUMENTS,
)
from src.utils import normalize_text


@dataclass(frozen=True)
class RoutingResult:
    """Structured result returned by the router."""

    route: str
    reason: str


KEYWORD_GROUPS = {
    ROUTE_COMPANY_OVERVIEW: [
        "company mission",
        "mission",
        "vision",
        "values",
        "working hours",
        "departments",
        "department",
        "organization structure",
        "organisational structure",
        "organizational structure",
        "company background",
        "company overview",
        "about the company",
        "culture",
    ],
    ROUTE_ROLE_DOCUMENTS: [
        "my role",
        "role",
        "team responsibility",
        "responsibilities",
        "responsibility",
        "sales executive duties",
        "duties",
        "trainer responsibility",
        "trainer",
        "hr role",
        "manager expectations",
        "reporting structure",
        "team",
        "manager",
    ],
    ROUTE_ADMIN_POLICIES: [
        "leave",
        "expense",
        "reimbursement",
        "payroll",
        "salary",
        "attendance",
        "code of conduct",
        "it policy",
        "laptop",
        "security",
        "password",
        "onboarding process",
        "policy",
        "admin",
    ],
}


def route_query(query: str) -> RoutingResult:
    """
    Route a user query using reliable keyword rules first.

    If no keyword confidently matches and GROQ_API_KEY is available, the router
    asks Groq to classify the query. If that also fails, the query goes to the
    direct LLM route.
    """
    normalized_query = normalize_text(query)

    keyword_result = _keyword_route(normalized_query)
    if keyword_result:
        return keyword_result

    llm_result = _llm_fallback_route(query)
    if llm_result:
        return llm_result

    return RoutingResult(
        route=ROUTE_DIRECT_LLM,
        reason="No company, role, or administrative keywords were detected, so semantic document search will be checked before using a direct LLM response.",
    )


def _keyword_route(normalized_query: str) -> RoutingResult | None:
    """Return the highest keyword-match route, if any route matches."""
    scores: dict[str, list[str]] = {}

    for route, keywords in KEYWORD_GROUPS.items():
        matches = [keyword for keyword in keywords if keyword in normalized_query]
        if matches:
            scores[route] = matches

    if not scores:
        return None

    selected_route = max(scores, key=lambda route: len(scores[route]))
    matched_terms = ", ".join(scores[selected_route][:5])
    return RoutingResult(
        route=selected_route,
        reason=f"Keyword routing selected this path because the query matched: {matched_terms}.",
    )


def _llm_fallback_route(query: str) -> RoutingResult | None:
    """Classify ambiguous queries with Groq when an API key is configured."""
    api_key = get_groq_api_key()
    if not api_key:
        return None

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        prompt = f"""
Classify this employee question into exactly one route:
- company_overview
- role_documents
- admin_policies
- direct_llm

Question: {query}

Return only the route name.
"""
        response = client.chat.completions.create(
            model=get_groq_model_name(),
            messages=[
                {"role": "system", "content": "Return only one route name."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=20,
        )
        route = (response.choices[0].message.content or "").strip().lower()
        if route in {
            ROUTE_COMPANY_OVERVIEW,
            ROUTE_ROLE_DOCUMENTS,
            ROUTE_ADMIN_POLICIES,
            ROUTE_DIRECT_LLM,
        }:
            return RoutingResult(
                route=route,
                reason="No strong keyword match was found, so the optional Groq fallback classifier selected this route.",
            )
    except Exception:
        return None

    return None
