"""Prompt templates for RAG and direct LLM answer generation."""

from __future__ import annotations


def build_rag_prompt(query: str, context: str, route_label: str) -> str:
    """Create a grounded RAG prompt for document-based answers."""
    return f"""
You are an AI Training Assistant for new employees.

The employee question was routed to: {route_label}.

Instructions:
- Answer only using the provided context when context is available.
- If the answer is not present in the context, say: "This information is not available in the current knowledge base."
- Do not hallucinate or invent company-specific policies.
- Keep the answer clear, professional, and useful for a new employee.
- Mention that the answer is based on company documents when you use the context.

Employee question:
{query}

Retrieved company context:
{context}

Final answer:
"""


def build_direct_prompt(query: str) -> str:
    """Create a direct answer prompt for non-document questions."""
    return f"""
You are an AI Training Assistant for new employees.

Instructions:
- Answer general learning questions briefly and clearly.
- Avoid company-specific claims unless they are supported by company documents.
- If the question asks for company-specific policy or HR details without a source, advise the employee to contact HR/admin or consult official documents.
- Use a professional and friendly tone.

Employee question:
{query}

Final answer:
"""

