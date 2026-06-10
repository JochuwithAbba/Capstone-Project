"""Answer evaluation helpers for BLEU and ROUGE scoring."""

from __future__ import annotations

from src.utils import format_context_chunks, is_fallback_context, normalize_text


def calculate_answer_scores(answer: str, retrieved_context: list[str]) -> dict:
    """
    Calculate BLEU and ROUGE scores against retrieved context.

    In this prototype there are no human-written reference answers. For RAG
    questions, the retrieved document context is used as the reference text.
    These scores should therefore be explained as document-overlap metrics, not
    as absolute answer-quality scores.
    """
    if not answer.strip() or is_fallback_context(retrieved_context):
        return {
            "available": False,
            "reason": "Scores require a final answer and retrieved document context.",
        }

    reference_text = format_context_chunks(retrieved_context)
    if not reference_text.strip():
        return {
            "available": False,
            "reason": "No retrieved document context was available for scoring.",
        }

    try:
        bleu_score = _calculate_bleu(reference_text, answer)
        rouge_scores = _calculate_rouge(reference_text, answer)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"Could not calculate evaluation scores. Reason: {exc}",
        }

    return {
        "available": True,
        "reference": "retrieved_document_context",
        "bleu": bleu_score,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "note": "BLEU and ROUGE are calculated against retrieved context, not a human reference answer.",
    }


def _calculate_bleu(reference_text: str, answer: str) -> float:
    """Calculate sentence-level BLEU score."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    reference_tokens = _tokenize(reference_text)
    answer_tokens = _tokenize(answer)
    if not reference_tokens or not answer_tokens:
        return 0.0

    score = sentence_bleu(
        [reference_tokens],
        answer_tokens,
        smoothing_function=SmoothingFunction().method1,
    )
    return round(float(score), 4)


def _calculate_rouge(reference_text: str, answer: str) -> dict[str, float]:
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_text, answer)
    return {
        metric: round(float(value.fmeasure), 4)
        for metric, value in scores.items()
    }


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for scoring."""
    return normalize_text(text).split()
