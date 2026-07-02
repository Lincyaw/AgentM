"""Pure scoring functions for TELBench evaluation.

No LLM calls, no pipeline knowledge. Operates on sets of span indices
and produces precision / recall / F1 / FEA (First Error Accuracy).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpanScores:
    """Per-instance span-level scores."""

    precision: float
    recall: float
    f1: float
    first_error_accurate: bool


@dataclass(frozen=True, slots=True)
class AggregateScores:
    """Macro-averaged scores across instances."""

    macro_precision: float
    macro_recall: float
    macro_f1: float
    first_error_accuracy: float
    n_instances: int


def score_instance(
    predicted_error_indices: set[int],
    gold_error_indices: set[int],
    n_spans: int,
) -> SpanScores:
    """Score one instance. Span-level P/R/F1 + FEA.

    Precision = |predicted ^ gold| / |predicted|  (0 if no predictions).
    Recall    = |predicted ^ gold| / |gold|        (0 if no gold).
    F1        = harmonic mean of P and R           (0 if either is 0).
    FEA       = min(predicted) == min(gold) if both non-empty, else False.
    """
    del n_spans  # available for future use (e.g. span-weighted metrics)
    tp = len(predicted_error_indices & gold_error_indices)

    precision = tp / len(predicted_error_indices) if predicted_error_indices else 0.0
    recall = tp / len(gold_error_indices) if gold_error_indices else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    if predicted_error_indices and gold_error_indices:
        first_error_accurate = min(predicted_error_indices) == min(gold_error_indices)
    else:
        first_error_accurate = False

    return SpanScores(
        precision=precision,
        recall=recall,
        f1=f1,
        first_error_accurate=first_error_accurate,
    )


def aggregate_scores(instance_scores: list[SpanScores]) -> AggregateScores:
    """Macro-average across instances."""
    n = len(instance_scores)
    if n == 0:
        return AggregateScores(
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            first_error_accuracy=0.0,
            n_instances=0,
        )
    return AggregateScores(
        macro_precision=sum(s.precision for s in instance_scores) / n,
        macro_recall=sum(s.recall for s in instance_scores) / n,
        macro_f1=sum(s.f1 for s in instance_scores) / n,
        first_error_accuracy=sum(1 for s in instance_scores if s.first_error_accurate) / n,
        n_instances=n,
    )


__all__ = [
    "AggregateScores",
    "SpanScores",
    "aggregate_scores",
    "score_instance",
]
