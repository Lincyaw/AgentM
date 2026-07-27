"""Stage 1: ask a benchmark which of its attempts failed.

Thin on purpose. Discovery is entirely benchmark knowledge -- what a verifier
writes, which metrics exist, where a reference solution lives -- and none of it
belongs in the loop. What stays here is the one thing that is not benchmark
knowledge: a case must be self-describing before it is written out, since every
later stage may be handed the artifact and nothing else.
"""

from __future__ import annotations

from loguru import logger

from .contracts import FailureCase
from .protocols import CaseSource


def collect(source: CaseSource, batch: str, *, only: str = "") -> list[FailureCase]:
    cases = source.discover(batch, only=only)
    for case in cases:
        _warn_if_thin(case)
    logger.info("collect: {} case(s) from {}", len(cases), source.name)
    return cases


def _warn_if_thin(case: FailureCase) -> None:
    """An adapter that omits these produces cases that look fine and diagnose
    badly, so it is said at discovery rather than left to be inferred from a
    weak diagnosis."""
    if not case.evidence:
        logger.warning("collect: {} lists no evidence to read", case.case_id)
    if not case.failing_assertions:
        logger.warning("collect: {} records no failing assertion", case.case_id)
    if not case.replayable:
        logger.info(
            "collect: {} carries no backend reference; diagnosable, not replayable",
            case.case_id,
        )


__all__ = ["collect"]
