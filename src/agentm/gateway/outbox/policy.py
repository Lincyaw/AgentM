"""Retry policy as free functions.

Kept separate from the outbox storage (mechanism) so callers — the
server's delivery worker — pick the delay and pass it to
:meth:`OutboxStore.nack`. Policy belongs to the caller; the store
just persists what it's told.
"""

from __future__ import annotations

import random


def exponential_backoff(
    attempts: int,
    base: float = 1.0,
    cap: float = 60.0,
    jitter_ratio: float = 0.1,
) -> float:
    """Exponential backoff with proportional jitter.

    ``min(cap, base * 2 ** (attempts - 1)) * (1 +/- jitter_ratio)``.
    ``attempts`` is 1 after the first failure. ``jitter_ratio`` is the
    fraction of the base delay added or subtracted uniformly.
    """
    if attempts < 1:
        attempts = 1
    raw = base * (2 ** (attempts - 1))
    delay = min(cap, raw)
    if jitter_ratio:
        delay *= 1.0 + random.uniform(-jitter_ratio, jitter_ratio)
    return max(0.0, delay)


__all__ = ["exponential_backoff"]
