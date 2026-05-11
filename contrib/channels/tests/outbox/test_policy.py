"""Exponential backoff math + jitter bounds."""

from __future__ import annotations

from agentm_channels.outbox import exponential_backoff


def test_no_jitter_doubles_each_attempt() -> None:
    delays = [
        exponential_backoff(a, base=1.0, cap=60.0, jitter_ratio=0.0)
        for a in range(1, 6)
    ]
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_cap_applies() -> None:
    delay = exponential_backoff(10, base=1.0, cap=60.0, jitter_ratio=0.0)
    assert delay == 60.0


def test_jitter_stays_within_ratio() -> None:
    for _ in range(50):
        d = exponential_backoff(3, base=1.0, cap=60.0, jitter_ratio=0.2)
        # raw=4.0, +/-20% → [3.2, 4.8]
        assert 3.2 <= d <= 4.8


def test_attempts_below_one_treated_as_one() -> None:
    assert exponential_backoff(0, base=1.0, cap=60.0, jitter_ratio=0.0) == 1.0
    assert exponential_backoff(-3, base=1.0, cap=60.0, jitter_ratio=0.0) == 1.0
