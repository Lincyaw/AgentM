"""Repeated nack then dead_letter — row leaves outbox, lands in dead_letter."""

from __future__ import annotations

from pathlib import Path

from agentm_channels.outbox import SqliteOutbox

from .conftest import make_envelope


def test_dead_letter_after_repeated_nack(tmp_path: Path) -> None:
    store = SqliteOutbox(str(tmp_path / "ob.sqlite"))
    try:
        store.enqueue("peer-A", make_envelope("m1"))

        # Three failed attempts.
        last_id: int | None = None
        for attempt in range(1, 4):
            leased = store.lease("peer-A", batch_max=10, now=1000.0 * attempt)
            assert len(leased) == 1
            assert leased[0].attempts == attempt
            last_id = leased[0].id
            store.nack([leased[0].id], next_retry_at=1000.0 * attempt)

        assert last_id is not None
        # Caller decides "enough" and dead-letters.
        store.dead_letter(last_id, reason="max_attempts")

        assert store.pending_count("peer-A") == 0
        assert store.dead_letter_count("peer-A") == 1

        # Subsequent lease yields nothing.
        assert store.lease("peer-A", batch_max=10, now=9000.0) == []
    finally:
        store.close()


