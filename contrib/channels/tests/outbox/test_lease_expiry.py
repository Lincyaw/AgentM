"""Lease without ack → after TTL, the same rows lease again."""

from __future__ import annotations

from pathlib import Path

from agentm_channels.outbox import LEASE_TTL_SECONDS, SqliteOutbox

from .conftest import make_envelope


def test_expired_lease_is_reissued_with_incremented_attempts(tmp_path: Path) -> None:
    store = SqliteOutbox(str(tmp_path / "ob.sqlite"))
    try:
        for i in range(3):
            store.enqueue("peer-A", make_envelope(f"m{i}"))

        first = store.lease("peer-A", batch_max=10, now=1000.0)
        assert [r.attempts for r in first] == [1, 1, 1]

        # Within the lease window, lease returns nothing.
        within = store.lease("peer-A", batch_max=10, now=1000.0 + 1.0)
        assert within == []

        # After TTL elapses, the same rows are leased again.
        after = store.lease(
            "peer-A", batch_max=10, now=1000.0 + LEASE_TTL_SECONDS + 1.0
        )
        assert [r.envelope.id for r in after] == [r.envelope.id for r in first]
        assert [r.attempts for r in after] == [2, 2, 2]
    finally:
        store.close()
