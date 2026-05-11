"""Catch-up batch delivery: lease in chunks, in FIFO order."""

from __future__ import annotations

from pathlib import Path

from agentm_channels.outbox import SqliteOutbox

from .conftest import make_envelope


def test_batched_lease_returns_fifo_chunks(tmp_path: Path) -> None:
    store = SqliteOutbox(str(tmp_path / "ob.sqlite"))
    try:
        for i in range(50):
            store.enqueue("peer-A", make_envelope(f"m{i:02d}"))

        first = store.lease("peer-A", batch_max=10, now=2000.0)
        assert [r.envelope.id for r in first] == [f"m{i:02d}" for i in range(10)]
        store.ack([r.id for r in first])

        second = store.lease("peer-A", batch_max=10, now=2000.0)
        assert [r.envelope.id for r in second] == [
            f"m{i:02d}" for i in range(10, 20)
        ]
        assert store.pending_count("peer-A") == 40
    finally:
        store.close()


def test_batch_max_zero_yields_empty(tmp_path: Path) -> None:
    store = SqliteOutbox(str(tmp_path / "ob.sqlite"))
    try:
        store.enqueue("peer-A", make_envelope("m1"))
        assert store.lease("peer-A", batch_max=0, now=2000.0) == []
    finally:
        store.close()
