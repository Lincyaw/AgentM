"""Happy-path enqueue/lease/ack + idempotent enqueue."""

from __future__ import annotations

from pathlib import Path

from agentm_channels.outbox import SqliteOutbox

from .conftest import make_envelope


def test_enqueue_lease_ack(tmp_path: Path) -> None:
    store = SqliteOutbox(str(tmp_path / "ob.sqlite"))
    try:
        store.enqueue("peer-A", make_envelope("m1"))
        store.enqueue("peer-A", make_envelope("m2"))

        assert store.pending_count("peer-A") == 2

        leased = store.lease("peer-A", batch_max=10, now=2000.0)
        assert [r.envelope.id for r in leased] == ["m1", "m2"]
        assert all(r.attempts == 1 for r in leased)

        store.ack([r.id for r in leased])
        assert store.pending_count("peer-A") == 0
    finally:
        store.close()


def test_enqueue_idempotent(tmp_path: Path) -> None:
    store = SqliteOutbox(str(tmp_path / "ob.sqlite"))
    try:
        env = make_envelope("dup")
        store.enqueue("peer-A", env)
        store.enqueue("peer-A", env)
        store.enqueue("peer-A", env)
        assert store.pending_count("peer-A") == 1
    finally:
        store.close()


def test_pending_count_isolated_per_peer(tmp_path: Path) -> None:
    store = SqliteOutbox(str(tmp_path / "ob.sqlite"))
    try:
        store.enqueue("peer-A", make_envelope("a1"))
        store.enqueue("peer-B", make_envelope("b1"))
        store.enqueue("peer-B", make_envelope("b2"))
        assert store.pending_count("peer-A") == 1
        assert store.pending_count("peer-B") == 2

        leased = store.lease("peer-A", batch_max=10, now=2000.0)
        assert [r.peer_id for r in leased] == ["peer-A"]
    finally:
        store.close()
