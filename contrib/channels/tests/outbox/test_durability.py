"""Enqueue, close, reopen, lease — all messages survive."""

from __future__ import annotations

from pathlib import Path

from agentm_channels.outbox import SqliteOutbox

from .conftest import make_envelope


def test_messages_survive_close_and_reopen(tmp_path: Path) -> None:
    db = str(tmp_path / "ob.sqlite")
    store = SqliteOutbox(db)
    try:
        for i in range(8):
            store.enqueue("peer-A", make_envelope(f"m{i}"))
    finally:
        store.close()

    reopened = SqliteOutbox(db)
    try:
        assert reopened.pending_count("peer-A") == 8
        leased = reopened.lease("peer-A", batch_max=100, now=2000.0)
        assert [r.envelope.id for r in leased] == [f"m{i}" for i in range(8)]
    finally:
        reopened.close()


def test_acks_persist_across_restart(tmp_path: Path) -> None:
    db = str(tmp_path / "ob.sqlite")
    store = SqliteOutbox(db)
    try:
        for i in range(5):
            store.enqueue("peer-A", make_envelope(f"m{i}"))
        leased = store.lease("peer-A", batch_max=3, now=2000.0)
        store.ack([r.id for r in leased])
    finally:
        store.close()

    reopened = SqliteOutbox(db)
    try:
        assert reopened.pending_count("peer-A") == 2
        leased = reopened.lease("peer-A", batch_max=10, now=3000.0)
        assert [r.envelope.id for r in leased] == ["m3", "m4"]
    finally:
        reopened.close()
