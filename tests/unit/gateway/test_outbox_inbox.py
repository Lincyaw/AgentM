"""Fail-stop: outbox at-least-once + dedup, inbox ack-on-process (§2.6).

The durable outbox/inbox are the delivery-reliability boundary. If a
crash mid-delivery drops a message, or a duplicate inbound is processed
twice, the gateway's at-least-once / at-most-once guarantees are void.
"""

from __future__ import annotations

from pathlib import Path

from agentm.gateway.outbox import SqliteInbox, SqliteOutbox
from agentm.gateway.wire import WIRE_VERSION, Envelope


def _env(env_id: str) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id=env_id,
        kind="outbound",
        ts=1.0,
        session_key="terminal:t1",
        body={"channel": "terminal", "chat_id": "t1", "content": "x"},
    )


def test_enqueue_is_idempotent_on_envelope_id(tmp_path: Path) -> None:
    ob = SqliteOutbox(str(tmp_path / "ob.sqlite"))
    try:
        ob.enqueue("peer1", _env("e1"))
        ob.enqueue("peer1", _env("e1"))  # duplicate — no-op
        assert ob.pending_count("peer1") == 1
    finally:
        ob.close()


def test_lease_then_crash_redelivers_after_lease_ttl(tmp_path: Path) -> None:
    # A short lease so an "expired" lease (crash mid-delivery) is leasable
    # again without sleeping real time — we drive ``now`` forward.
    ob = SqliteOutbox(str(tmp_path / "ob.sqlite"), lease_ttl=10.0)
    try:
        ob.enqueue("peer1", _env("e1"))
        leased = ob.lease("peer1", 10, now=100.0)
        assert [r.envelope.id for r in leased] == ["e1"]
        # Crash before ack: the row is still there.
        assert ob.pending_count("peer1") == 1
        # Within the lease window it is NOT re-leasable...
        assert ob.lease("peer1", 10, now=105.0) == []
        # ...but after the lease expires it comes back.
        again = ob.lease("peer1", 10, now=200.0)
        assert [r.envelope.id for r in again] == ["e1"]
    finally:
        ob.close()


def test_ack_removes_the_row(tmp_path: Path) -> None:
    ob = SqliteOutbox(str(tmp_path / "ob.sqlite"))
    try:
        ob.enqueue("peer1", _env("e1"))
        leased = ob.lease("peer1", 10, now=1.0)
        ob.ack([r.id for r in leased])
        assert ob.pending_count("peer1") == 0
    finally:
        ob.close()


def test_dead_letter_after_exhausted_attempts(tmp_path: Path) -> None:
    ob = SqliteOutbox(str(tmp_path / "ob.sqlite"))
    try:
        ob.enqueue("peer1", _env("e1"))
        leased = ob.lease("peer1", 10, now=1.0)
        ob.dead_letter(leased[0].id, "peer unreachable")
        assert ob.pending_count("peer1") == 0
        assert ob.dead_letter_count("peer1") == 1
    finally:
        ob.close()


def test_inbox_records_once_dedups_thereafter(tmp_path: Path) -> None:
    ib = SqliteInbox(str(tmp_path / "ib.sqlite"))
    try:
        assert ib.record_seen("peer1", "i1", 1.0) is True  # newly seen -> process
        assert ib.record_seen("peer1", "i1", 1.0) is False  # dup -> skip
        # A different envelope id IS processed.
        assert ib.record_seen("peer1", "i2", 1.0) is True
    finally:
        ib.close()


def test_inbox_dedup_survives_restart(tmp_path: Path) -> None:
    path = str(tmp_path / "ib.sqlite")
    ib = SqliteInbox(path)
    ib.record_seen("peer1", "i1", 1.0)
    ib.close()
    # Restart: the ledger persists, so the same inbound is still a dup.
    ib2 = SqliteInbox(path)
    try:
        assert ib2.record_seen("peer1", "i1", 1.0) is False
    finally:
        ib2.close()
