"""Outbox / inbox extension protocols.

Two ``typing.Protocol`` definitions form the only extension surface
for durable delivery in the gateway. The default SQLite implementation
lives in :mod:`agentm_channels.outbox.sqlite`. A contrib backend
(Redis Streams, NATS JetStream, ...) only needs to satisfy these
Protocols.

See ``.claude/designs/client-server-architecture.md`` §4.5 for the
delivery semantics this surface supports (at-least-once outbound,
at-most-once-with-ack inbound, dead-letter on retry exhaustion).

Mechanism only — retry/backoff policy lives in
:mod:`agentm_channels.outbox.policy` so callers (the server) decide
when to nack vs dead_letter and what delay to set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentm_channels.wire import Envelope


@dataclass(frozen=True, slots=True)
class OutboxRecord:
    """A leased outbox row handed to a delivery worker.

    ``id`` is the storage-assigned row id used by ack/nack/dead_letter.
    ``attempts`` is the attempt count *including* the lease that
    produced this record (i.e. starts at 1 on first lease).
    """

    id: int
    peer_id: str
    envelope: Envelope
    attempts: int
    enqueued_at: float
    next_retry_at: float


@runtime_checkable
class OutboxStore(Protocol):
    """Durable per-peer queue. At-least-once delivery."""

    def enqueue(self, peer_id: str, env: Envelope) -> None:
        """Append ``env`` to ``peer_id``'s queue.

        Idempotent on ``(peer_id, env.id)``: a duplicate enqueue is a
        silent no-op so producer retry is safe.
        """
        ...

    def lease(
        self, peer_id: str, batch_max: int, now: float
    ) -> list[OutboxRecord]:
        """Lease up to ``batch_max`` ready records for ``peer_id``.

        A record is *ready* when ``next_retry_at <= now`` and no
        unexpired lease holds it. Leasing increments ``attempts`` and
        marks the record leased until ``now + LEASE_TTL``. Returns
        records ordered by enqueue order (FIFO).
        """
        ...

    def ack(self, record_ids: list[int]) -> None:
        """Permanently remove successfully-delivered records."""
        ...

    def nack(self, record_ids: list[int], next_retry_at: float) -> None:
        """Release the lease and reschedule for later retry.

        ``next_retry_at`` is computed by the caller (see
        :mod:`agentm_channels.outbox.policy`).
        """
        ...

    def dead_letter(self, record_id: int, reason: str) -> None:
        """Move a record to the dead-letter table.

        Caller invokes this after the retry policy says "give up".
        """
        ...

    def pending_count(self, peer_id: str) -> int:
        """Total rows queued for ``peer_id`` regardless of lease state."""
        ...

    def close(self) -> None:
        """Release storage resources."""
        ...


@runtime_checkable
class InboxLog(Protocol):
    """Idempotent receive ledger for ``peer -> server`` inbound.

    Mirrors the design's at-most-once-with-ack-on-process model
    (§4.5.1): the server records the envelope before processing and
    treats duplicates as no-op acks.
    """

    def record_seen(self, peer_id: str, envelope_id: str, ts: float) -> bool:
        """Record an inbound envelope id.

        Returns ``True`` if newly inserted, ``False`` if it was already
        present (the caller should re-ack without reprocessing).
        """
        ...

    def prune(self, older_than: float) -> int:
        """Drop entries older than ``older_than``. Returns rows removed."""
        ...

    def close(self) -> None:
        """Release storage resources."""
        ...


__all__ = ["InboxLog", "OutboxRecord", "OutboxStore"]
