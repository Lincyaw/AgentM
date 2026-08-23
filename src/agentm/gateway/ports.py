# code-health: ignore-file[AM025] -- gateway wire DTOs validate external transport payloads
"""Pluggable persistence and authentication ports for gateway hosts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, Protocol, runtime_checkable

from agentm.core.abi.messages import JsonValue, freeze_json
from agentm.gateway.wire import GatewayEnvelope

TransportKind = Literal["unix", "ws", "wss", "custom"]
_TRANSPORT_KINDS: Final = frozenset({"unix", "ws", "wss", "custom"})


def _empty_json_object() -> Mapping[str, JsonValue]:
    return MappingProxyType({})


def _require_nonempty(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def _require_finite(value: float, label: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be finite")


@dataclass(frozen=True, slots=True)
class OutboxRecord:
    """One leased durable outbound record."""

    record_id: str
    peer_id: str
    envelope: GatewayEnvelope
    attempts: int
    enqueued_at: float
    lease_expires_at: float

    def __post_init__(self) -> None:
        _require_nonempty(self.record_id, "outbox record_id")
        _require_nonempty(self.peer_id, "outbox peer_id")
        if not isinstance(self.envelope, GatewayEnvelope):
            raise TypeError("outbox envelope must be a GatewayEnvelope")
        if self.envelope.kind != "outbound":
            raise ValueError("outbox records require an outbound envelope")
        if (
            not isinstance(self.attempts, int)
            or isinstance(self.attempts, bool)
            or self.attempts < 1
        ):
            raise ValueError("outbox attempts must be a positive integer")
        _require_finite(self.enqueued_at, "outbox enqueued_at")
        _require_finite(self.lease_expires_at, "outbox lease_expires_at")
        if self.lease_expires_at <= self.enqueued_at:
            raise ValueError("outbox lease_expires_at must be after enqueued_at")


@runtime_checkable
class OutboxStore(Protocol):
    """Durable per-peer queue providing at-least-once outbound replay."""

    async def enqueue(
        self,
        peer_id: str,
        envelope: GatewayEnvelope,
        *,
        enqueued_at: float,
    ) -> str:
        """Idempotently enqueue by ``(peer_id, envelope.envelope_id)``."""
        ...

    async def lease(
        self,
        peer_id: str,
        *,
        limit: int,
        now: float,
        lease_expires_at: float,
    ) -> Sequence[OutboxRecord]:
        """Lease ready rows in enqueue order and increment attempts."""
        ...

    async def acknowledge(self, record_ids: Sequence[str]) -> None:
        """Permanently remove delivered rows."""
        ...

    async def release(
        self,
        record_ids: Sequence[str],
        *,
        retry_at: float,
    ) -> None:
        """Release leases and make rows available at ``retry_at``."""
        ...

    async def dead_letter(self, record_id: str, *, reason: str) -> None:
        """Move an exhausted row out of the live queue."""
        ...

    async def pending_count(self, peer_id: str) -> int:
        """Return live durable rows for one peer."""
        ...

    async def close(self) -> None:
        """Release backend resources."""
        ...


@runtime_checkable
class InboxLog(Protocol):
    """Durable idempotency ledger for peer-to-gateway envelopes."""

    async def record_seen(
        self,
        peer_id: str,
        envelope_id: str,
        *,
        received_at: float,
    ) -> bool:
        """Return true only for the first durable observation."""
        ...

    async def prune(self, *, older_than: float) -> int:
        """Remove expired deduplication rows."""
        ...

    async def close(self) -> None:
        """Release backend resources."""
        ...


@dataclass(frozen=True, slots=True)
class PeerCredentials:
    """Transport-neutral authentication evidence from a hello request."""

    peer_id: str
    transport: TransportKind
    token: str | None = None
    process_id: int | None = None
    user_id: int | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        _require_nonempty(self.peer_id, "peer credentials peer_id")
        if self.transport not in _TRANSPORT_KINDS:
            raise ValueError(f"unsupported peer transport: {self.transport!r}")
        if self.token is not None:
            _require_nonempty(self.token, "peer credentials token")
        for label, value in (
            ("process_id", self.process_id),
            ("user_id", self.user_id),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(
                    f"peer credentials {label} must be a non-negative integer"
                )
        frozen = freeze_json(self.metadata)
        if not isinstance(frozen, Mapping):
            raise TypeError("peer credential metadata must be a JSON object")
        object.__setattr__(self, "metadata", frozen)


@dataclass(frozen=True, slots=True)
class AuthenticationResult:
    """Final identity decision returned by an Authenticator."""

    accepted: bool
    principal_id: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("authentication accepted must be a boolean")
        if self.accepted:
            if self.principal_id is None:
                raise ValueError("accepted authentication requires principal_id")
            _require_nonempty(
                self.principal_id,
                "authentication principal_id",
            )
            if self.reason is not None:
                raise ValueError(
                    "accepted authentication cannot include a rejection reason"
                )
            return
        if self.principal_id is not None:
            raise ValueError("rejected authentication cannot include principal_id")
        if self.reason is None:
            raise ValueError("rejected authentication requires a reason")
        _require_nonempty(self.reason, "authentication rejection reason")


@runtime_checkable
class Authenticator(Protocol):
    """Authenticate a peer without coupling policy to a transport object."""

    async def authenticate(
        self,
        credentials: PeerCredentials,
    ) -> AuthenticationResult: ...


@dataclass(frozen=True, slots=True)
class SessionBinding:
    """Durable mapping used to resume one opaque chat conversation."""

    session_key: str
    session_id: str
    scenario: str | None = None
    model: str | None = None
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        _require_nonempty(self.session_key, "session binding session_key")
        _require_nonempty(self.session_id, "session binding session_id")
        if self.scenario is not None:
            _require_nonempty(self.scenario, "session binding scenario")
        if self.model is not None:
            _require_nonempty(self.model, "session binding model")
        _require_finite(self.updated_at, "session binding updated_at")


@runtime_checkable
class SessionBindingStore(Protocol):
    """Persistence port for ``session_key -> session_id`` recovery."""

    async def get(self, session_key: str) -> SessionBinding | None: ...

    async def put(self, binding: SessionBinding) -> None: ...

    async def delete(self, session_key: str) -> None: ...

    async def list(self) -> Sequence[SessionBinding]: ...

    async def close(self) -> None: ...


__all__ = [
    "AuthenticationResult",
    "Authenticator",
    "InboxLog",
    "OutboxRecord",
    "OutboxStore",
    "PeerCredentials",
    "SessionBinding",
    "SessionBindingStore",
    "TransportKind",
]
