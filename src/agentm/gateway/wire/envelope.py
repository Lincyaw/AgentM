"""Wire envelope v2.

JSON-safe dataclass implementing
``.claude/designs/single-process-gateway.md`` §2.2. The v2 envelope is
intentionally minimal: routing primitives from v1 (``to`` /
``correlation_id`` / ``hops`` / ``root_session_key`` / ``session_id`` /
``peer_kind``) are all gone because there is no cross-process routing —
only ``session_key`` (chat-client computed, opaque to the gateway) and
``scenario`` (set on the first inbound for an unseen chat) survive.

Pure module: no I/O, no logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import InvalidEnvelope
from .kinds import VALID_KINDS

WIRE_VERSION: int = 2


@dataclass(frozen=True, slots=True)
class Envelope:
    """A single wire message (v2).

    Required on every envelope: ``v``, ``id``, ``kind``, ``ts``.
    Conditional: ``session_key`` (on ``inbound`` / ``outbound``),
    ``scenario`` (on the first ``inbound`` for an unseen chat).
    """

    v: int
    id: str
    kind: str
    ts: float
    body: dict[str, Any] = field(default_factory=dict)
    session_key: str | None = None
    scenario: str | None = None

    def __post_init__(self) -> None:
        if self.v != WIRE_VERSION:
            raise InvalidEnvelope(
                f"unsupported wire version {self.v!r}; expected {WIRE_VERSION}"
            )
        if not isinstance(self.id, str) or not self.id:
            raise InvalidEnvelope("envelope id must be a non-empty string")
        if self.kind not in VALID_KINDS:
            raise InvalidEnvelope(f"unknown envelope kind {self.kind!r}")
        if not isinstance(self.body, dict):
            raise InvalidEnvelope("envelope body must be a dict")

    def to_dict(self) -> dict[str, Any]:
        """Render to a JSON-safe dict. None-valued conditionals omitted."""
        out: dict[str, Any] = {
            "v": self.v,
            "id": self.id,
            "kind": self.kind,
            "ts": self.ts,
            "body": self.body,
        }
        if self.session_key is not None:
            out["session_key"] = self.session_key
        if self.scenario is not None:
            out["scenario"] = self.scenario
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Envelope:
        """Reconstruct from a JSON-decoded dict.

        Raises :class:`InvalidEnvelope` if required fields are missing or
        any field has the wrong type. A v1 envelope (``v=1``) hits the
        version check in ``__post_init__`` and is rejected — the wire
        version bump is a hard break (§2).
        """
        if not isinstance(data, dict):
            raise InvalidEnvelope("envelope wire payload must be a JSON object")
        for required in ("v", "id", "kind", "ts", "body"):
            if required not in data:
                raise InvalidEnvelope(f"envelope missing required field {required!r}")
        try:
            return cls(
                v=int(data["v"]),
                id=str(data["id"]),
                kind=str(data["kind"]),
                ts=float(data["ts"]),
                body=data["body"],
                session_key=_opt_str(data.get("session_key"), "session_key"),
                scenario=_opt_str(data.get("scenario"), "scenario"),
            )
        except (TypeError, ValueError) as exc:
            raise InvalidEnvelope(f"envelope field has wrong type: {exc}") from exc


def _opt_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidEnvelope(f"envelope field {name!r} must be a string or null")
    return value


__all__ = ["Envelope", "WIRE_VERSION"]
