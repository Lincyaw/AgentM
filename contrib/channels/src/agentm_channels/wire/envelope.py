"""Wire envelope.

JSON-safe dataclass implementing designs/client-server-architecture.md
§4.2. All optional fields from §10 decision #8 land Day 1 even where
unused — mid-cluster wire bumps are expensive.

Pure module: no I/O, no logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import InvalidEnvelope
from .kinds import VALID_KINDS

WIRE_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class Envelope:
    """A single wire message.

    Required: ``v``, ``id``, ``kind``, ``ts``, ``body``.
    Optional (defaulted): ``to``, ``correlation_id``, ``hops``,
    ``root_session_key``, ``peer_kind``.
    """

    v: int
    id: str
    kind: str
    ts: float
    body: dict[str, Any] = field(default_factory=dict)
    to: str | None = None
    correlation_id: str | None = None
    hops: int = 0
    root_session_key: str | None = None
    peer_kind: str | None = None

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
        if not isinstance(self.hops, int) or self.hops < 0:
            raise InvalidEnvelope("envelope hops must be a non-negative int")

    def to_dict(self) -> dict[str, Any]:
        """Render to a JSON-safe dict. None-valued optionals omitted."""
        out: dict[str, Any] = {
            "v": self.v,
            "id": self.id,
            "kind": self.kind,
            "ts": self.ts,
            "body": self.body,
            "hops": self.hops,
        }
        if self.to is not None:
            out["to"] = self.to
        if self.correlation_id is not None:
            out["correlation_id"] = self.correlation_id
        if self.root_session_key is not None:
            out["root_session_key"] = self.root_session_key
        if self.peer_kind is not None:
            out["peer_kind"] = self.peer_kind
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Envelope:
        """Reconstruct from a JSON-decoded dict.

        Raises :class:`InvalidEnvelope` if required fields are missing
        or any field has the wrong type.
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
                to=_opt_str(data.get("to"), "to"),
                correlation_id=_opt_str(data.get("correlation_id"), "correlation_id"),
                hops=int(data.get("hops", 0)),
                root_session_key=_opt_str(
                    data.get("root_session_key"), "root_session_key"
                ),
                peer_kind=_opt_str(data.get("peer_kind"), "peer_kind"),
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
