"""Shared fixtures for outbox tests."""

from __future__ import annotations

from agentm_channels.wire import KIND_OUTBOUND, WIRE_VERSION, Envelope


def make_envelope(env_id: str, ts: float = 1000.0) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id=env_id,
        kind=KIND_OUTBOUND,
        ts=ts,
        body={"text": f"hello {env_id}"},
    )
