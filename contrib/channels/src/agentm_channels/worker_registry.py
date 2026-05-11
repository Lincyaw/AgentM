"""Gateway-side registry for ``agent_worker`` wire peers.

Phase 5a routing surface — see
``.claude/designs/client-server-architecture.md`` for the topology and
the brief in ``.claude/plans/2026-05-11-phase5-agent-worker.md`` for
the policy this module implements (or absence thereof).

Policy (minimum-viable):

* A worker advertises ``capabilities.scenarios: [<name>, ...]`` at
  hello. A worker matches an inbound message if its advertised list
  contains the gateway's currently-configured ``scenario`` string.
  Equality only — no glob, no version, no LB weight.
* A ``session_key`` is bound to the first matching worker on first
  inbound and stays sticky for the rest of that worker's lifetime.
  When the worker disconnects the binding is released; the next
  inbound for that session_key picks again from the live pool.

Anything fancier (multi-worker LB, presence-tracked load, scenario
glob match, sticky persistence across worker restarts) belongs to a
later phase — keep this surface small per
``feedback_simple_and_pluggable`` (memory).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("agentm_channels.worker_registry")


@dataclass
class WorkerInfo:
    """One connected ``agent_worker`` peer's capabilities."""

    peer_id: str
    scenarios: frozenset[str]
    cwd: str | None = None
    raw_capabilities: dict[str, Any] = field(default_factory=dict)


class WorkerRegistry:
    """In-memory registry of connected workers + sticky session bindings.

    Single asyncio loop → no locking. The gateway owns the only
    instance; ``WireBridge`` reaches into it on hello / disconnect /
    inbound.
    """

    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        self._sticky: dict[str, str] = {}

    # -- worker lifecycle --------------------------------------------

    def register(self, peer_id: str, capabilities: dict[str, Any]) -> None:
        raw = capabilities or {}
        scenarios_raw = raw.get("scenarios") or []
        if isinstance(scenarios_raw, list):
            scenarios = frozenset(str(s) for s in scenarios_raw)
        else:
            scenarios = frozenset()
        cwd_raw = raw.get("cwd")
        cwd = str(cwd_raw) if isinstance(cwd_raw, str) and cwd_raw else None
        self._workers[peer_id] = WorkerInfo(
            peer_id=peer_id,
            scenarios=scenarios,
            cwd=cwd,
            raw_capabilities=dict(raw),
        )
        log.info(
            "worker registered peer_id=%s scenarios=%s cwd=%s",
            peer_id,
            sorted(scenarios),
            cwd,
        )

    def unregister(self, peer_id: str) -> set[str]:
        """Drop the worker. Return the set of ``session_key`` that were
        bound to it — the caller is responsible for notifying chat
        clients that their in-flight turns are lost."""
        self._workers.pop(peer_id, None)
        lost = {
            key for key, owner in self._sticky.items() if owner == peer_id
        }
        for key in lost:
            self._sticky.pop(key, None)
        if lost:
            log.info(
                "worker unregistered peer_id=%s lost_sessions=%s",
                peer_id,
                sorted(lost),
            )
        return lost

    # -- routing -----------------------------------------------------

    def find_worker(self, scenario: str, session_key: str) -> str | None:
        """Look up the worker to receive an inbound for ``session_key``.

        Returns ``None`` when no live worker advertises ``scenario``.
        Side effect: on a cold session_key the first match is *bound*
        sticky so subsequent calls within the same turn-stream stay on
        the same worker.
        """
        sticky_owner = self._sticky.get(session_key)
        if sticky_owner is not None and sticky_owner in self._workers:
            return sticky_owner
        # Stale sticky entry (worker disconnected without unregister
        # being called yet) → drop and re-pick.
        if sticky_owner is not None and sticky_owner not in self._workers:
            self._sticky.pop(session_key, None)
        # First-match-wins. ``dict`` iteration is insertion-ordered,
        # so the earliest-connected matching worker is chosen.
        for peer_id, info in self._workers.items():
            if scenario in info.scenarios:
                self._sticky[session_key] = peer_id
                return peer_id
        return None

    def release_session(self, session_key: str) -> None:
        self._sticky.pop(session_key, None)

    # -- introspection (tests, logging) ------------------------------

    def workers(self) -> list[WorkerInfo]:
        return list(self._workers.values())

    def sticky_owner(self, session_key: str) -> str | None:
        return self._sticky.get(session_key)

    def has(self, peer_id: str) -> bool:
        return peer_id in self._workers


__all__ = ["WorkerInfo", "WorkerRegistry"]
