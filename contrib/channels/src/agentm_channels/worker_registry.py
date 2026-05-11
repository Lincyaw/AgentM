"""Gateway-side host pool + persistent session→host binding.

This used to be a "worker registry with an in-memory sticky map".
The refactor flips the abstraction order:

* **Session is the primary routable thing.** The persistent binding
  (``SessionBindingStore``) is what answers "where does this
  conversation live?".
* **Hosts are just session storage.** A worker, an inproc gateway
  loop, or any future ``agent_worker`` peer is interchangeable as
  long as it can host an ``AgentSession``.

Routing logic on inbound:

1. Look up the binding for the inbound's ``session_key``.
2. If bound and host is still online → forward there.
3. If bound but host is offline → pick any online host advertising
   the requested scenario, rebind, and forward; the new host receives
   the binding's ``resume_id`` (if any) and resumes the session so
   the conversation continues instead of restarting cold.
4. If no binding exists → pick any online host, write the binding,
   forward.

Host disconnect does **not** release the binding — that is what lets
a fresh worker take over the same session on the next inbound. The
"worker disconnected, in-flight turn lost" surface only fires when
there is no replacement host *and* the chat is mid-turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .session_bindings import SessionBinding, SessionBindingStore

log = logging.getLogger("agentm_channels.worker_registry")


@dataclass
class WorkerInfo:
    """One connected ``agent_worker`` peer's advertised capabilities."""

    peer_id: str
    scenarios: frozenset[str]
    cwd: str | None = None
    raw_capabilities: dict[str, Any] = field(default_factory=dict)


class WorkerRegistry:
    """Online-host set + persistent session bindings.

    Single asyncio loop owns the only instance; ``WireBridge`` calls
    in on hello / disconnect / inbound dispatch. SQLite work happens
    on the asyncio thread — callers offload to ``asyncio.to_thread``
    if they care about not blocking the loop.
    """

    def __init__(self, bindings: SessionBindingStore | None = None) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        # Default to an in-memory store so tests / quick scripts can
        # construct a registry without filesystem state. Production
        # (cli.py) supplies a disk-backed store.
        self._bindings = bindings or SessionBindingStore(":memory:")

    # -- host lifecycle ----------------------------------------------

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
            "host online peer_id=%s scenarios=%s cwd=%s",
            peer_id,
            sorted(scenarios),
            cwd,
        )

    def unregister(self, peer_id: str) -> set[str]:
        """Drop the host from the online set. Returns the set of
        ``session_key`` that *had been* bound to it — useful for
        diagnostics. The binding rows themselves stay in the store
        so a new host can pick the session up on the next inbound.
        """
        self._workers.pop(peer_id, None)
        stranded = {b.session_key for b in self._bindings.all_for_host(peer_id)}
        if stranded:
            log.info(
                "host offline peer_id=%s stranded_sessions=%s (bindings retained)",
                peer_id,
                sorted(stranded),
            )
        return stranded

    # -- routing -----------------------------------------------------

    def find_host(
        self, scenario: str, session_key: str
    ) -> tuple[str | None, str | None]:
        """Resolve the host that should receive an inbound for
        ``session_key``. Returns ``(host_peer_id, resume_id)``:

        * ``host_peer_id is None`` — no online host advertises
          ``scenario`` (and no fallback). The bridge should emit a
          "no host available" outbound.
        * ``host_peer_id`` set, ``resume_id`` set — the session was
          previously bound; the host (this one, or another) should
          resume the prior AgentSession from ``resume_id``.
        * ``host_peer_id`` set, ``resume_id`` None — first-time
          binding for this session_key (or the prior host never
          reported a session_id).

        Side effect: writes the binding when a rebind or first-bind
        happens, so subsequent inbound traffic stays sticky.
        """
        binding = self._bindings.get(session_key)
        if binding is not None and binding.host_id in self._workers:
            # Happy path: bound + online.
            return binding.host_id, binding.resume_id

        # Need to (re)bind. Pick any online host advertising the scenario.
        for peer_id, info in self._workers.items():
            if scenario in info.scenarios:
                # Preserve the existing resume_id if any — that is the
                # whole point of persistent bindings. Pass resume_id=None
                # to `upsert` so it doesn't clobber.
                self._bindings.upsert(session_key, peer_id, resume_id=None)
                return peer_id, binding.resume_id if binding is not None else None
        return None, None

    def find_worker(self, scenario: str, session_key: str) -> str | None:
        """Backward-compatible accessor — returns just the host peer_id
        (drops the resume_id half of ``find_host``). New code should
        prefer ``find_host`` so it can pass the resume_id through to
        the destination."""
        host_id, _ = self.find_host(scenario, session_key)
        return host_id

    def record_resume_id(self, session_key: str, resume_id: str) -> None:
        """Workers report their AgentSession id back via outbound
        metadata; the bridge funnels it here so future rebinds know
        what to resume."""
        self._bindings.set_resume_id(session_key, resume_id)

    # -- introspection (tests, logging) ------------------------------

    def workers(self) -> list[WorkerInfo]:
        return list(self._workers.values())

    def binding(self, session_key: str) -> SessionBinding | None:
        return self._bindings.get(session_key)

    def sticky_owner(self, session_key: str) -> str | None:
        """Diagnostic accessor: which host is currently bound to
        ``session_key``? Returns ``None`` if no binding exists. The
        host might be offline — callers that need "currently routable"
        should use ``find_host`` instead, which transparently rebinds."""
        b = self._bindings.get(session_key)
        return b.host_id if b is not None else None

    def has(self, peer_id: str) -> bool:
        return peer_id in self._workers


__all__ = ["WorkerInfo", "WorkerRegistry"]
