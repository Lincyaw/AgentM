"""Shared test fixtures for the rca_hfsm scenario tests.

Each test file owns one tiny scenario; this module hosts the
``_StubAPI``, the ``install_store_and_gate`` helper, and the slightly richer
``install_full_stack`` helper that adds the evidence-tools + observation-cache
atoms (commit 3). Tests rely on these to stay focused on behaviour rather
than wiring.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentm.core.abi import Tool

from agentm_rca_hfsm.atoms import (
    rca_brief_builder,
    rca_evidence_tools,
    rca_falsification_gate,
    rca_finalize,
    rca_fsm_policy,
    rca_hgraph_store,
    rca_observation_cache,
)

from tests._phase1_mimic_judges import all_mimics


class StubAPI:
    """Minimal ``ExtensionAPI`` shim.

    Covers the surface every scenario atom actually touches at install /
    agent_start time:

    * ``set_service`` / ``get_service`` — service publication.
    * ``register_tool`` / ``tools`` — tool catalog (a plain list the cache
      atom mutates by index, mirroring the real ``api.tools`` contract).
    * ``on`` / ``events.emit`` — event subscription + sync emission. The
      stub's ``StubEventBus`` records every emitted ``DiagnosticEvent`` so
      tests can assert on the cache's ``tool_call_cached`` diagnostic.
    """

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}
        self._handlers: dict[str, list[Any]] = {}
        self.tools: list[Tool] = []
        self.events = StubEventBus(self._handlers)

    # -- Services ----------------------------------------------------------

    def set_service(self, name: str, obj: Any) -> None:
        self._services[name] = obj

    def get_service(self, name: str) -> Any:
        return self._services.get(name)

    # -- Tools -------------------------------------------------------------

    def register_tool(self, tool: Tool) -> None:
        self.tools.append(tool)

    # -- Events ------------------------------------------------------------

    def on(self, channel: str, handler: Any, **_: Any) -> Any:
        self._handlers.setdefault(channel, []).append(handler)
        return lambda: None


class StubEventBus:
    """Records every emitted event by channel so tests can assert."""

    def __init__(self, handlers: dict[str, list[Any]]) -> None:
        self._handlers = handlers
        self.emitted: list[tuple[str, Any]] = []

    async def emit(self, channel: str, event: Any) -> list[Any]:
        self.emitted.append((channel, event))
        results: list[Any] = []
        for handler in list(self._handlers.get(channel, [])):
            res = handler(event)
            if asyncio.iscoroutine(res):
                res = await res
            results.append(res)
        return results

    def emit_sync(self, channel: str, event: Any) -> list[Any]:
        self.emitted.append((channel, event))
        results: list[Any] = []
        for handler in list(self._handlers.get(channel, [])):
            res = handler(event)
            if asyncio.iscoroutine(res):
                # The stub does not run a loop here; tests that need async
                # handlers should use the async ``emit`` path.
                continue
            results.append(res)
        return results

    def fire_handlers(self, channel: str, event: Any) -> list[Any]:
        """Test-only helper: invoke handlers without recording the event.

        Used to fire ``AgentStartEvent`` so the cache atom wraps registered
        tools without polluting ``emitted`` with a synthetic kernel event.
        """

        results: list[Any] = []
        for handler in list(self._handlers.get(channel, [])):
            res = handler(event)
            if asyncio.iscoroutine(res):
                results.append(asyncio.get_event_loop().run_until_complete(res))
            else:
                results.append(res)
        return results


def install_store_and_gate() -> tuple[StubAPI, Any, Any]:
    """Return ``(api, gate, read_handle)`` after wiring store + gate.

    Resets the store's module-level claim registry first so test ordering
    cannot cross-contaminate. Mounts the Phase-1 mimic judges so the gate
    can install (it requires the 4 ``rca.judge.*`` services to be
    published before its own install runs) AND so the post-refactor gate
    behaves identically to the Phase-1 gate on Phase-1 fail-stop test
    inputs (design §8.2 behavior-preservation acceptance). Returns the
    gate instance directly (via the ``rca.gate`` service) — every gate
    test pokes at ``gate.apply(...)``.
    """

    rca_hgraph_store._reset_for_tests()
    api = StubAPI()
    rca_hgraph_store.install(api, {})
    for service_name, mimic in all_mimics().items():
        api.set_service(service_name, mimic)
    rca_falsification_gate.install(api, {})
    gate = api.get_service("rca.gate")
    read = api.get_service("rca.hgraph.read")
    return api, gate, read


def install_full_stack() -> tuple[StubAPI, Any, Any]:
    """Return ``(api, gate, read_handle)`` after wiring the full commit-3 stack.

    Adds ``rca_evidence_tools`` (registers the five LLM-facing tools) and
    ``rca_observation_cache`` (subscribes to ``agent_start``) on top of the
    store + gate. Tests that drive evidence tools or the cache go through
    this helper.
    """

    api, gate, read = install_store_and_gate()
    rca_evidence_tools.install(api, {})
    rca_observation_cache.install(api, {})
    return api, gate, read


def install_with_fsm() -> tuple[StubAPI, Any, Any, Any]:
    """Return ``(api, gate, read_handle, fsm)`` after wiring the commit-4 stack.

    Adds ``rca_fsm_policy``, ``rca_brief_builder``, and ``rca_finalize`` on top
    of :func:`install_full_stack`. Tests that exercise FSM transitions or the
    finalize coverage check use this helper.
    """

    api, gate, read = install_full_stack()
    rca_fsm_policy.install(api, {})
    rca_brief_builder.install(api, {})
    rca_finalize.install(api, {})
    fsm = api.get_service("rca.fsm")
    return api, gate, read, fsm
