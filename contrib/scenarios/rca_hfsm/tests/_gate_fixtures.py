"""Shared test fixtures for the falsification-gate test files.

Each test file owns one tiny scenario; this module hosts the
``_StubAPI`` and the ``install_store_and_gate`` helper they all reuse so the
test bodies stay focused on the gate behaviour rather than wiring.
"""

from __future__ import annotations

from typing import Any

from agentm_rca_hfsm.atoms import rca_falsification_gate, rca_hgraph_store


class StubAPI:
    """Minimal ``ExtensionAPI`` shim covering only ``set_service`` /
    ``get_service``. Both store and gate atoms touch nothing else at install
    time (the gate publishes ``rca.gate``, the store publishes
    ``rca.hgraph.read`` and ``rca.hgraph.write_token``).
    """

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    def set_service(self, name: str, obj: Any) -> None:
        self._services[name] = obj

    def get_service(self, name: str) -> Any:
        return self._services.get(name)


def install_store_and_gate() -> tuple[StubAPI, Any, Any]:
    """Return ``(api, gate, read_handle)`` after wiring store + gate.

    Resets the store's module-level claim registry first so test ordering
    cannot cross-contaminate. Returns the gate instance directly (via the
    ``rca.gate`` service) — every gate test pokes at ``gate.apply(...)``.
    """

    rca_hgraph_store._reset_for_tests()
    api = StubAPI()
    rca_hgraph_store.install(api, {})
    rca_falsification_gate.install(api, {})
    gate = api.get_service("rca.gate")
    read = api.get_service("rca.hgraph.read")
    return api, gate, read
