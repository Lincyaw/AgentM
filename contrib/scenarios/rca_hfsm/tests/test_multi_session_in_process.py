"""Regression test for #156 — multiple sessions in one process.

Until 2026-05-14 the store atom's claim registry was treated as a
process-wide one-shot lock (``if _claimed: raise``). The first session
bootstrap consumed the lock; every subsequent bootstrap in the same
process raised in ``claim_write_handle``, the falsification gate's
``install`` propagated that as ``RuntimeError`` into the substrate's
silent-swallow path (``session_factory._install_with_events``'s
``except``), and the cascade of "rca.gate missing" errors silently
stripped the five ``rca_evidence_tools`` tools from the LLM's catalog.

The fix made the one-shot rule per-token rather than per-process — each
``install`` mints a fresh token and only that token is consumed on claim.
This file fails if that rule regresses, without depending on the rest
of the smoke-test wiring.
"""

from __future__ import annotations

from typing import Any

from agentm_rca_hfsm.atoms import (
    rca_evidence_tools,
    rca_falsification_gate,
    rca_hgraph_store,
)

from tests._gate_fixtures import StubAPI
from tests._phase1_mimic_judges import all_mimics


def _install_session_stack(api: StubAPI) -> None:
    """Drive the same install ladder the manifest declares.

    Mounts store → judges → gate → evidence tools without touching the
    module-level ``_reset_for_tests`` hook. A second call in the same
    process must succeed without resetting state.
    """

    rca_hgraph_store.install(api, {})
    for service_name, mimic in all_mimics().items():
        api.set_service(service_name, mimic)
    rca_falsification_gate.install(api, {})
    rca_evidence_tools.install(api, {})


def test_second_session_in_process_keeps_all_evidence_tools() -> None:
    rca_hgraph_store._reset_for_tests()

    api_a = StubAPI()
    _install_session_stack(api_a)
    tools_a = {t.name for t in api_a.tools}

    # Second session in the same process — no reset between installs.
    api_b = StubAPI()
    _install_session_stack(api_b)
    tools_b = {t.name for t in api_b.tools}

    evidence_tools = {
        "record_symptom",
        "record_observation",
        "propose_hypothesis",
        "attach_check",
        "propose_update",
    }
    assert evidence_tools.issubset(tools_a), (
        f"first session missing tools: {evidence_tools - tools_a}"
    )
    assert evidence_tools.issubset(tools_b), (
        f"second session lost evidence tools (issue #156 regression): "
        f"{evidence_tools - tools_b}"
    )

    # Each session must have published its own ``rca.gate`` instance.
    gate_a = api_a.get_service("rca.gate")
    gate_b = api_b.get_service("rca.gate")
    assert gate_a is not None and gate_b is not None
    assert gate_a is not gate_b


def test_eight_sequential_sessions_each_install_cleanly() -> None:
    """Stress the per-token rule: eight sessions, every gate must install."""

    rca_hgraph_store._reset_for_tests()

    tokens: list[Any] = []
    for _ in range(8):
        api = StubAPI()
        _install_session_stack(api)
        token = api.get_service("rca.hgraph.write_token")
        assert isinstance(token, str) and token
        tokens.append(token)
        # The gate must have claimed (and therefore consumed) this token.
        assert api.get_service("rca.gate") is not None

    assert len(set(tokens)) == 8, "tokens must be unique across installs"
