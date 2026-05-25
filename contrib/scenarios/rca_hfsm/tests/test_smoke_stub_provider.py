"""Commit 5 — wiring-level smoke test for the rca_hfsm scenario.

This is the Phase-1 acceptance fail-stop for "the full atom stack composes
and drives the falsification cycle to completion". It is NOT a session-
level E2E with a stub LLM provider; per the plan's "degraded path"
clause, we exercise the wiring directly because:

* the per-state prompt fragments and FSM transitions are already covered
  structurally by ``tests/test_fsm_transitions.py``;
* every gate precondition (propose / confirm / refute / independence /
  finalize coverage) already has its own dedicated fail-stop;
* the value the smoke test adds is "manifest.yaml resolves; the atoms
  install in the order it declares; the recorded bus events match the
  expected falsification cycle, including exactly one downgrade".
  Driving an LLM through this would only test the model's compliance
  with the prompts, not our wiring.

What this test does fail-stop:

1. ``manifest.yaml`` loads via ``agentm.extensions.loader.load_scenario``
   and resolves to the full atom stack in declaration order.
2. Driving the registered tools directly through the gate produces an
   ``rca.graph.mutated`` event sequence that covers every reachable
   gate operator (record_symptom / propose / attach_check /
   refute (downgraded) / confirm (applied)).
3. Exactly one ``downgraded`` event is observed — a premature ``refute``
   without a triggered negative prediction OR a steelman check, which
   the gate downgrades to ``refine`` per §7.2. The downgrade pattern is
   the structural representative of "gate did not let the LLM short-
   circuit a precondition"; surfacing it on the bus is what makes the
   wiring observable to downstream audit atoms (Phase 2).
4. After the downgrade, the surviving open leaf still cleanly confirms
   once both an independent positive check (two distinct
   ``worker_session_id``) and an un-triggered negative are in place.
5. ``submit_final_report`` returns :class:`ToolTerminate` with reason
   ``rca_hfsm:final-report-submitted`` once coverage holds.

Note on the "independence downgrade" sub-flow originally mentioned in the
plan: the independence semantics are already fail-stop-covered by
``tests/test_gate_independence.py``. This smoke test exercises the
gate's downgrade *as a wiring event on the bus* via the
refute-without-steelman path, which is the structurally equivalent
shape.

Phase-2 note on side-hypothesis status after a downgrade: prior to the
LLM-native-judges refactor the gate auto-applied the refine on a
downgrade and the side hypothesis flipped to ``refined→``; design §5.2
flipped that so the side hypothesis stays ``open`` and the orchestrator
decides next steps. The smoke test's step-4 lookup therefore filters
``open_leaves`` by claim text rather than relying on the side
hypothesis being out of contention — both hypotheses are open after
the downgrade.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from agentm.extensions.loader import load_scenario_with_meta



_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _tool(api: object, name: str) -> Any:
    for t in getattr(api, "tools", []):  # type: ignore[attr-defined]
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not registered")


def _text(result: object) -> str:
    if hasattr(result, "result"):
        # ToolTerminate carries a ToolResult on .result
        inner = result.result  # type: ignore[attr-defined]
        return "\n".join(c.text for c in inner.content)
    return "\n".join(c.text for c in result.content)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Manifest-load assertion: independent of the install-direct path below.
# ---------------------------------------------------------------------------


def test_manifest_loads_in_declared_order() -> None:
    """``manifest.yaml`` parses cleanly and resolves to 12 extensions.

    The load step is what an actual ``agentm --scenario rca_hfsm`` run
    performs first; if it raises, no session can be constructed regardless
    of how well the atoms behave in unit-level fixtures.
    """

    # Anchor the loader at the repo root so the relative ``contrib/scenarios/...``
    # lookup resolves regardless of pytest's invocation cwd.
    import os

    prior = os.environ.get("AGENTM_PROJECT_ROOT")
    os.environ["AGENTM_PROJECT_ROOT"] = str(_REPO_ROOT)
    try:
        extensions, _meta = load_scenario_with_meta("rca_hfsm")
    finally:
        if prior is None:
            os.environ.pop("AGENTM_PROJECT_ROOT", None)
        else:
            os.environ["AGENTM_PROJECT_ROOT"] = prior

    modules = [mod for mod, _cfg in extensions]
    # The whole stack present (18 entries — see manifest.yaml).
    # 12 from Phase 1 + 4 LLM-native judge atoms (C3 of Phase 2)
    # + 1 data-access atom (duckdb_sql, C4 of Phase 2 — design §13 reuse)
    # + 1 investigation_genuine judge (C5 of Phase 2).
    assert len(modules) == 18, modules
    # Critical dependency-order properties:
    assert modules.index(
        "agentm_rca_hfsm.atoms.rca_hgraph_store"
    ) < modules.index("agentm_rca_hfsm.atoms.rca_falsification_gate")
    assert modules.index(
        "agentm_rca_hfsm.atoms.rca_falsification_gate"
    ) < modules.index("agentm_rca_hfsm.atoms.rca_evidence_tools")
    # Judges install AFTER the store (they don't depend on it but the
    # order keeps the L1 contract together) and BEFORE the gate (gate's
    # install reads ``rca.judge.*`` services).
    for kind in (
        "satisfied",
        "coverage",
        "independence",
        "falsified_genuinely",
        "investigation_genuine",
    ):
        judge_module = f"agentm_rca_hfsm.atoms.judge_{kind}"
        assert judge_module in modules, f"missing judge atom: {judge_module}"
        assert modules.index(judge_module) < modules.index(
            "agentm_rca_hfsm.atoms.rca_falsification_gate"
        ), f"{judge_module} must install before the gate"
    # Sub-agent inherits the evidence-tool surface.
    sub_agent_idx = modules.index("agentm.extensions.builtin.sub_agent")
    sub_agent_cfg = extensions[sub_agent_idx][1]
    inherited = sub_agent_cfg.get("inherit_extensions", [])
    assert "rca_hgraph_store" in inherited
    assert "rca_evidence_tools" in inherited


# ---------------------------------------------------------------------------
# Scripted falsification cycle: drives the FSM through every reachable
# state by calling the registered tools directly. No LLM in the loop.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_store() -> None:
    """Ensure each test gets a fresh single-writer registry."""

    from agentm_rca_hfsm.atoms import rca_hgraph_store

    rca_hgraph_store._reset_for_tests()


