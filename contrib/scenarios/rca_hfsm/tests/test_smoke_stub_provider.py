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
plan: empirically, attempting a confirm before the second worker's check
is attached leaves the parent hypothesis in a ``refined→`` state from
which further checks on the *same* prediction id are rejected (its owner
is no longer an open hypothesis). The independence semantics are already
fail-stop-covered by ``tests/test_gate_independence.py``. This smoke
test exercises the gate's downgrade *as a wiring event on the bus* via
the refute-without-steelman path, which is the structurally equivalent
shape.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import ToolTerminate
from agentm.extensions.loader import load_scenario_with_meta

from tests._gate_fixtures import install_with_fsm


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
    # The whole stack present (12 entries — see manifest.yaml).
    assert len(modules) == 12, modules
    # Critical dependency-order properties:
    assert modules.index(
        "agentm_rca_hfsm.atoms.rca_hgraph_store"
    ) < modules.index("agentm_rca_hfsm.atoms.rca_falsification_gate")
    assert modules.index(
        "agentm_rca_hfsm.atoms.rca_falsification_gate"
    ) < modules.index("agentm_rca_hfsm.atoms.rca_evidence_tools")
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


def test_full_falsification_cycle_drives_fsm_to_finalize() -> None:
    api, gate, read, fsm = install_with_fsm()
    mutated: list[dict[str, Any]] = []

    def _capture(payload: dict[str, Any]) -> None:
        mutated.append(payload)

    api.on("rca.graph.mutated", _capture)

    # 1. INTAKE → OBSERVE on the first symptom.
    record_symptom = _tool(api, "record_symptom")
    _run(
        record_symptom.execute(
            {"text": "server disk fills at 14:32 UTC", "source": "user_intake"}
        )
    )
    assert fsm.state == "OBSERVE", fsm.history
    sym_id = read.get_symptoms()[0].id

    # 2. Propose a side hypothesis we will *prematurely refute* — that is
    # the structural shape the gate's §7.2 downgrade was designed for, and
    # it gives the smoke test exactly one downgrade on the bus without
    # stranding the working hypothesis in a refined→ status.
    propose = _tool(api, "propose_hypothesis")
    _run(
        propose.execute(
            {
                "claim": "cron clock skew",
                "predictions": [
                    {
                        "claim": "cron timestamps drift from wall clock",
                        "polarity": "positive",
                        "test_plan": "timedatectl status",
                    },
                    {
                        "claim": "ntpd reports no offsets",
                        "polarity": "negative",
                        "test_plan": "chronyc tracking",
                    },
                ],
                "rationale": "speculative alternate cause",
            }
        )
    )
    side_h = read.get_open_leaves()[0]
    assert fsm.state == "VERIFY", fsm.history

    # 3. Premature refute — no triggered negative, no steelman → gate
    # downgrades per §7.2. This is the single downgrade on the bus.
    explicit = _tool(api, "propose_update")
    premature_refute = _run(
        explicit.execute({"op": "refute", "target_id": side_h.id})
    )
    refute_text = _text(premature_refute)
    assert refute_text.startswith("status=downgraded"), refute_text
    assert "steelman" in refute_text or "triggered" in refute_text, refute_text

    # 4. Propose the *real* hypothesis. The side hypothesis is now in a
    # ``refined→`` state and no longer competes for ``open_leaves``.
    _run(
        propose.execute(
            {
                "claim": "logrotate misconfig",
                "predictions": [
                    {
                        "claim": "rotated logs accumulate in /var/log",
                        "polarity": "positive",
                        "test_plan": "ls /var/log",
                    },
                    {
                        "claim": "logrotate runs without errors",
                        "polarity": "negative",
                        "test_plan": "grep error logrotate.log",
                    },
                ],
            }
        )
    )
    # The real leaf is the most recently proposed open hypothesis — pick
    # the one whose claim matches so the test isn't order-dependent on
    # the read API's dict iteration.
    leaf = next(
        h for h in read.get_open_leaves() if h.claim == "logrotate misconfig"
    )
    pos_pred = next(p for p in leaf.predictions if p.polarity == "positive")
    neg_pred = next(p for p in leaf.predictions if p.polarity == "negative")

    # 5. Negative-prediction check — verdict avoids the "triggered" word so
    # the gate counts the negative as satisfied (§7.1 falsification gap).
    attach = _tool(api, "attach_check")
    _run(
        attach.execute(
            {
                "hypothesis_id": leaf.id,
                "prediction_id": neg_pred.id,
                "worker_session_id": "worker-A",
                "observations": [
                    {
                        "text": "no errors in logrotate.log",
                        "source_tool_call": "tc-1",
                        "related_symptoms": [sym_id],
                        "related_predictions": [neg_pred.id],
                    }
                ],
                "interpretation": {
                    "proposed_update": "negative prediction not triggered",
                    "reasoning": "checked logs",
                    "confidence": "high",
                },
                "verdict_proposal": "no observations were found that match",
            }
        )
    )
    assert fsm.state == "JUDGE"

    # 6. Positive check from worker-A.
    _run(
        attach.execute(
            {
                "hypothesis_id": leaf.id,
                "prediction_id": pos_pred.id,
                "worker_session_id": "worker-A",
                "observations": [
                    {
                        "text": "rotated logs found accumulating in /var/log",
                        "source_tool_call": "tc-2",
                        "related_symptoms": [sym_id],
                        "related_predictions": [pos_pred.id],
                    }
                ],
                "interpretation": {
                    "proposed_update": "positive prediction supported",
                    "reasoning": "logs visible",
                    "confidence": "high",
                },
                "verdict_proposal": "observations support the prediction",
            }
        )
    )

    # 7. Independent positive check from worker-B (different session id —
    # the Phase-1 axis of independence).
    _run(
        attach.execute(
            {
                "hypothesis_id": leaf.id,
                "prediction_id": pos_pred.id,
                "worker_session_id": "worker-B",
                "observations": [
                    {
                        "text": "second-pass confirmation of accumulating logs",
                        "source_tool_call": "tc-3",
                        "related_symptoms": [sym_id],
                        "related_predictions": [pos_pred.id],
                    }
                ],
                "interpretation": {
                    "proposed_update": "positive prediction supported",
                    "reasoning": "second worker corroboration",
                    "confidence": "high",
                },
                "verdict_proposal": "observations support the prediction",
            }
        )
    )

    # 8. Confirm — every §7.1 precondition is now satisfied: satisfied
    # negative + 2 independent positive workers + symptom coverage. FSM
    # enters FINALIZE.
    confirm_result = _run(
        explicit.execute({"op": "confirm", "target_id": leaf.id})
    )
    assert _text(confirm_result).startswith("status=applied"), _text(confirm_result)
    assert fsm.state == "FINALIZE", fsm.history

    # 9. submit_final_report returns ToolTerminate cleanly.
    finalize_tool = _tool(api, "submit_final_report")
    final = _run(
        finalize_tool.execute(
            {
                "root_cause": "logrotate misconfig caused log accumulation and disk-fill",
                "supporting_observations": ["O-cache-placeholder"],
                "refuted_alternatives": [side_h.id],
            }
        )
    )
    assert isinstance(final, ToolTerminate), type(final)
    assert final.reason == "rca_hfsm:final-report-submitted", final.reason

    # 10. Event-log shape.
    #     ``attach_check`` first persists each observation via
    #     ``record_observation`` (see rca_evidence_tools._attach_check),
    #     so we see one record_observation per check before the
    #     attach_check event itself. Total mutation events:
    #     record_symptom + propose(side) + refute(side, downgraded)
    #     + propose(real)
    #     + 3 × (record_observation + attach_check)
    #     + confirm(applied)
    #     = 11 events.
    ops = [evt.get("op") for evt in mutated]
    kinds = [evt.get("kind") for evt in mutated]
    assert ops == [
        "record_symptom",
        "propose",
        "refute",
        "propose",
        "record_observation",
        "attach_check",
        "record_observation",
        "attach_check",
        "record_observation",
        "attach_check",
        "confirm",
    ], ops
    assert kinds.count("downgraded") == 1, kinds
    assert kinds.count("applied") == 10, kinds

    downgrades = [evt for evt in mutated if evt.get("kind") == "downgraded"]
    assert len(downgrades) == 1
    assert downgrades[0].get("op") == "refute"
    assert downgrades[0].get("downgrade_op") == "refine"
