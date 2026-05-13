"""Commit 3 — every evidence tool routes through the falsification gate.

One sub-test per tool. Each verifies that:

* The tool builds a well-formed ``UpdateProposal``.
* The gate's ``UpdateResult`` is reflected in ``ToolResult.text``
  (``status=applied|downgraded|rejected`` plus the reason on the latter
  two paths).
* The graph state mutates (or doesn't, on downgrade / rejection) as
  expected.

Tests use ``install_full_stack`` from ``_gate_fixtures`` so the wiring
mirrors what the scenario manifest will produce.
"""

from __future__ import annotations

import asyncio

from tests._gate_fixtures import install_full_stack


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _tool(api: object, name: str) -> object:
    for t in getattr(api, "tools", []):  # type: ignore[attr-defined]
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not registered")


def _text(result: object) -> str:
    chunks = [c.text for c in result.content]  # type: ignore[attr-defined]
    return "\n".join(chunks)


def test_record_symptom_routes_through_gate() -> None:
    api, _, read = install_full_stack()
    tool = _tool(api, "record_symptom")

    result = _run(tool.execute({"text": "disk fills at 14:32 UTC", "source": "user_intake"}))  # type: ignore[attr-defined]

    text = _text(result)
    assert text.startswith("status=applied"), text
    assert len(read.get_symptoms()) == 1
    assert read.get_symptoms()[0].text == "disk fills at 14:32 UTC"


def test_record_symptom_rejects_empty_text() -> None:
    api, _, read = install_full_stack()
    tool = _tool(api, "record_symptom")

    result = _run(tool.execute({"text": "   "}))  # type: ignore[attr-defined]

    text = _text(result)
    assert text.startswith("status=rejected"), text
    assert read.get_symptoms() == []


def test_record_observation_routes_through_gate() -> None:
    api, _, read = install_full_stack()
    tool = _tool(api, "record_observation")

    result = _run(
        tool.execute(  # type: ignore[attr-defined]
            {
                "text": "disk usage 99%",
                "source_tool_call": "duckdb_sql-1",
                "related_symptoms": ["S1"],
                "related_predictions": [],
            }
        )
    )

    text = _text(result)
    assert text.startswith("status=applied"), text
    # The observation lives in the log; the store's by-signature index
    # makes the cache atom able to find it later.
    sig_obs = read.get_observation_by_signature  # store helper
    # Recompute the signature the tool used.
    from agentm_rca_hfsm.atoms.rca_evidence_tools import _tool_signature

    sig = _tool_signature("duckdb_sql-1", {"text": "disk usage 99%"})
    assert sig_obs(sig) is not None


def test_propose_hypothesis_with_negative_prediction_applies() -> None:
    api, _, read = install_full_stack()
    tool = _tool(api, "propose_hypothesis")

    result = _run(
        tool.execute(  # type: ignore[attr-defined]
            {
                "claim": "logrotate failed",
                "predictions": [
                    {"claim": "logrotate.service inactive", "polarity": "positive"},
                    {"claim": "no rotated archive newer than 7d", "polarity": "negative"},
                ],
            }
        )
    )

    text = _text(result)
    assert text.startswith("status=applied"), text
    # The hypothesis id is opaque (uuid suffix); fetch via open leaves.
    leaves = read.get_open_leaves()
    assert len(leaves) == 1
    assert leaves[0].claim == "logrotate failed"


def test_propose_hypothesis_without_negative_is_applied_under_judge_gate() -> None:
    """Phase 2: propose-time no longer enforces "≥1 negative prediction".

    The structural rule moved into ``rca.judge.falsified_genuinely`` at
    confirm-time (design §4.4). Propose now only enforces shape rules,
    so a positive-only hypothesis IS applied; the equivalent rejection
    fires later when the orchestrator tries to confirm it.
    """

    api, _, read = install_full_stack()
    tool = _tool(api, "propose_hypothesis")

    result = _run(
        tool.execute(  # type: ignore[attr-defined]
            {
                "claim": "logrotate failed",
                "predictions": [
                    {"claim": "logrotate.service inactive", "polarity": "positive"},
                ],
            }
        )
    )

    text = _text(result)
    assert text.startswith("status=applied"), text
    leaves = read.get_open_leaves()
    assert len(leaves) == 1
    assert leaves[0].claim == "logrotate failed"


def test_attach_check_records_observations_and_check() -> None:
    api, _, read = install_full_stack()
    propose = _tool(api, "propose_hypothesis")
    attach = _tool(api, "attach_check")

    _run(
        propose.execute(  # type: ignore[attr-defined]
            {
                "claim": "H",
                "predictions": [
                    {"claim": "p-pos", "polarity": "positive"},
                    {"claim": "p-neg", "polarity": "negative"},
                ],
            }
        )
    )
    leaf = read.get_open_leaves()[0]
    pos_pred = next(p for p in leaf.predictions if p.polarity == "positive")

    result = _run(
        attach.execute(  # type: ignore[attr-defined]
            {
                "hypothesis_id": leaf.id,
                "prediction_id": pos_pred.id,
                "worker_session_id": "w1",
                "mode": "verify",
                "observations": [
                    {
                        "text": "logrotate.service is inactive",
                        "source_tool_call": "sql-1",
                        "related_symptoms": ["S1"],
                    }
                ],
                "interpretation": {
                    "proposed_update": "confirm H",
                    "reasoning": "observation matches the claim",
                    "confidence": "high",
                },
                "verdict_proposal": "observations support the prediction",
            }
        )
    )

    text = _text(result)
    assert text.startswith("status=applied"), text
    # Observation is in the log; check is on the prediction.
    fresh_leaf = read.get_hypothesis(leaf.id)
    assert fresh_leaf is not None
    assert len(fresh_leaf.predictions[0].checks) + len(fresh_leaf.predictions[1].checks) == 1


def test_propose_update_dispatches_explicit_op() -> None:
    api, _, read = install_full_stack()
    propose = _tool(api, "propose_hypothesis")
    explicit = _tool(api, "propose_update")

    _run(
        propose.execute(  # type: ignore[attr-defined]
            {
                "claim": "H",
                "predictions": [
                    {"claim": "p-neg", "polarity": "negative"},
                ],
            }
        )
    )
    target = read.get_open_leaves()[0]

    # Confirm without supporting evidence — gate downgrades to refine.
    result = _run(
        explicit.execute({"op": "confirm", "target_id": target.id})  # type: ignore[attr-defined]
    )
    text = _text(result)
    assert text.startswith("status=downgraded"), text
    assert "to=refine" in text
    assert "reason=" in text


def test_propose_update_rejects_unknown_op() -> None:
    api, _, _ = install_full_stack()
    tool = _tool(api, "propose_update")

    result = _run(tool.execute({"op": "explode", "target_id": "X"}))  # type: ignore[attr-defined]
    text = _text(result)
    assert text.startswith("status=rejected"), text
    assert "unknown op" in text


def test_propose_update_refuses_record_observation_alias() -> None:
    api, _, _ = install_full_stack()
    tool = _tool(api, "propose_update")

    result = _run(tool.execute({"op": "record_observation"}))  # type: ignore[attr-defined]
    text = _text(result)
    assert text.startswith("status=rejected"), text
    assert "dedicated tool" in text
