"""Commit 4 — acceptance #4: ingesting a WorkerReturn is interpretation-independent.

The orchestrator (L2) by default consumes only ``observations`` and
re-derives the update decision; ``interpretation`` lives in the trace for
audit but does not enter the graph automatically. This is the structural
anti-bias core (design §6).

Property: ingesting a ``WorkerReturn`` twice — once with full
``interpretation``, once with ``interpretation`` blanked — yields equal
L1 state (modulo observation-id randomness, which we eliminate by reusing
the same observation payload across both runs).
"""

from __future__ import annotations

from agentm_rca.hfsm.schema import Interpretation, Observation, WorkerReturn
from agentm_rca.hfsm.worker_return import (
    ingest_observations,
)

from tests.hfsm._gate_fixtures import install_with_fsm


def _make_wr(*, interpretation_filled: bool) -> WorkerReturn:
    """Build a ``WorkerReturn`` reused across both runs with stable obs ids.

    The two runs differ ONLY in ``interpretation`` — observation ids and
    payloads are pinned so the post-ingest graph state is structurally
    comparable.
    """

    obs = [
        Observation(
            id="O-stable-1",
            text="logrotate.service is inactive",
            source_tool_call="sql-1",
            tool_signature="sig-1",
            related_symptoms=["S1"],
            related_predictions=["P1"],
            ts=1.0,
        ),
        Observation(
            id="O-stable-2",
            text="last rotation 2026-04-01",
            source_tool_call="sql-2",
            tool_signature="sig-2",
            related_symptoms=["S1"],
            related_predictions=["P1"],
            ts=2.0,
        ),
    ]
    interp = Interpretation(
        proposed_update="confirm H — observations clearly back the claim",
        reasoning="strong correlation, no contradictions found",
        confidence="high",
    ) if interpretation_filled else Interpretation(
        proposed_update="",
        reasoning="",
        confidence="",
    )
    return WorkerReturn(observations=obs, interpretation=interp)


def _snapshot_graph(read: object) -> tuple[object, ...]:
    """Return a comparable tuple of L1 state.

    Compares observation ids, symptoms, hypotheses by id. Interpretation
    explicitly excluded — it does not enter the graph, so any difference
    on the orchestrator-side would indicate the structural separation
    has leaked.
    """

    obs_sigs = sorted(
        (o.id, o.text, o.tool_signature)
        for o in read._state.observations  # type: ignore[attr-defined]
    )
    symptoms = sorted((s.id, s.text) for s in read.get_symptoms())  # type: ignore[attr-defined]
    hypotheses = sorted(
        (h.id, h.claim, h.status)
        for h in read._state.hypotheses.values()  # type: ignore[attr-defined]
    )
    return (tuple(obs_sigs), tuple(symptoms), tuple(hypotheses))




def test_graph_state_is_interpretation_independent() -> None:
    """The L1 graph after ingest is equal across full vs blank interpretation."""

    api_a, gate_a, read_a, _fsm_a = install_with_fsm()
    api_b, gate_b, read_b, _fsm_b = install_with_fsm()

    wr_full = _make_wr(interpretation_filled=True)
    wr_blank = _make_wr(interpretation_filled=False)

    ids_a = ingest_observations(gate_a, wr_full)
    ids_b = ingest_observations(gate_b, wr_blank)

    assert ids_a == ids_b, (ids_a, ids_b)
    assert _snapshot_graph(read_a) == _snapshot_graph(read_b)
