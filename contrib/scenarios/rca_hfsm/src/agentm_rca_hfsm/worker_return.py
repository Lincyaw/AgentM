"""Pure helpers for the worker → orchestrator two-column contract (design §6).

The worker returns ``observations`` (facts) and ``interpretation`` (advice)
as two separate columns. The orchestrator (L2) ingests observations into
L1 unconditionally and **re-derives** the update decision from observations
alone — ``interpretation`` is recorded in the trace but does not enter the
graph. This is the structural anti-bias core of the design (acceptance #4).

Module-level constant ``APPLY_INTERPRETATION`` is the toggle Phase 2 will
flip; Phase 1 hard-codes it to ``False``.

This module is **not** an atom — no ``MANIFEST``, no ``install``. The FSM
policy atom (and the brief builder atom's dispatch path) imports it as a
pure helper.
"""

from __future__ import annotations

from typing import Any

from agentm_rca_hfsm.schema import Interpretation, Observation, WorkerReturn
from agentm_rca_hfsm.updates import UpdateProposal


# Phase 1: never apply the worker's interpretation automatically. Phase 2
# may toggle this once the orchestrator's re-derivation is mature enough
# to A/B against an interpretation-trusting variant.
APPLY_INTERPRETATION: bool = False


def parse_worker_return(structured_output: dict[str, Any]) -> WorkerReturn:
    """Parse a worker's structured output (design §5.4 ``expected_output``).

    Accepts a dict shaped as::

        {
          "observations": [
              {"id":"O1","text":"...","source_tool_call":"...",
               "tool_signature":"...","related_symptoms":[],"related_predictions":[]},
              ...
          ],
          "interpretation": {
              "proposed_update": "...",
              "reasoning": "...",
              "confidence": "...",
          },
        }

    Raises ``ValueError`` on malformed input. ``observations`` may be empty
    (an empty steelman is a meaningful refutation signal); ``interpretation``
    fields default to empty strings if absent so a worker that wants to
    decline a verdict still parses.
    """

    if not isinstance(structured_output, dict):
        raise ValueError("worker return must be a dict")
    raw_obs = structured_output.get("observations")
    if raw_obs is None:
        raw_obs = []
    if not isinstance(raw_obs, list):
        raise ValueError("worker return: 'observations' must be a list")
    observations: list[Observation] = []
    for idx, item in enumerate(raw_obs):
        if not isinstance(item, dict):
            raise ValueError(f"worker return: observations[{idx}] is not a dict")
        observations.append(
            Observation(
                id=str(item.get("id", "")),
                text=str(item.get("text", "")),
                source_tool_call=str(item.get("source_tool_call", "")),
                tool_signature=str(item.get("tool_signature", "")),
                related_symptoms=list(item.get("related_symptoms", []) or []),
                related_predictions=list(item.get("related_predictions", []) or []),
                ts=float(item.get("ts", 0.0) or 0.0),
            )
        )
    interp_raw = structured_output.get("interpretation") or {}
    if not isinstance(interp_raw, dict):
        raise ValueError("worker return: 'interpretation' must be a dict")
    interpretation = Interpretation(
        proposed_update=str(interp_raw.get("proposed_update", "")),
        reasoning=str(interp_raw.get("reasoning", "")),
        confidence=str(interp_raw.get("confidence", "")),
    )
    return WorkerReturn(observations=observations, interpretation=interpretation)


def ingest_observations(gate: Any, wr: WorkerReturn) -> list[str]:
    """Append the worker's observations to L1 via the gate.

    Dispatches one ``record_observation`` ``UpdateProposal`` per observation
    through ``gate.apply``. Returns the list of ``applied_id`` values from
    the gate (the per-observation ids the store assigned). On any rejection
    the corresponding slot is omitted from the return list — the gate's
    rejection event already carries the failure reason on the bus.

    The split between "always ingest observations" and "never auto-apply
    interpretation" (see :func:`should_apply_interpretation`) is the
    structural enforcement of design §6: a worker cannot replace evidence
    with rhetoric.
    """

    ids: list[str] = []
    for obs in wr.observations:
        result = gate.apply(UpdateProposal(op="record_observation", observation=obs))
        if result.kind == "applied" and result.applied_id is not None:
            ids.append(result.applied_id)
    return ids


def should_apply_interpretation(wr: WorkerReturn) -> bool:
    """Phase 1 always returns ``False`` (orchestrator re-derives).

    The argument is accepted (not ``del``-ed) so Phase 2 can switch on the
    interpretation's free-form ``confidence`` / ``proposed_update`` without
    changing the call sites. The module-level :data:`APPLY_INTERPRETATION`
    flag is the single source of truth for the toggle.
    """

    del wr  # Phase 1: hard False; argument reserved for Phase 2.
    return APPLY_INTERPRETATION


__all__ = [
    "APPLY_INTERPRETATION",
    "ingest_observations",
    "parse_worker_return",
    "should_apply_interpretation",
]
