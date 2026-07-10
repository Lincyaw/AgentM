"""RCA scenario adapter for the generic rescue-window harness.

Implements ``rescue_window.harness.ScenarioAdapter``: how to terminate/judge an
RCA trajectory and how to read a case's ground truth. The generic harness selects
this by name (``--adapter rca``) and never imports it statically, keeping the
measurement machinery scenario-agnostic.

Both halves reuse the scenario's existing fpg evaluation code rather than
re-deriving anything: scoring goes through ``rca_eval.rescue_window_judge``
(fpg ``compare_model_to_ground_truth``), and the ground-truth root cause is read
from ``fpg.Scenario.graph.root_causes`` — the exact roots the judge scores
against, so the oracle target can never diverge from the metric.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.core.abi import AgentMessage
from agentm_eval.benchmarks.rescue_window.harness import (
    GroundTruth,
    ScoredOutcome,
    TrajectoryRef,
    extract_tool_args,
)

_FINAL_TOOL = "submit_final_report"


@dataclass(frozen=True, slots=True)
class RcaRescueAdapter:
    """Scenario adapter scoring RCA rollouts against ops-lite-clean ground truth."""

    name: str = "rca"
    final_tool: str = _FINAL_TOOL

    async def judge(
        self, messages: list[AgentMessage], ref: TrajectoryRef
    ) -> ScoredOutcome:
        payload = extract_tool_args(messages, self.final_tool)
        if payload is None:
            return ScoredOutcome(
                binary_success=None,
                normalized_score=None,
                detail={},
                error="no submit_final_report payload in fork messages",
            )
        from rca_eval.rescue_window_judge import RcabenchJudge

        outcome = await RcabenchJudge().judge(
            agent_output_json=json.dumps(payload, ensure_ascii=False),
            data_dir=ref.data_dir,
            case_id=ref.case_id,
        )
        return ScoredOutcome(
            binary_success=outcome.correct,
            normalized_score=_continuous_score(outcome.detail),
            detail=outcome.detail,
            error=outcome.error,
        )

    def ground_truth(self, ref: TrajectoryRef) -> GroundTruth:
        return load_ground_truth(ref.data_dir)


def load_ground_truth(data_dir: str) -> GroundTruth:
    """Read the injected root cause(s) via fpg — the same roots the judge uses.

    ``fpg.Scenario.graph.root_causes`` are the nodes with no incoming edges
    (injection points); ``compare_model_to_ground_truth`` scores the model's
    ``root_causes`` against exactly these. Their ``subject`` is an entity ref like
    ``svc:ts-order-service``; we strip the type prefix for the human-facing nudge.
    The ``graph.nodes`` at large are propagated symptoms and are never used.
    """

    from fpg import Scenario

    graph_path = Path(data_dir) / "causal_graph_verified.json"
    if not graph_path.is_file():
        return GroundTruth(targets=(), summary="the documented fault")

    scenario = Scenario.model_validate_json(graph_path.read_text(encoding="utf-8"))
    roots = scenario.graph.root_causes
    targets = _dedupe([_strip_prefix(node.subject) for node in roots])
    fault_kinds = _dedupe([str(inj.fault_type) for inj in scenario.injections])
    return GroundTruth(
        targets=tuple(targets),
        summary=_summarize(targets, fault_kinds),
        fault_kinds=tuple(fault_kinds),
        raw={"root_subjects": [node.subject for node in roots]},
    )


def _continuous_score(detail: dict[str, Any]) -> float | None:
    for key in ("fpg_score", "f1"):
        value = detail.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _strip_prefix(subject: str) -> str:
    """``svc:ts-order-service`` -> ``ts-order-service``."""

    return subject.split(":", 1)[1] if ":" in subject else subject


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _summarize(targets: list[str], fault_kinds: list[str]) -> str:
    if targets and fault_kinds:
        return f"`{targets[0]}` exhibiting `{fault_kinds[0]}`"
    if targets:
        return f"`{targets[0]}`"
    if fault_kinds:
        return f"a `{fault_kinds[0]}` fault"
    return "the documented fault"
