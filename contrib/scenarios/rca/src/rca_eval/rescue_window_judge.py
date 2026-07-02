"""Judge RCA branch submissions against ground truth.

The success metric must match the baseline numbers it is compared against.
rcabench-platform's ``RCABenchProcesser`` writes ``correct =
EvaluationResult.exact_match`` (strict multiset match of every GT
``(service, fault_kind)`` pair, no over/under-claim), so the default judge
here wraps the same ``evaluation.evaluate`` and reports ``exact_match`` as
``correct``. Ground truth is read from the case's ``injection.json`` in the
dataset directory, never from the storage the backbone came from.

``exact_match`` is computed by the deterministic set-match path and needs no
LLM client; the per-evidence LLM judge (``evidence_support_rate``) is the
only thing that would, and it does not affect correctness. So the judge
runs with ``llm_client=None`` -- fully offline and deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

__all__ = ["JudgeOutcome", "LeafJudge", "RcabenchJudge"]


@dataclass(frozen=True, slots=True)
class JudgeOutcome:
    """Result of judging one submission against ground truth."""

    correct: bool
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class LeafJudge(Protocol):
    """Scores an agent submission against the ground truth at ``data_dir``."""

    async def judge(
        self, *, agent_output_json: str | None, data_dir: str, case_id: str
    ) -> JudgeOutcome:
        ...


class RcabenchJudge:
    """Default judge: wraps rcabench-platform's ``evaluate_v2``."""

    async def judge(
        self, *, agent_output_json: str | None, data_dir: str, case_id: str
    ) -> JudgeOutcome:
        fpg = _judge_fpg_output(agent_output_json, data_dir)
        if fpg is not None:
            return fpg

        from rcabench_platform.v3.sdk.evaluation.v2 import evaluate_v2

        injection_path = Path(data_dir) / "injection.json"
        try:
            injection = json.loads(injection_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return JudgeOutcome(correct=False, error=f"injection.json unreadable: {exc}")

        try:
            result = await evaluate_v2(
                agent_output_raw=agent_output_json or "",
                injection=injection,
                parquet_dir=data_dir,
                gt_graph=None,
                llm_client=None,
                case_name=case_id,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad case must not sink the batch
            logger.exception(f"RcabenchJudge: evaluate_v2() raised for case {case_id}")
            return JudgeOutcome(correct=False, error=f"evaluate_v2 raised: {exc!s:.200}")

        detail = {
            "exact_match": result.exact_match,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "any_root_cause_hit": result.any_root_cause_hit,
            "any_service_hit": result.any_service_hit,
            "all_service_hit": result.all_service_hit,
            "fault_kind_accuracy": result.fault_kind_accuracy,
            "parse_error": result.parse_error,
        }
        return JudgeOutcome(correct=bool(result.exact_match), detail=detail)


def _judge_fpg_output(agent_output_json: str | None, data_dir: str) -> JudgeOutcome | None:
    payload = _parse_json_object(agent_output_json)
    if payload is None or not _looks_like_fpg_payload(payload):
        return None
    graph_path = Path(data_dir) / "causal_graph_verified.json"
    if not graph_path.is_file():
        return None

    if _is_empty_fpg_payload(payload):
        empty_correct = _verified_graph_is_empty(graph_path)
        empty_detail: dict[str, Any] = {
            "exact_match": empty_correct,
            "precision": 1.0 if empty_correct else 0.0,
            "recall": 1.0 if empty_correct else 0.0,
            "f1": 1.0 if empty_correct else 0.0,
            "any_root_cause_hit": False,
            "any_service_hit": False,
            "all_service_hit": empty_correct,
            "fault_kind_accuracy": None,
            "parse_error": None if empty_correct else "empty_fpg_output",
            "fpg_score": 1.0 if empty_correct else 0.0,
            "fpg_root_subject_f1": 1.0 if empty_correct else 0.0,
            "fpg_subject_path_reachability_hit": empty_correct,
        }
        return JudgeOutcome(correct=empty_correct, detail=empty_detail)

    try:
        from fpg import ModelRCAOutput, Scenario, compare_model_to_ground_truth

        model_output = ModelRCAOutput.model_validate(payload)
        scenario = Scenario.model_validate_json(graph_path.read_text(encoding="utf-8"))
        comparison = compare_model_to_ground_truth(model_output, scenario).model_dump(
            mode="json"
        )
    except Exception as exc:  # noqa: BLE001 -- scoring should not sink a batch
        logger.exception("FPG judge failed for rescue-window case")
        error_detail: dict[str, Any] = {
            "exact_match": False,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "any_root_cause_hit": False,
            "any_service_hit": False,
            "all_service_hit": False,
            "fault_kind_accuracy": None,
            "parse_error": f"fpg validation/comparison error: {exc!s:.200}",
        }
        return JudgeOutcome(correct=False, detail=error_detail)

    root_subject_f1 = _metric_field(comparison, "root_subjects", "f1")
    root_subject_precision = _metric_field(comparison, "root_subjects", "precision")
    root_subject_recall = _metric_field(comparison, "root_subjects", "recall")
    root_subject_matched = _metric_field(comparison, "root_subjects", "matched")
    subject_path_reachability_hit = (
        comparison.get("subject_path_reachability_hit") is True
    )
    fpg_score = float(comparison.get("score") or 0.0)
    correct = fpg_score >= 1.0
    detail: dict[str, Any] = {
        "exact_match": correct,
        "precision": root_subject_precision,
        "recall": root_subject_recall,
        "f1": fpg_score,
        "any_root_cause_hit": root_subject_matched > 0.0,
        "any_service_hit": root_subject_matched > 0.0,
        "all_service_hit": root_subject_recall >= 1.0,
        "fault_kind_accuracy": None,
        "parse_error": None,
        "fpg_score": fpg_score,
        "fpg_root_subject_f1": root_subject_f1,
        "fpg_root_subject_precision": root_subject_precision,
        "fpg_root_subject_recall": root_subject_recall,
        "fpg_subject_f1": _metric_field(comparison, "subjects", "f1"),
        "fpg_soft_subject_edge_f1": _metric_field(
            comparison, "soft_subject_edges", "f1"
        ),
        "fpg_subject_path_match_hit": comparison.get("subject_path_match_hit") is True,
        "fpg_subject_path_reachability_hit": subject_path_reachability_hit,
        "fpg_missing_root_nodes": comparison.get("missing_root_nodes") or [],
        "fpg_extra_root_nodes": comparison.get("extra_root_nodes") or [],
        "fpg_missing_subjects": comparison.get("missing_subjects") or [],
        "fpg_extra_subjects": comparison.get("extra_subjects") or [],
        "fpg_missing_subject_edges": comparison.get("missing_subject_edges") or [],
        "fpg_extra_subject_edges": comparison.get("extra_subject_edges") or [],
    }
    return JudgeOutcome(correct=correct, detail=detail)


def _parse_json_object(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _looks_like_fpg_payload(payload: dict[str, Any]) -> bool:
    nodes = payload.get("nodes")
    root_causes = payload.get("root_causes")
    return isinstance(nodes, list) and isinstance(root_causes, list)


def _is_empty_fpg_payload(payload: dict[str, Any]) -> bool:
    return (
        payload.get("nodes") == []
        and payload.get("edges") == []
        and payload.get("root_causes") == []
    )


def _verified_graph_is_empty(graph_path: Path) -> bool:
    try:
        raw = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    graph = raw.get("graph")
    return (
        isinstance(graph, dict)
        and graph.get("nodes") == []
        and graph.get("edges") == []
    )


def _metric_field(comparison: dict[str, Any], key: str, field: str) -> float:
    metric = comparison.get(key)
    if not isinstance(metric, dict):
        return 0.0
    raw = metric.get(field)
    return float(raw) if isinstance(raw, int | float) else 0.0
