"""Judge a leaf submission against ground truth.

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
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_logger = logging.getLogger(__name__)

__all__ = ["JudgeOutcome", "LeafJudge", "RcabenchJudge"]


@dataclass(frozen=True)
class JudgeOutcome:
    """Result of judging one submission against ground truth.

    ``correct`` is the headline success bit (baseline-comparable
    ``exact_match``). ``detail`` carries the secondary metrics
    (precision / recall / f1 / service_exact_match / any_*_hit) for
    downstream analysis without re-judging.
    """

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
    """Default judge: wraps rcabench-platform's ``evaluation.evaluate``.

    ``correct = exact_match`` to match the baseline's stored correctness.
    Reads ``injection.json`` from ``data_dir`` for ground truth and uses
    ``data_dir`` as the parquet dir for the (correctness-irrelevant) SQL
    executability check.
    """

    async def judge(
        self, *, agent_output_json: str | None, data_dir: str, case_id: str
    ) -> JudgeOutcome:
        # ``evaluate_v2`` lives in the published rcabench-platform wheel's
        # ``evaluation.v2`` package -- the same module ``agent.py`` imports
        # ``AgentRCAOutput`` from, so agent + judge stay on one rcabench. (The
        # source repo has since flattened ``v2`` into ``evaluation`` and
        # renamed it ``evaluate``; that tree drops ``v2`` and breaks the agent,
        # so do not install it editable.)
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
            _logger.exception("RcabenchJudge: evaluate_v2() raised for case %s", case_id)
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
