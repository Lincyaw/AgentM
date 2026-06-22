"""Prompt builders/config carrier for verifier audit agents."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import ExtensionAPI
from agentm.extensions import ExtensionManifest


class AuditContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    instruction: str
    payload: dict[str, Any] = Field(default_factory=dict)


MANIFEST = ExtensionManifest(
    name="audit_context",
    description="Builds one bounded verifier audit prompt.",
    registers=(),
    config_schema=AuditContextConfig,
)


def _payload_prompt(role: str, instruction: str, payload: Mapping[str, Any]) -> str:
    return (
        f"## Audit role\n{role}\n\n"
        f"## Bounded question\n{instruction}\n\n"
        "## Input payload\n"
        "```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n```\n\n"
        "Return only the structured `submit_result` payload."
    )


def build_audit_prompt(
    *,
    role: str,
    instruction: str,
    payload: Mapping[str, Any],
) -> str:
    return _payload_prompt(role, instruction, payload)


def build_anomaly_prompt(
    *,
    scope: str,
    case: Mapping[str, Any],
    graph: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> str:
    return _payload_prompt(
        "anomaly_coverage",
        (
            "Inspect this entry/service scope. Identify meaningful abnormal "
            "trace, metric, or log symptoms and decide whether the current "
            "candidate graph explains each one."
        ),
        {
            "scope": scope,
            "case": case,
            "candidate_graph": graph,
            "evidence_ledger": ledger,
        },
    )


def build_anomaly_context(
    *,
    scope: str,
    case: Mapping[str, Any],
    graph: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> AuditContextConfig:
    return AuditContextConfig(
        role="anomaly_coverage",
        instruction=(
            "Inspect this entry/service scope. Identify meaningful abnormal "
            "trace, metric, or log symptoms and decide whether the current "
            "candidate graph explains each one."
        ),
        payload={
            "scope": scope,
            "case": case,
            "candidate_graph": graph,
            "evidence_ledger": ledger,
        },
    )


def build_causal_prompt(
    *,
    path_id: str,
    path: Sequence[str],
    case: Mapping[str, Any],
    graph: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> str:
    return _payload_prompt(
        "causal_path",
        (
            "Audit whether this single seed-to-entry path is a coherent "
            "causal explanation. Reject paths that only borrow another "
            "seed's symptoms or are merely topologically reachable."
        ),
        {
            "path_id": path_id,
            "path": list(path),
            "case": case,
            "candidate_graph": graph,
            "evidence_ledger": ledger,
        },
    )


def build_causal_context(
    *,
    path_id: str,
    path: Sequence[str],
    case: Mapping[str, Any],
    graph: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> AuditContextConfig:
    return AuditContextConfig(
        role="causal_path",
        instruction=(
            "Audit whether this single seed-to-entry path is a coherent "
            "causal explanation. Reject paths that only borrow another "
            "seed's symptoms or are merely topologically reachable."
        ),
        payload={
            "path_id": path_id,
            "path": list(path),
            "case": case,
            "candidate_graph": graph,
            "evidence_ledger": ledger,
        },
    )


def build_seed_coverage_prompt(
    *,
    seed: str,
    case: Mapping[str, Any],
    graph: Mapping[str, Any],
    ledger: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    causal_reports: Sequence[Mapping[str, Any]],
    anomaly_reports: Sequence[Mapping[str, Any]],
) -> str:
    return _payload_prompt(
        "seed_coverage",
        (
            "Classify this seed as explains_entry, local_only, "
            "benign_or_no_effect, needs_recheck, or invalid_path."
        ),
        {
            "seed": seed,
            "case": case,
            "candidate_graph": graph,
            "evidence_ledger": ledger,
            "paths_from_seed": [list(path) for path in paths],
            "causal_reports": list(causal_reports),
            "anomaly_reports": list(anomaly_reports),
        },
    )


def build_seed_coverage_context(
    *,
    seed: str,
    case: Mapping[str, Any],
    graph: Mapping[str, Any],
    ledger: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    causal_reports: Sequence[Mapping[str, Any]],
    anomaly_reports: Sequence[Mapping[str, Any]],
) -> AuditContextConfig:
    return AuditContextConfig(
        role="seed_coverage",
        instruction=(
            "Classify this seed as explains_entry, local_only, "
            "benign_or_no_effect, needs_recheck, or invalid_path."
        ),
        payload={
            "seed": seed,
            "case": case,
            "candidate_graph": graph,
            "evidence_ledger": ledger,
            "paths_from_seed": [list(path) for path in paths],
            "causal_reports": list(causal_reports),
            "anomaly_reports": list(anomaly_reports),
        },
    )


def build_reducer_prompt(
    *,
    case: Mapping[str, Any],
    graph: Mapping[str, Any],
    ledger: Mapping[str, Any],
    anomaly_reports: Sequence[Mapping[str, Any]],
    causal_reports: Sequence[Mapping[str, Any]],
    seed_coverage_reports: Sequence[Mapping[str, Any]],
    round_n: int,
) -> str:
    return _payload_prompt(
        "audit_reducer",
        (
            "Merge the audit reports. Accept only when all meaningful "
            "anomalies are explained or resolved and every seed has a "
            "resolved coverage status. Otherwise emit concrete rework "
            "requests and/or path-specific edge drops."
        ),
        {
            "round": round_n,
            "case": case,
            "candidate_graph": graph,
            "evidence_ledger": ledger,
            "anomaly_reports": list(anomaly_reports),
            "causal_reports": list(causal_reports),
            "seed_coverage_reports": list(seed_coverage_reports),
        },
    )


def build_reducer_context(
    *,
    case: Mapping[str, Any],
    graph: Mapping[str, Any],
    ledger: Mapping[str, Any],
    anomaly_reports: Sequence[Mapping[str, Any]],
    causal_reports: Sequence[Mapping[str, Any]],
    seed_coverage_reports: Sequence[Mapping[str, Any]],
    round_n: int,
) -> AuditContextConfig:
    return AuditContextConfig(
        role="audit_reducer",
        instruction=(
            "Merge the audit reports. Accept only when all meaningful "
            "anomalies are explained or resolved and every seed has a "
            "resolved coverage status. Otherwise emit concrete rework "
            "requests and/or path-specific edge drops."
        ),
        payload={
            "round": round_n,
            "case": case,
            "candidate_graph": graph,
            "evidence_ledger": ledger,
            "anomaly_reports": list(anomaly_reports),
            "causal_reports": list(causal_reports),
            "seed_coverage_reports": list(seed_coverage_reports),
        },
    )


def install(api: ExtensionAPI, config: AuditContextConfig) -> None:
    del api, config


__all__: Final = [
    "MANIFEST",
    "install",
    "build_audit_prompt",
    "build_anomaly_prompt",
    "build_causal_prompt",
    "build_seed_coverage_prompt",
    "build_reducer_prompt",
    "build_anomaly_context",
    "build_causal_context",
    "build_seed_coverage_context",
    "build_reducer_context",
]
