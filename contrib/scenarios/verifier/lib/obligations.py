"""Final-check obligations and rework planning.

The verifier's output contract is not an edge set; it is a set of proof
obligations: each seed must reach a front-end target, and each front-end anomaly
must be explained by a seed-rooted path or separated as unrelated. This module
keeps that obligation identity attached to final-check re-dispatch work.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .candidates import graph_rel_type
from .final_checks import reaches
from .schema import FinalCheckReport, HopRecheckRequest, ReworkRequest, SeedRecheckRequest

ObligationKind = Literal["seed_confirmed", "seed_reachability", "frontend_anomaly"]


@dataclass(frozen=True)
class FinalObligation:
    """A concrete final condition that may require additional verification."""

    id: str
    kind: ObligationKind
    target_service: str
    source_seed: str | None = None
    anomaly_id: str | None = None
    anomaly_component: str | None = None
    anomaly_subject: object | None = None
    context: str = ""


def obligations_from_report(report: FinalCheckReport) -> list[FinalObligation]:
    """Convert deterministic final-check issues into proof obligations."""
    obligations: list[FinalObligation] = []
    for issue in report["issues"]:
        check = issue["check"]
        details = issue.get("details", {})
        if check == "seed_confirmed":
            seed = issue["item"]
            obligations.append(
                FinalObligation(
                    id=_obligation_id("seed_confirmed", seed, "seed"),
                    kind="seed_confirmed",
                    source_seed=seed,
                    target_service=seed,
                    context=(
                        "Final invariant gap: every declared injection seed must "
                        f"be verified before propagation. Re-evaluate seed {seed} "
                        "using all local evidence plus nearby service/link telemetry; "
                        "do not reject solely because the single injected point has "
                        "sparse local observations if surrounding evidence shows the "
                        "injected fault manifested."
                    ),
                )
            )
        elif check == "frontend_anomaly_explained":
            target = str(details.get("service") or "")
            anomaly_id = issue["item"]
            component = details.get("component")
            obligations.append(
                FinalObligation(
                    id=_obligation_id("frontend_anomaly", anomaly_id, target),
                    kind="frontend_anomaly",
                    target_service=target,
                    anomaly_id=anomaly_id,
                    anomaly_component=str(component) if component is not None else None,
                    anomaly_subject=details.get("subject"),
                    context=_frontend_anomaly_context(issue, target),
                )
            )
        elif check == "seed_reaches_entry":
            seed = issue["item"]
            targets = details.get("frontend_services", [])
            if not isinstance(targets, list) or not targets:
                targets = report.get("entry_services", [])
            anomaly_context = _format_frontend_anomalies(
                details.get("frontend_anomalies", []),
            )
            for target in sorted(str(target) for target in targets if target):
                obligations.append(
                    FinalObligation(
                        id=_obligation_id("seed_reachability", seed, target),
                        kind="seed_reachability",
                        source_seed=seed,
                        target_service=target,
                        context=(
                            "Final invariant gap: confirmed seed must reach an "
                            f"SLO/entry endpoint. Verify whether the path from "
                            f"{seed} reaches {target} through this hop. Because "
                            f"{target} is an entry/SLO target, accept "
                            "path-specific endpoint impact as propagation: "
                            "vanished or collapsed endpoint calls, endpoint "
                            "latency/error/timeout changes, or frontend/client "
                            "spans aligned with the upstream fault path. Do not "
                            "require whole-frontend CPU, memory, or log "
                            "degradation. Reject only when the observed frontend "
                            "change is a global/background demand shift or is not "
                            "selective to this seed path. In a multi-fault case, "
                            "other frontend endpoints may be degraded by other "
                            "seeds; do not use those co-fault symptoms to reject "
                            "an endpoint/anomaly that is specifically aligned "
                            "with this seed path."
                            + anomaly_context
                        ),
                    )
                )
    return obligations


def rework_requests_for_obligations(
    *,
    graph: Mapping[str, Sequence[Sequence[str]]],
    nodes: Mapping[str, Mapping[str, Any]],
    adj: Mapping[str, Sequence[str]],
    source_adj_by_seed: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    node_sources: Mapping[str, set[str]],
    obligations: Sequence[FinalObligation],
) -> list[ReworkRequest]:
    """Plan hop rechecks without collapsing distinct final obligations."""
    out: list[ReworkRequest] = []
    seen: set[tuple[str, str, str, str]] = set()
    source_adjs = source_adj_by_seed or {}
    for obligation in obligations:
        if obligation.kind == "seed_confirmed":
            if obligation.source_seed:
                out.append(
                    SeedRecheckRequest(
                        kind="seed_recheck",
                        seed=obligation.source_seed,
                        context=obligation.context,
                    )
                )
            continue

        allow_other = obligation.kind == "frontend_anomaly"
        frontier_only = obligation.kind == "frontend_anomaly"
        for from_svc in sorted(nodes):
            if from_svc == obligation.target_service:
                continue
            if frontier_only and adj.get(from_svc):
                continue
            rel_type = graph_rel_type(graph, from_svc, obligation.target_service)
            if not rel_type and not allow_other:
                continue
            rel_type = rel_type or "other"
            source_seeds = (
                [obligation.source_seed]
                if obligation.source_seed
                else sorted(node_sources.get(from_svc, set()))
            )
            for seed in source_seeds:
                if not seed or not reaches(source_adjs.get(seed, adj), seed, from_svc):
                    continue
                key = (from_svc, obligation.target_service, seed, obligation.id)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    HopRecheckRequest(
                        kind="hop_recheck",
                        from_service=from_svc,
                        to_service=obligation.target_service,
                        rel_type=rel_type,
                        source_seed=seed,
                        context=obligation.context,
                        obligation_id=obligation.id,
                        obligation_kind=obligation.kind,
                        target_frontend=obligation.target_service,
                        anomaly_id=obligation.anomaly_id,
                        anomaly_component=obligation.anomaly_component,
                    )
                )
    return out


def obligation_payload(request: HopRecheckRequest) -> dict[str, Any]:
    """Return the stable obligation metadata passed to hop agents and ledgers."""
    payload = {
        "id": request.obligation_id,
        "kind": request.obligation_kind,
        "target_frontend": request.target_frontend,
        "anomaly_id": request.anomaly_id,
        "anomaly_component": request.anomaly_component,
    }
    return {key: value for key, value in payload.items() if value is not None}


def _frontend_anomaly_context(issue: Mapping[str, Any], target: str) -> str:
    details = issue.get("details", {})
    if not isinstance(details, Mapping):
        details = {}
    component = details.get("component")
    return (
        "Final invariant gap: explain SLO/frontend anomaly "
        f"{issue['item']} on {target}. "
        f"Subject={details.get('subject')}; component={component}. "
        f"Path-aligned services={details.get('matched_services', [])}. "
        "This is an endpoint/anomaly coverage check, not a generic "
        "service-health check. Add this hop only if same-trace or "
        "endpoint-specific evidence connects the confirmed upstream fault path "
        "to this requested SLO symptom. Aggregate frontend/proxy error rate, "
        "latency, or span count alone is insufficient. If only a subset of "
        "frontend/proxy failures is path-aligned with the fault, say exactly "
        "which subset is explained and separate unrelated/background frontend "
        "failures. In multi-fault cases, unrelated frontend changes do not "
        "disprove this requested anomaly if the anomaly itself is connected to "
        "the source path."
    )


def _obligation_id(kind: ObligationKind, item: str, target: str) -> str:
    safe_item = item.replace(" ", "_")
    safe_target = target.replace(" ", "_")
    return f"{kind}:{safe_item}->{safe_target}"


def _format_frontend_anomalies(raw: object) -> str:
    if not isinstance(raw, list) or not raw:
        return ""
    rows: list[str] = []
    for item in raw[:8]:
        if not isinstance(item, Mapping):
            continue
        bits = [
            str(item.get("id") or ""),
            str(item.get("component") or ""),
            str(item.get("summary") or ""),
        ]
        text = " | ".join(bit for bit in bits if bit)
        if text:
            rows.append(text)
    if not rows:
        return ""
    return " Visible frontend anomalies to consider: " + " ; ".join(rows)


__all__ = [
    "FinalObligation",
    "ObligationKind",
    "obligation_payload",
    "obligations_from_report",
    "rework_requests_for_obligations",
]
