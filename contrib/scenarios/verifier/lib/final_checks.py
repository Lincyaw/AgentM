"""Deterministic final invariants for verifier outputs."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schema import FinalCheckIssue, FinalCheckReport

_KNOWN_ENTRY_SERVICES = {"frontend", "frontend-proxy", "ts-ui-dashboard"}


def frontend_like(service: str, entry_services: set[str]) -> bool:
    """Return whether a service should be treated as a user-facing entry tier."""
    name = service.lower()
    return (
        service in entry_services
        or service in _KNOWN_ENTRY_SERVICES
        or "frontend" in name
        or name.endswith("-ui")
        or name.endswith("-dashboard")
    )


def frontend_services(
    *,
    graph: Mapping[str, Sequence[Sequence[str]]],
    data_profile: Mapping[str, Any],
    entry_services: set[str],
) -> list[str]:
    services = set(entry_services)
    services.update(
        str(service)
        for service in data_profile.get("structure", {}).get("services", [])
        if isinstance(service, str)
    )
    services.update(graph)
    for neighbors in graph.values():
        services.update(str(info[0]) for info in neighbors if info)
    return sorted(service for service in services if frontend_like(service, entry_services))


def _service_from_subject(subject: object) -> str | None:
    text = str(subject or "")
    if text.startswith("svc:"):
        return text.removeprefix("svc:")
    return None


def _causal_graph_frontend_anomalies(
    data_dir: str,
    frontend_set: set[str],
) -> list[dict[str, Any]]:
    path = Path(data_dir) / "causal_graph.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    component_to_service = payload.get("component_to_service", {})
    if not isinstance(component_to_service, Mapping):
        component_to_service = {}

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for section in ("path_terminal_alarm_nodes", "alarm_nodes"):
        rows = payload.get(section, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            component = str(row.get("component", ""))
            service = component_to_service.get(component)
            if not isinstance(service, str) or service not in frontend_set:
                continue
            anomaly_id = f"causal_graph:{component}"
            if anomaly_id in seen:
                continue
            seen.add(anomaly_id)
            out.append(
                {
                    "id": anomaly_id,
                    "subject": f"svc:{service}",
                    "service": service,
                    "modality": "trace",
                    "signal": "frontend_alarm",
                    "status": "changed",
                    "component": component,
                    "state": row.get("state", []),
                    "source": section,
                    "summary": f"{component} is an entry-tier alarm in causal_graph",
                }
            )
    return out


def frontend_anomalies(
    *,
    data_dir: str,
    anomaly_inventory: Sequence[Mapping[str, Any]],
    frontend_set: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in anomaly_inventory:
        if record.get("status") != "changed":
            continue
        service = _service_from_subject(record.get("subject"))
        if not service or service not in frontend_set:
            continue
        item = dict(record)
        item["service"] = service
        anomaly_id = str(item.get("id") or f"anomaly:{service}:{len(out)}")
        if anomaly_id in seen:
            continue
        seen.add(anomaly_id)
        out.append(item)

    for item in _causal_graph_frontend_anomalies(data_dir, frontend_set):
        anomaly_id = str(item["id"])
        if anomaly_id not in seen:
            seen.add(anomaly_id)
            out.append(item)
    return out


def reaches(adj: Mapping[str, Sequence[str]], src: str, dst: str) -> bool:
    if src == dst:
        return True
    stack = [src]
    seen = {src}
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, []):
            if nxt == dst:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def paths_to_targets(
    adj: Mapping[str, Sequence[str]],
    src: str,
    targets: set[str],
    *,
    max_depth: int = 12,
) -> list[list[str]]:
    if src not in adj and src not in targets:
        return [[src]] if src in targets else []
    out: list[list[str]] = []
    stack: list[tuple[str, list[str]]] = [(src, [src])]
    while stack:
        cur, path = stack.pop()
        if cur in targets and len(path) > 1:
            out.append(path)
            continue
        if len(path) >= max_depth:
            continue
        for nxt in adj.get(cur, []):
            if nxt not in path:
                stack.append((nxt, path + [nxt]))
    return out


def run_final_checks(
    *,
    data_dir: str,
    graph: Mapping[str, Sequence[Sequence[str]]],
    data_profile: Mapping[str, Any],
    anomaly_inventory: Sequence[Mapping[str, Any]],
    entry_services: set[str],
    seeds: set[str],
    confirmed_seed_ids: set[str],
    nodes: Mapping[str, Mapping[str, Any]],
    adj: Mapping[str, Sequence[str]],
) -> FinalCheckReport:
    """Check output-level invariants that must hold even when audit is skipped."""
    entries = sorted(entry_services)
    frontends = frontend_services(
        graph=graph,
        data_profile=data_profile,
        entry_services=entry_services,
    )
    frontend_set = set(frontends)
    anomalies = frontend_anomalies(
        data_dir=data_dir,
        anomaly_inventory=anomaly_inventory,
        frontend_set=frontend_set,
    )

    issues: list[FinalCheckIssue] = []
    seed_reachability: dict[str, list[list[str]]] = {}
    slo_targets = set(frontends)
    for seed in sorted(seeds):
        if seed not in confirmed_seed_ids:
            issues.append(
                {
                    "check": "seed_confirmed",
                    "item": seed,
                    "reason": "seed was not confirmed, so it cannot explain entry impact",
                }
            )
            seed_reachability[seed] = []
            continue
        paths = paths_to_targets(adj, seed, slo_targets)
        seed_reachability[seed] = paths
        if not paths:
            issues.append(
                {
                    "check": "seed_reaches_entry",
                    "item": seed,
                    "reason": (
                        "confirmed seed has no accepted FPG path to an SLO/entry "
                        "service"
                    ),
                    "details": {
                        "entry_services": entries,
                        "frontend_services": frontends,
                    },
                }
            )

    unexplained: list[dict[str, Any]] = []
    for anomaly in anomalies:
        service = str(anomaly.get("service") or _service_from_subject(anomaly.get("subject")) or "")
        explained = service in nodes and any(
            reaches(adj, seed, service)
            for seed in confirmed_seed_ids
        )
        if explained:
            continue
        item = dict(anomaly)
        item["reason"] = (
            "frontend anomaly service is not represented by a node reachable "
            "from any confirmed seed"
        )
        unexplained.append(item)
        issues.append(
            {
                "check": "frontend_anomaly_explained",
                "item": str(anomaly.get("id") or service),
                "reason": item["reason"],
                "details": {
                    "service": service,
                    "subject": anomaly.get("subject"),
                    "component": anomaly.get("component"),
                },
            }
        )

    return {
        "passed": not issues,
        "entry_services": entries,
        "frontend_services": frontends,
        "seed_reachability": seed_reachability,
        "frontend_anomalies": anomalies,
        "unexplained_frontend_anomalies": unexplained,
        "issues": issues,
    }


__all__ = [
    "frontend_anomalies",
    "frontend_like",
    "frontend_services",
    "paths_to_targets",
    "reaches",
    "run_final_checks",
]
