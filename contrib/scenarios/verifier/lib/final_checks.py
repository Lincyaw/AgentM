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


def _compact_text(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _all_services(
    graph: Mapping[str, Sequence[Sequence[str]]],
    data_profile: Mapping[str, Any],
) -> set[str]:
    services = {
        str(service)
        for service in data_profile.get("structure", {}).get("services", [])
        if isinstance(service, str)
    }
    services.update(graph)
    for neighbors in graph.values():
        services.update(str(info[0]) for info in neighbors if info)
    return services


def _service_forms(service: str) -> set[str]:
    compact = _compact_text(service)
    forms = {compact}
    if compact.startswith("ts"):
        forms.add(compact[2:])
    for form in list(forms):
        if form.endswith("service") and len(form) > len("service"):
            forms.add(form[: -len("service")])
    return {form for form in forms if len(form) >= 4}


def _component_service_matches(
    component: object,
    *,
    services: set[str],
    frontend_set: set[str],
) -> list[str]:
    """Infer service-affinity tokens embedded in a frontend span component."""
    text = _compact_text(component)
    if not text:
        return []
    scored: list[tuple[int, str]] = []
    for service in sorted(services - frontend_set):
        forms = _service_forms(service)
        score = max((len(form) for form in forms if form in text), default=0)
        if score:
            scored.append((score, service))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return [service for score, service in scored if score == best]


def _causal_upstream_services(
    *,
    component: str,
    edges: object,
    component_to_service: Mapping[str, object],
    frontend_set: set[str],
) -> list[str]:
    """Find non-entry services on causal-graph paths that feed a frontend alarm."""
    if not isinstance(edges, list):
        return []

    reverse: dict[str, list[str]] = {}
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if isinstance(source, str) and isinstance(target, str):
            reverse.setdefault(target, []).append(source)

    services: set[str] = set()
    stack = list(reverse.get(component, []))
    seen = set(stack)
    while stack:
        cur = stack.pop()
        service = component_to_service.get(cur)
        if isinstance(service, str) and service not in frontend_set:
            services.add(service)
        for parent in reverse.get(cur, []):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return sorted(services)


def _matched_services_for_anomaly(
    anomaly: Mapping[str, Any],
    *,
    services: set[str],
    frontend_set: set[str],
) -> list[str]:
    upstream = anomaly.get("causal_upstream_services")
    if isinstance(upstream, Sequence) and not isinstance(upstream, str):
        matched: list[str] = []
        for service in upstream:
            if (
                isinstance(service, str)
                and service in services
                and service not in frontend_set
                and service not in matched
            ):
                matched.append(service)
        return matched
    return _component_service_matches(
        anomaly.get("component"),
        services=services,
        frontend_set=frontend_set,
    )


def _compact_anomaly(anomaly: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        "id": anomaly.get("id"),
        "service": anomaly.get("service")
        or _service_from_subject(anomaly.get("subject")),
        "signal": anomaly.get("signal"),
        "component": anomaly.get("component"),
        "causal_upstream_services": anomaly.get("causal_upstream_services"),
        "summary": anomaly.get("summary"),
    }
    return {key: value for key, value in out.items() if value is not None}


def _frontend_anomaly_explained(
    *,
    frontend_service: str,
    matched_services: Sequence[str],
    confirmed_seed_ids: set[str],
    adj: Mapping[str, Sequence[str]],
    source_adjs: Mapping[str, Mapping[str, Sequence[str]]],
) -> bool:
    for seed in confirmed_seed_ids:
        seed_adj = source_adjs.get(seed, adj)
        if not reaches(seed_adj, seed, frontend_service):
            continue
        if not matched_services:
            return True
        for matched in matched_services:
            if reaches(seed_adj, seed, matched) and reaches(
                seed_adj,
                matched,
                frontend_service,
            ):
                return True
    return False


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
    edges = payload.get("edges", [])

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
            item = {
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
                "causal_upstream_services": _causal_upstream_services(
                    component=component,
                    edges=edges,
                    component_to_service=component_to_service,
                    frontend_set=frontend_set,
                ),
            }
            out.append(item)
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
    source_adj_by_seed: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    resolved_frontend_anomalies: Mapping[str, Mapping[str, Any]] | None = None,
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
    services = _all_services(graph, data_profile)
    compact_frontend_anomalies = [_compact_anomaly(anomaly) for anomaly in anomalies[:12]]

    issues: list[FinalCheckIssue] = []
    resolved_map = resolved_frontend_anomalies or {}
    resolved: list[dict[str, Any]] = []
    seed_reachability: dict[str, list[list[str]]] = {}
    slo_targets = set(frontends)
    source_adjs = source_adj_by_seed or {}
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
        seed_adj = source_adjs.get(seed, adj)
        paths = paths_to_targets(seed_adj, seed, slo_targets)
        seed_reachability[seed] = paths
        if not paths:
            has_outgoing = bool(seed_adj.get(seed))
            if not has_outgoing:
                # Confirmed but no outgoing edges: the seed's effect stopped
                # at the injection target (e.g. MemoryStress with no latency
                # propagation). This is a legitimate local-only seed, not a
                # gap the verifier should try to close.
                continue
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
                        "frontend_anomalies": compact_frontend_anomalies,
                    },
                }
            )

    unexplained: list[dict[str, Any]] = []
    for anomaly in anomalies:
        service = str(anomaly.get("service") or _service_from_subject(anomaly.get("subject")) or "")
        anomaly_id = str(anomaly.get("id") or service)
        resolution = resolved_map.get(anomaly_id)
        if resolution:
            item = dict(anomaly)
            item["resolution"] = dict(resolution)
            resolved.append(item)
            continue
        matched_services = _matched_services_for_anomaly(
            anomaly,
            services=services,
            frontend_set=frontend_set,
        )
        explained = service in nodes and _frontend_anomaly_explained(
            frontend_service=service,
            matched_services=matched_services,
            confirmed_seed_ids=confirmed_seed_ids,
            adj=adj,
            source_adjs=source_adjs,
        )
        if explained:
            continue
        item = dict(anomaly)
        if matched_services:
            item["matched_services"] = matched_services
            item["reason"] = (
                "frontend anomaly endpoint is not represented by a path through "
                "its path-aligned upstream service"
            )
        else:
            item["reason"] = (
                "frontend anomaly service is not represented by a node reachable "
                "from any confirmed seed"
            )
        unexplained.append(item)
        issues.append(
            {
                "check": "frontend_anomaly_explained",
                "item": anomaly_id,
                "reason": item["reason"],
                "details": {
                    "service": service,
                    "subject": anomaly.get("subject"),
                    "component": anomaly.get("component"),
                    "matched_services": matched_services,
                },
            }
        )

    return {
        "passed": not issues,
        "entry_services": entries,
        "frontend_services": frontends,
        "seed_reachability": seed_reachability,
        "frontend_anomalies": anomalies,
        "resolved_frontend_anomalies": resolved,
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
