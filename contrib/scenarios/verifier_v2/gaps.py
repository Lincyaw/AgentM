"""Deterministic gap evaluation — no LLM.

Checks structural invariants against the current graph state and returns
all unsatisfied gaps. This replaces final_checks.py + obligations.py +
the audit agent gap-detection role.
"""
from __future__ import annotations

from .schema import Gap, GapReport
from .state import Case, GraphState


def evaluate_gaps(case: Case, state: GraphState) -> GapReport:
    """Evaluate all structural invariants. Returns gaps ordered by priority."""
    gaps: list[Gap] = []

    gaps.extend(_unconfirmed_seed_gaps(case, state))
    gaps.extend(_unreachable_seed_gaps(case, state))
    gaps.extend(_unexplained_anomaly_gaps(case, state))

    return GapReport(gaps=gaps)


def _unconfirmed_seed_gaps(case: Case, state: GraphState) -> list[Gap]:
    """Every declared injection must be confirmed as observable.

    Exception: if a seed's target is already in the graph as a hop node
    (reached by another seed's propagation), its effect is masked — it's
    not a valid independent root cause for RCA purposes and should not
    block convergence.
    """
    gaps: list[Gap] = []
    for seed in sorted(case.seeds):
        if seed in state.confirmed_seeds:
            continue
        # Check if this seed's target is already explained by another seed's
        # propagation — if so, it's masked, not a gap.
        seed_services = _seed_target_services(seed)
        masked = any(
            svc in state.nodes
            and state.nodes[svc].get("kind") == "hop"
            for svc in seed_services
        )
        if masked:
            continue
        gaps.append(Gap(
            kind="unconfirmed_seed",
            id=f"unconfirmed_seed:{seed}",
            source_seed=seed,
            target=seed,
            context=(
                f"Injection seed {seed} has not been verified. "
                "Search for telemetry evidence (traces, metrics, logs) "
                "showing the injected fault took observable effect."
            ),
        ))
    return gaps


def _seed_target_services(seed: str) -> list[str]:
    """Extract the service(s) a seed targets."""
    if seed.startswith("link:") and "->" in seed:
        body = seed.removeprefix("link:")
        return [p.strip() for p in body.split("->", 1)]
    return [seed]


def _unreachable_seed_gaps(case: Case, state: GraphState) -> list[Gap]:
    """Every confirmed seed must have a path to an entry/frontend service."""
    gaps: list[Gap] = []
    for seed in sorted(state.confirmed_seeds):
        reaches_entry = any(
            state.reaches(seed, entry, seed=seed)
            for entry in case.entry_services
        )
        if not reaches_entry:
            # Seeds with no outgoing edges in the topology are local-only
            has_outgoing = bool(case.graph.get(seed))
            if not has_outgoing:
                continue
            gaps.append(Gap(
                kind="unreachable_seed",
                id=f"unreachable_seed:{seed}",
                source_seed=seed,
                target=seed,
                context=(
                    f"Confirmed seed {seed} has no verified path to any "
                    f"entry service ({', '.join(sorted(case.entry_services))}). "
                    "Expand the propagation frontier toward entry services."
                ),
            ))
    return gaps


def _unexplained_anomaly_gaps(case: Case, state: GraphState) -> list[Gap]:
    """Every frontend anomaly must be explained by a confirmed path."""
    gaps: list[Gap] = []
    frontend_anomalies = _frontend_anomalies(case)

    for anomaly in frontend_anomalies:
        service = anomaly.get("service", "")
        anomaly_id = anomaly.get("id", f"anomaly:{service}")
        component = anomaly.get("component", "")

        explained = _anomaly_explained(state, case, service, component)
        if not explained:
            gaps.append(Gap(
                kind="unexplained_anomaly",
                id=f"unexplained_anomaly:{anomaly_id}",
                source_seed=None,
                target=service,
                anomaly_id=anomaly_id,
                context=(
                    f"Frontend anomaly on {service} "
                    f"(component: {component or 'service-wide'}) "
                    "is not explained by any confirmed propagation path."
                ),
            ))

    return gaps


def _frontend_anomalies(case: Case) -> list[dict[str, str]]:
    """Extract frontend/entry anomalies from the anomaly inventory."""
    out: list[dict[str, str]] = []
    for record in case.anomaly_inventory:
        if record.get("status") != "changed":
            continue
        subject = str(record.get("subject", ""))
        if not subject.startswith("svc:"):
            continue
        service = subject.removeprefix("svc:")
        if service not in case.entry_services:
            continue
        out.append({
            "id": str(record.get("id", f"anomaly:{service}:{len(out)}")),
            "service": service,
            "component": str(record.get("component", "")),
        })
    return out


def _anomaly_explained(
    state: GraphState,
    case: Case,
    service: str,
    component: str,
) -> bool:
    """Check if any confirmed seed path reaches the anomaly's service/endpoint."""
    # If the service is already a confirmed node in the graph, it's explained
    if service in state.nodes:
        return True

    for seed in state.confirmed_seeds:
        # For link seeds, also check reachability from endpoint services
        start_nodes = [seed]
        if seed.startswith("link:") and "->" in seed:
            body = seed.removeprefix("link:")
            parts = body.split("->", 1)
            start_nodes.extend(p.strip() for p in parts if p.strip())

        for start in start_nodes:
            if state.reaches(start, service, seed=seed):
                if not component:
                    return True
                for node_id, node_meta in state.nodes.items():
                    endpoints = node_meta.get("affected_endpoints", [])
                    if not endpoints:
                        return True
                    if any(component in ep for ep in endpoints):
                        return True
                return True
    return False
