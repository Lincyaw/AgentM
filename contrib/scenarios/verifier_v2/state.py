"""Mutable graph state — no LLM, no I/O.

All graph mutations go through three methods: accept, mark_rejected,
mark_exhausted. Pure and unit-testable.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .schema import (
    TaskAttempt,
    Verdict,
    VerificationTask,
)


@dataclass
class Case:
    """Immutable per-case inputs."""

    injections: list[dict[str, str]]
    graph: dict[str, list[list[str]]]
    infra_set: set[str]
    entry_services: set[str]
    seeds: set[str]
    data_dir: str
    fault_docs: dict[str, str]
    data_profile: dict[str, Any]
    anomaly_inventory: list[dict[str, Any]]
    window: dict[str, str]
    max_parallel: int = 8
    max_retries: int = 3

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> Case:
        injections = args["injections"]
        graph = args["graph"]
        infra_set = set(args.get("infra_nodes", []))
        seeds = {inj.get("node_id") or inj["target"] for inj in injections}
        entry_services = set(args.get("entry_services", []))
        if not entry_services:
            entry_services = _infer_entry_services(graph)
        return cls(
            injections=injections,
            graph=graph,
            infra_set=infra_set,
            entry_services=entry_services,
            seeds=seeds,
            data_dir=args["data_dir"],
            fault_docs=args.get("fault_docs", {}),
            data_profile=args.get("data_profile", {}),
            anomaly_inventory=args.get("anomaly_inventory", []),
            window=args["window"],
            max_parallel=args.get("max_parallel", 8),
            max_retries=args.get("max_retries", 3),
        )


def _link_services(node_id: str) -> list[str]:
    """Extract endpoint services from a link node id (link:A->B → [A, B])."""
    if node_id.startswith("link:") and "->" in node_id:
        body = node_id.removeprefix("link:")
        parts = body.split("->", 1)
        return [p.strip() for p in parts if p.strip()]
    return []


def _infer_entry_services(graph: dict[str, list[list[str]]]) -> set[str]:
    callers: set[str] = set()
    callees: set[str] = set()
    for svc, neighbors in graph.items():
        for info in neighbors:
            if len(info) >= 2 and info[1] == "caller_to_callee":
                callers.add(svc)
                callees.add(info[0])
    _ENTRY_KEYWORDS = {"frontend", "ui", "dashboard", "gateway", "proxy", "loadgenerator"}
    explicit = {
        s for s in callers | callees
        if any(kw in s.lower() for kw in _ENTRY_KEYWORDS)
    }
    return explicit or (callers - callees)


@dataclass
class GraphState:
    """Mutable FPG graph under construction."""

    case: Case
    log: Callable[[str], None]

    # Confirmed nodes: id → node metadata
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Adjacency: src → [dst, ...]
    adj: dict[str, list[str]] = field(default_factory=dict)
    # Edges: list of edge dicts
    edges: list[dict[str, Any]] = field(default_factory=list)
    # Source seed attribution per node
    node_sources: dict[str, set[str]] = field(default_factory=dict)
    # Per-seed adjacency
    seed_adj: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    # Confirmed seed ids
    confirmed_seeds: set[str] = field(default_factory=set)

    # Verdict store
    verdicts: dict[str, Verdict] = field(default_factory=dict)

    # Exhausted/rejected tracking
    exhausted_edges: set[str] = field(default_factory=set)
    rejected_edges: set[str] = field(default_factory=set)

    # Attempt history per task
    attempts: dict[str, list[TaskAttempt]] = field(default_factory=dict)

    # Counters
    agent_calls: int = 0
    rounds: int = 0

    def accept(self, task: VerificationTask, verdict: Verdict) -> bool:
        """Accept a confirmed verdict: add node/edge to graph. Returns True if graph changed."""
        if verdict.kind != "confirmed":
            return False

        self.verdicts[task.edge_key] = verdict
        changed = False

        if task.kind == "seed":
            seed_id = task.from_entity
            if seed_id not in self.nodes:
                self.nodes[seed_id] = {
                    "id": seed_id,
                    "predicate": verdict.predicate,
                    "kind": "seed",
                    "affected_endpoints": verdict.affected_endpoints,
                }
                self.confirmed_seeds.add(seed_id)
                changed = True

            # For link seeds (link:A->B), also register both endpoint services
            # as nodes so frontier expansion and reachability work.
            link_services = _link_services(seed_id)
            for svc in link_services:
                if svc not in self.nodes:
                    self.nodes[svc] = {
                        "id": svc,
                        "predicate": verdict.predicate,
                        "kind": "link_endpoint",
                        "affected_endpoints": verdict.affected_endpoints,
                    }
                    changed = True
                self.node_sources.setdefault(svc, set()).add(task.source_seed)

            # Add edge between link endpoints (A→B)
            if len(link_services) == 2:
                src, dst = link_services
                edge_id = f"{src}->{dst}"
                if not any(e.get("id") == edge_id for e in self.edges):
                    self.edges.append({
                        "id": edge_id,
                        "src": src,
                        "dst": dst,
                        "relationship_type": "network_link",
                        "source_seed": task.source_seed,
                    })
                    self.adj.setdefault(src, []).append(dst)
                    self.seed_adj.setdefault(task.source_seed, {}).setdefault(
                        src, []
                    ).append(dst)
                    changed = True

            # For non-link seeds, just register the node source
            if not link_services:
                self.node_sources.setdefault(seed_id, set()).add(task.source_seed)

        elif task.kind == "hop":
            to_svc = task.to_entity
            if to_svc not in self.nodes:
                self.nodes[to_svc] = {
                    "id": to_svc,
                    "predicate": verdict.predicate,
                    "kind": "hop",
                    "affected_endpoints": verdict.affected_endpoints,
                }
                changed = True
            self.node_sources.setdefault(to_svc, set()).add(task.source_seed)

            edge_id = f"{task.from_entity}->{to_svc}"
            if not any(e.get("id") == edge_id for e in self.edges):
                self.edges.append({
                    "id": edge_id,
                    "src": task.from_entity,
                    "dst": to_svc,
                    "relationship_type": verdict.relationship_type,
                    "source_seed": task.source_seed,
                })
                self.adj.setdefault(task.from_entity, []).append(to_svc)
                self.seed_adj.setdefault(task.source_seed, {}).setdefault(
                    task.from_entity, []
                ).append(to_svc)
                changed = True

        return changed

    def mark_rejected(self, task: VerificationTask) -> None:
        self.rejected_edges.add(task.edge_key)

    def mark_exhausted(self, task: VerificationTask) -> None:
        self.exhausted_edges.add(task.edge_key)

    def record_attempt(self, task: VerificationTask, attempt: TaskAttempt) -> None:
        self.attempts.setdefault(task.edge_key, []).append(attempt)

    def attempt_history(self, task: VerificationTask) -> list[TaskAttempt]:
        return self.attempts.get(task.edge_key, [])

    def reaches(self, src: str, dst: str, seed: str | None = None) -> bool:
        """Check graph reachability from src to dst."""
        adj = self.seed_adj.get(seed, self.adj) if seed else self.adj
        stack, seen = [src], {src}
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            for nxt in adj.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    def frontier_nodes(self, seed: str) -> list[str]:
        """Nodes confirmed under this seed that have no outgoing confirmed edges."""
        adj = self.seed_adj.get(seed, {})
        return [
            node for node in self.nodes
            if seed in self.node_sources.get(node, set())
            and not adj.get(node)
            and node not in self.case.infra_set
            and node not in self.case.entry_services
        ]
