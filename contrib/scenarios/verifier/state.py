"""Pure propagation domain — no ctx, no LLM, no I/O.

``Case`` carries the immutable per-case inputs derived from ``ctx.args``.
``GraphState`` carries the mutable fpg graph + evidence ledger and every
operation that does not require an agent call: accepting seed/hop verdicts,
cycle/reachability checks, candidate-path enumeration, edge-drop surgery,
and the final validation/output. All of this is unit-testable without a
workflow context; the only effect is the injected ``log`` callable.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from .lib.fpg import (
    edge_dict,
    fault_record,
    injection_node_id,
    is_link_injection,
    node_from_link_effect,
    node_from_seed,
    node_from_verdict,
    seed_effect_target,
)
from .lib.schema import (
    EdgeDrop,
    ExecutionError,
    GlobalAudit,
    HopLogEntry,
    HopResult,
    Injection,
    PropagationResult,
    SeedCoverageStatus,
    SeedResult,
)


def _reaches(adj: dict[str, list[str]], src: str, dst: str) -> bool:
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


def _entry_services_from_graph(graph: dict[str, list[list[str]]]) -> set[str]:
    callers: set[str] = set()
    callees: set[str] = set()
    for svc, neighbors in graph.items():
        for info in neighbors:
            if info[1] == "caller_to_callee":
                callers.add(svc)
                callees.add(info[0])
    explicit_entries = {
        svc for svc in callers | callees if svc in {"frontend", "ts-ui-dashboard"}
    }
    return explicit_entries or (callers - callees)


@dataclass
class Case:
    """Immutable per-case inputs assembled from the workflow args."""

    injections: list[Injection]
    window: dict[str, str]
    rel_mechanism: dict[str, str]
    graph: dict[str, list[list[str]]]
    infra_set: set[str]
    data_dir: str
    fault_docs: dict[str, str]
    skip_judge: bool
    judge_model: str | None
    gate_retries: int
    agent_retries: int
    max_audit_rounds: int
    seeds: set[str]
    entry_services: set[str]
    all_faults: list[list[str]]

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> Case:
        injections = cast(list[Injection], args["injections"])
        judge_model_raw = args.get("judge_model")
        judge_model = (
            judge_model_raw.strip()
            if isinstance(judge_model_raw, str) and judge_model_raw.strip()
            else None
        )
        graph = cast(dict[str, list[list[str]]], args["graph"])
        seeds = {injection_node_id(inj) for inj in injections if inj.get("target")}
        all_faults = [
            fault_record(inj)
            for inj in injections
            if inj.get("target")
        ]
        return cls(
            injections=injections,
            window=cast(dict[str, str], args["window"]),
            rel_mechanism=cast(dict[str, str], args.get("rel_mechanism", {})),
            graph=graph,
            infra_set=set(args.get("infra_nodes", [])),
            data_dir=cast(str, args["data_dir"]),
            fault_docs=args.get("fault_docs", {}),
            skip_judge=args.get("skip_judge", False),
            judge_model=judge_model,
            gate_retries=int(args.get("gate_retries", 3)),
            agent_retries=int(args.get("agent_retries", 3)),
            max_audit_rounds=int(args.get("max_audit_rounds", 3)),
            seeds=seeds,
            entry_services=_entry_services_from_graph(graph),
            all_faults=all_faults,
        )

    def case_summary(self) -> dict[str, Any]:
        return {
            "injections": self.injections,
            "entry_services": sorted(self.entry_services),
            "all_faults": self.all_faults,
        }


class GraphState:
    """Mutable fpg graph + evidence ledger with pure graph operations."""

    def __init__(self, case: Case, log: Callable[[str], None]) -> None:
        self.case = case
        self.log = log
        self.nodes: dict[str, dict[str, Any]] = {}  # svc -> fpg EventNode dict
        self.edges: list[dict[str, Any]] = []
        self.adj: dict[str, list[str]] = {}  # accepted-edge adjacency, cycle guard
        self.in_deg: dict[str, int] = {}
        self.verdicts: dict[str, HopResult] = {}  # "from__to" -> hop verdict
        self.hop_log: list[HopLogEntry] = []
        self.round_n: int = 0
        self.node_fault: dict[str, list[str]] = {}
        self.seed_verdicts: dict[str, SeedResult] = {}
        self.confirmed_seed_ids: set[str] = set()
        self.execution_errors: dict[str, ExecutionError] = {}
        self.gate_log: list[dict[str, Any]] = []
        self.audit_rounds_log: list[dict[str, Any]] = []
        self.propagation_roots: list[str] = []
        self.checked_edges: set[str] = set()
        self.reachability_warnings: list[str] = []
        self.unreachable: list[str] = []

    # -- Execution-error ledger -------------------------------------------
    def execution_key(self, stage: str, item: str) -> str:
        return stage + ":" + item

    def record_error(self, stage: str, item: str, reason: str) -> None:
        self.execution_errors[self.execution_key(stage, item)] = {
            "stage": stage,
            "item": item,
            "reason": reason,
        }
        self.log(f"⚠ {stage} {item}: {reason}")

    def clear_error(self, stage: str, item: str) -> None:
        self.execution_errors.pop(self.execution_key(stage, item), None)

    # -- Initialization ---------------------------------------------------
    def init_fresh(self) -> None:
        self.node_fault = {
            injection_node_id(inj): fault_record(inj)
            for inj in self.case.injections
            if inj.get("target")
        }

    # -- Graph indices ----------------------------------------------------
    def rebuild_adjacency(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        a: dict[str, list[str]] = {}
        d: dict[str, int] = {}
        for e in self.edges:
            a.setdefault(e["src"], []).append(e["dst"])
            d[e["dst"]] = d.get(e["dst"], 0) + 1
        return a, d

    def rebuild_graph_indices(self) -> None:
        self.adj, self.in_deg = self.rebuild_adjacency()

    def prune_unreachable_nodes(self) -> None:
        reachable: set[str] = set()
        stack = [seed for seed in self.confirmed_seed_ids if seed in self.nodes]
        while stack:
            cur = stack.pop()
            if cur in reachable:
                continue
            reachable.add(cur)
            stack.extend(self.adj.get(cur, []))
        for node_id in list(self.nodes):
            if node_id not in reachable:
                self.nodes.pop(node_id, None)
        self.edges[:] = [
            e for e in self.edges
            if e.get("src") in self.nodes and e.get("dst") in self.nodes
        ]
        self.rebuild_graph_indices()

    def unreachable_seed_nodes(self) -> list[str]:
        fpg_adj: dict[str, set[str]] = {}
        for e in self.edges:
            fpg_adj.setdefault(e["src"], set()).add(e["dst"])

        unreachable: list[str] = []
        for seed_svc in sorted(self.confirmed_seed_ids):
            if seed_svc not in self.nodes:
                unreachable.append(seed_svc)
                continue
            visited: set[str] = set()
            queue = [seed_svc]
            while queue:
                cur = queue.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                for nxt in fpg_adj.get(cur, set()):
                    queue.append(nxt)
            if not (visited & self.case.entry_services):
                unreachable.append(seed_svc)
        return unreachable

    # -- Accepting verdicts into the graph --------------------------------
    def accept_seed_node(self, inj: Injection, seed_verdict: SeedResult) -> str:
        root_id = injection_node_id(inj)
        self.nodes[root_id] = node_from_seed(inj, seed_verdict, self.case.window)
        if not is_link_injection(inj):
            self.propagation_roots.append(root_id)
            return root_id

        effect_target = seed_effect_target(inj, seed_verdict)
        if effect_target not in self.nodes:
            self.nodes[effect_target] = node_from_link_effect(
                inj,
                seed_verdict,
                self.case.window,
            )
        self.node_fault[effect_target] = fault_record(inj)
        if effect_target not in self.adj.get(root_id, []):
            self.edges.append(
                edge_dict(
                    root_id,
                    effect_target,
                    "link_to_service",
                    self.case.rel_mechanism,
                    "link fault manifests on the rule-bearing service side",
                )
            )
            self.adj.setdefault(root_id, []).append(effect_target)
            self.in_deg[effect_target] = self.in_deg.get(effect_target, 0) + 1
        self.propagation_roots.append(effect_target)
        return root_id

    def accept_hop_result(
        self,
        from_svc: str,
        to_svc: str,
        rel_type: str,
        result: HopResult,
        *,
        claim_override: str = "",
    ) -> bool:
        if _reaches(self.adj, to_svc, from_svc):
            self.hop_log.append(
                {
                    "round": self.round_n,
                    "from": from_svc,
                    "to": to_svc,
                    "verdict": "dropped_cycle",
                }
            )
            return False
        changed = False
        if to_svc not in self.nodes:
            self.nodes[to_svc] = node_from_verdict(to_svc, result, self.case.window)
            self.node_fault[to_svc] = self.node_fault.get(
                from_svc,
                [self.case.all_faults[0][0], self.case.all_faults[0][1]],
            )
            changed = True
        if to_svc not in self.adj.get(from_svc, []):
            self.edges.append(
                edge_dict(
                    from_svc,
                    to_svc,
                    rel_type,
                    self.case.rel_mechanism,
                    claim_override or str(result.get("claim", "")),
                )
            )
            self.adj.setdefault(from_svc, []).append(to_svc)
            self.in_deg[to_svc] = self.in_deg.get(to_svc, 0) + 1
            changed = True
        return changed

    # -- Snapshots for the audit layer ------------------------------------
    def graph_snapshot(self) -> dict[str, Any]:
        return {
            "nodes": [self.nodes[k] for k in sorted(self.nodes)],
            "edges": list(self.edges),
            "entry_services": sorted(self.case.entry_services),
        }

    def ledger_snapshot(self) -> dict[str, Any]:
        return {
            "seed_verdicts": self.seed_verdicts,
            "hop_verdicts": self.verdicts,
            "hop_log": self.hop_log,
            "gate_log": self.gate_log,
            "confirmed_seeds": sorted(self.confirmed_seed_ids),
        }

    # -- Candidate paths --------------------------------------------------
    def paths_from(self, seed: str, max_depth: int = 10) -> list[list[str]]:
        if seed not in self.nodes:
            return []
        paths: list[list[str]] = []
        stack: list[tuple[str, list[str]]] = [(seed, [seed])]
        while stack:
            cur, path = stack.pop()
            if cur in self.case.entry_services and len(path) > 1:
                paths.append(path)
                continue
            if len(path) >= max_depth:
                continue
            for nxt in self.adj.get(cur, []):
                if nxt in path:
                    continue
                stack.append((nxt, path + [nxt]))
        return paths

    def candidate_paths(self) -> list[tuple[str, str, list[str]]]:
        out: list[tuple[str, str, list[str]]] = []
        for seed in sorted(self.case.seeds):
            for idx, path in enumerate(self.paths_from(seed)):
                out.append((f"{seed}:path:{idx}", seed, path))
        return out

    def candidate_path_by_id(self, path_id: str) -> list[str] | None:
        for candidate_id, _, path in self.candidate_paths():
            if candidate_id == path_id:
                return path
        return None

    # -- Edge-drop surgery ------------------------------------------------
    @staticmethod
    def path_contains_edge(path: Sequence[str], src: str, dst: str) -> bool:
        return any(
            path[idx] == src and path[idx + 1] == dst
            for idx in range(len(path) - 1)
        )

    @staticmethod
    def path_edges(path: Sequence[str]) -> list[tuple[str, str]]:
        return [
            (path[idx], path[idx + 1])
            for idx in range(len(path) - 1)
        ]

    def drop_matches_reported_lineage(self, drop: EdgeDrop) -> bool:
        if drop.path_id:
            return any(
                path_id == drop.path_id
                and self.path_contains_edge(path, drop.src, drop.dst)
                for path_id, _, path in self.candidate_paths()
            )
        if drop.seed:
            return any(
                self.path_contains_edge(path, drop.src, drop.dst)
                for path in self.paths_from(drop.seed)
            )
        return True

    @staticmethod
    def valid_path_edges(
        causal_reports: Sequence[dict[str, Any]],
    ) -> set[tuple[str, str]]:
        protected: set[tuple[str, str]] = set()
        for report in causal_reports:
            if report.get("verdict") != "valid":
                continue
            path = report.get("path")
            if (
                not isinstance(path, list)
                or not all(isinstance(node, str) for node in path)
            ):
                continue
            protected.update(GraphState.path_edges(path))
        return protected

    def path_specific_cut(
        self,
        drop: EdgeDrop,
        protected_edges: set[tuple[str, str]],
    ) -> tuple[str, str] | None:
        current_edges = {
            (str(edge.get("src", "")), str(edge.get("dst", "")))
            for edge in self.edges
        }
        requested = (drop.src, drop.dst)
        if requested in current_edges and requested not in protected_edges:
            return requested

        if not drop.path_id:
            return None
        path = self.candidate_path_by_id(drop.path_id)
        if not path:
            return None
        for candidate in reversed(self.path_edges(path)):
            if candidate in current_edges and candidate not in protected_edges:
                return candidate
        return None

    def drop_audit_edges(
        self,
        drop_edges: Sequence[EdgeDrop],
        causal_reports: Sequence[dict[str, Any]],
    ) -> list[EdgeDrop]:
        if not drop_edges:
            return []
        doomed: set[tuple[str, str]] = set()
        applied: list[EdgeDrop] = []
        protected_edges = self.valid_path_edges(causal_reports)
        for item in drop_edges:
            if not item.src or not item.dst:
                continue
            if not self.drop_matches_reported_lineage(item):
                self.log(
                    f"  audit drop skipped for {item.src} -> {item.dst}: "
                    "edge is not on the reported seed/path lineage"
                )
                continue
            cut = self.path_specific_cut(item, protected_edges)
            if cut is None:
                self.log(
                    f"  audit drop skipped for {item.src} -> {item.dst}: "
                    "no unprotected edge can cut the reported invalid path"
                )
                continue
            if cut != (item.src, item.dst):
                self.log(
                    f"  audit drop remapped for {item.src} -> {item.dst}: "
                    f"cutting {cut[0]} -> {cut[1]} to preserve valid paths"
                )
                item = item.model_copy(
                    update={
                        "src": cut[0],
                        "dst": cut[1],
                        "reason": (
                            item.reason
                            + "\nHarness closure: requested edge is shared "
                            "with a valid causal path, so an unprotected edge "
                            "on the same invalid path was removed instead."
                        ),
                    }
                )
            doomed.add((item.src, item.dst))
            applied.append(item)
        if not doomed:
            return []
        before = len(self.edges)
        self.edges[:] = [
            e for e in self.edges
            if (str(e.get("src", "")), str(e.get("dst", ""))) not in doomed
        ]
        changed = len(self.edges) != before
        if changed:
            self.rebuild_graph_indices()
            self.prune_unreachable_nodes()
            return applied
        return []

    # -- Finalization + output --------------------------------------------
    def audit_seed_coverage(
        self,
        audit_result: GlobalAudit | None,
        seed: str,
    ) -> SeedCoverageStatus | None:
        if audit_result is None:
            return None
        return audit_result.seed_coverage.get(seed)

    def finalize(self, audit_result: GlobalAudit | None) -> None:
        # fpg rule: in-degree >= 2 requires combine; each confirmed edge is
        # an independently sufficient path, so the combination is OR.
        for svc, node in self.nodes.items():
            if self.in_deg.get(svc, 0) >= 2:
                node["combine"] = "OR"
            else:
                node.pop("combine", None)

        # Validation: unresolved seeds should either reach entry or be
        # audit-resolved.
        self.reachability_warnings = self.unreachable_seed_nodes()
        resolved_non_entry = {"local_only", "benign_or_no_effect"}
        self.unreachable = [
            seed
            for seed in self.reachability_warnings
            if self.audit_seed_coverage(audit_result, seed) not in resolved_non_entry
        ]
        for seed_svc in sorted(self.case.seeds):
            if seed_svc not in self.confirmed_seed_ids:
                self.log(f"⚠ seed {seed_svc}: not confirmed")
                continue
            if seed_svc in self.reachability_warnings:
                coverage = self.audit_seed_coverage(audit_result, seed_svc)
                if coverage in resolved_non_entry:
                    self.log(
                        f"ⓘ seed {seed_svc}: no entry path, resolved by audit as "
                        f"{coverage}"
                    )
                    continue
                self.log(
                    f"⚠ seed {seed_svc}: no path to entry services "
                    f"{sorted(self.case.entry_services)} in fpg"
                )

        if not self.unreachable:
            self.log("✓ no unresolved confirmed seeds lack entry-service coverage")

    def to_result(self, audit_result: GlobalAudit | None) -> PropagationResult:
        result_out: PropagationResult = {
            "nodes": [self.nodes[k] for k in sorted(self.nodes)],
            "edges": self.edges,
            "verdicts": self.verdicts,
            "hop_log": self.hop_log,
            "rounds": self.round_n,
            "seed_verdicts": self.seed_verdicts,
            "confirmed_seeds": sorted(self.confirmed_seed_ids),
            "gate_log": self.gate_log,
        }
        if audit_result:
            result_out["audit"] = audit_result.model_dump(mode="json")
        if self.audit_rounds_log:
            result_out["audit_rounds"] = self.audit_rounds_log
        if self.execution_errors:
            result_out["execution_errors"] = [
                self.execution_errors[key] for key in sorted(self.execution_errors)
            ]
        if self.reachability_warnings:
            result_out["reachability_warnings"] = self.reachability_warnings
        if self.unreachable:
            result_out["unreachable_seeds"] = self.unreachable
        return result_out
