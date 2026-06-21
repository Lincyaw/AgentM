"""Quality report for verifier batch outputs."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .injection import get_injections


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    return value if isinstance(value, dict) else None


def _entry_like_nodes(node_ids: set[str]) -> set[str]:
    explicit = {n for n in node_ids if n in {"frontend", "ts-ui-dashboard"}}
    if explicit:
        return explicit
    return {n for n in node_ids if "frontend" in n or "dashboard" in n}


def _reachable_from(seed: str, adj: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    stack = [seed]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj.get(cur, ()))
    return seen


def _case_faults(dataset_dir: Path | None, case_name: str) -> list[dict[str, Any]]:
    if dataset_dir is None:
        return []
    case_data_dir = dataset_dir / case_name
    if not case_data_dir.is_dir():
        return []
    try:
        injections = get_injections(case_data_dir)
    except Exception:  # noqa: BLE001
        return []
    return [
        {
            "node_id": inj.get("node_id"),
            "chaos_type": inj.get("chaos_type"),
            "target": inj.get("target"),
            "params": inj.get("params"),
        }
        for inj in injections
    ]


def _fault_tuple(faults: list[dict[str, Any]]) -> str:
    kinds = [f.get("chaos_type") for f in faults if f.get("chaos_type")]
    return " + ".join(str(k) for k in kinds) if kinds else "unknown"


def _fault_seed_ids(faults: list[dict[str, Any]]) -> list[str]:
    seeds: list[str] = []
    for fault in faults:
        seed = fault.get("node_id") or fault.get("target")
        if isinstance(seed, str) and seed:
            seeds.append(seed)
    return seeds


def _resolve_case_from_meta(
    meta: dict[str, Any] | None,
    case_name: str,
    dataset_dir: Path | None,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    faults = _case_faults(dataset_dir, case_name)
    if not meta:
        return case_name, faults, _fault_seed_ids(faults)

    meta_case_name = meta.get("scenario_id")
    if not isinstance(meta_case_name, str) or not meta_case_name:
        return case_name, faults, _fault_seed_ids(faults)
    if meta_case_name == case_name:
        return case_name, faults, _fault_seed_ids(faults)

    meta_faults = _case_faults(dataset_dir, meta_case_name)
    if not meta_faults:
        return case_name, faults, _fault_seed_ids(faults)
    return meta_case_name, meta_faults, _fault_seed_ids(meta_faults)


def _seed_verdict_map(meta: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not meta:
        return {}
    raw = meta.get("seed_verdicts", {})
    if not isinstance(raw, dict):
        return {}
    return {
        seed: verdict
        for seed, verdict in raw.items()
        if isinstance(seed, str) and isinstance(verdict, dict)
    }


def _case_quality(case_dir: Path, dataset_dir: Path | None = None) -> dict[str, Any]:
    case_name = case_dir.name
    meta_path = case_dir / "run_meta.json"
    scenario_path = case_dir / "fpg_scenario.json"
    meta = _load_json(meta_path) if meta_path.exists() else None
    case_name, faults, dataset_seeds = _resolve_case_from_meta(
        meta,
        case_name,
        dataset_dir,
    )

    if not meta_path.exists() and not scenario_path.exists():
        return {
            "case": case_name,
            "faults": faults,
            "fault_tuple": _fault_tuple(faults),
            "status": "missing_outputs",
            "passed": False,
            "failures": ["missing_outputs"],
        }

    if meta and meta.get("error") == "no seeds confirmed":
        seed_verdicts = _seed_verdict_map(meta)
        if any(
            verdict.get("_error")
            for verdict in seed_verdicts.values()
        ):
            return {
                "case": case_name,
                "faults": faults,
                "fault_tuple": _fault_tuple(faults),
                "status": "seed_agent_error",
                "passed": False,
                "failures": ["run_error"],
                "error": meta.get("error"),
                "seed_verdicts": seed_verdicts,
            }
        seeds = sorted(set(dataset_seeds) | set(seed_verdicts))
        error_seed_status: list[dict[str, Any]] = []
        for seed in seeds:
            verdict = seed_verdicts.get(seed)
            predicate = verdict.get("predicate") if verdict else None
            seed_verdict = verdict.get("verdict") if verdict else None
            error_seed_status.append({
                "seed": seed,
                "confirmed": False,
                "seed_verdict_available": verdict is not None,
                "seed_verdict": seed_verdict,
                "seed_verdict_confirmed": seed_verdict == "confirmed"
                if verdict is not None
                else None,
                "node_present": False,
                "graph_confirmed": False,
                "predicate": predicate,
                "reaches_entry": False,
                "reached_entry": [],
            })
        error_failures = ["seed_not_confirmed"]
        if not seeds:
            error_failures.append("missing_seed_records")
        return {
            "case": case_name,
            "faults": faults,
            "fault_tuple": _fault_tuple(faults),
            "status": "no_seeds_confirmed",
            "passed": False,
            "failures": error_failures,
            "error": meta.get("error"),
            "seeds": seeds,
            "seed_status": error_seed_status,
            "reachable_seeds": [],
            "missing_seeds": seeds,
            "unreachable_seeds": seeds,
            "workflow_unreachable_seeds": [],
            "workflow_unreachable_warnings": [],
            "entry_nodes": [],
            "node_count": 0,
            "edge_count": 0,
            "seed_verdicts": seed_verdicts if isinstance(seed_verdicts, dict) else {},
        }

    if meta and meta.get("error"):
        return {
            "case": case_name,
            "faults": faults,
            "fault_tuple": _fault_tuple(faults),
            "status": "error",
            "passed": False,
            "failures": ["run_error"],
            "error": meta.get("error"),
            "seed_verdicts": meta.get("seed_verdicts", {}),
        }

    scenario = _load_json(scenario_path) if scenario_path.exists() else None
    if scenario is None:
        return {
            "case": case_name,
            "faults": faults,
            "fault_tuple": _fault_tuple(faults),
            "status": "missing_fpg",
            "passed": False,
            "failures": ["missing_fpg"],
        }

    scenario_case_name = str(scenario.get("scenario_id") or case_name)
    if scenario_case_name != case_name:
        case_name = scenario_case_name
        scenario_faults = _case_faults(dataset_dir, case_name)
        if scenario_faults:
            faults = scenario_faults
            dataset_seeds = _fault_seed_ids(faults)
    graph = scenario.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    injections = scenario.get("injections", [])
    scenario_seeds = [i.get("node_id") for i in injections if i.get("node_id")]
    seed_verdicts = _seed_verdict_map(meta)
    raw_confirmed_seeds = meta.get("confirmed_seeds", []) if meta else []
    confirmed_seed_records = {
        seed for seed in raw_confirmed_seeds if isinstance(seed, str)
    }
    raw_workflow_unreachable = meta.get("unreachable_seeds", []) if meta else []
    workflow_unreachable = list(dict.fromkeys(raw_workflow_unreachable))
    seeds = sorted(
        set(dataset_seeds)
        | set(seed_verdicts)
        | set(scenario_seeds)
        | set(workflow_unreachable)
    )
    node_by_id = {
        node_id: n
        for n in nodes
        if isinstance(n, dict)
        for node_id in [n.get("id")]
        if isinstance(node_id, str)
    }
    node_ids = set(node_by_id)
    entry_nodes = _entry_like_nodes(node_ids)

    adj: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = edge.get("src")
        dst = edge.get("dst")
        if isinstance(src, str) and isinstance(dst, str):
            adj[src].add(dst)

    seed_status: list[dict[str, Any]] = []
    for seed in seeds:
        seed_node = node_by_id.get(seed)
        predicate = seed_node.get("predicate") if seed_node else None
        graph_confirmed = seed_node is not None and predicate != "other"
        verdict = seed_verdicts.get(seed)
        seed_verdict = verdict.get("verdict") if verdict else None
        if verdict is not None:
            confirmed = seed_verdict == "confirmed" and graph_confirmed
        elif confirmed_seed_records:
            confirmed = seed in confirmed_seed_records and graph_confirmed
        else:
            confirmed = graph_confirmed
        reachable = _reachable_from(seed, adj) if seed_node is not None else set()
        reached_entry = sorted(reachable & entry_nodes)
        seed_status.append({
            "seed": seed,
            "confirmed": confirmed,
            "seed_verdict_available": verdict is not None,
            "seed_verdict": seed_verdict,
            "seed_verdict_confirmed": seed_verdict == "confirmed"
            if verdict is not None
            else None,
            "node_present": seed_node is not None,
            "included_in_fpg_injections": seed in scenario_seeds,
            "graph_confirmed": graph_confirmed,
            "predicate": predicate,
            "reaches_entry": confirmed and bool(reached_entry),
            "reached_entry": reached_entry,
        })

    missing_seeds = [s["seed"] for s in seed_status if not s["confirmed"]]
    unreachable_seeds = [s["seed"] for s in seed_status if s["confirmed"] and not s["reaches_entry"]]
    reachable_seeds = [s["seed"] for s in seed_status if s["reaches_entry"]]
    actual_unreachable = set(missing_seeds) | set(unreachable_seeds)
    workflow_unreachable_failures = sorted(set(workflow_unreachable) & actual_unreachable)
    workflow_unreachable_warnings = sorted(set(workflow_unreachable) - actual_unreachable)

    failures: list[str] = []
    if not seeds:
        failures.append("missing_seed_records")
    if missing_seeds:
        failures.append("seed_not_confirmed")
    if unreachable_seeds:
        failures.append("seed_no_entry_path")
    if workflow_unreachable_failures:
        failures.append("workflow_unreachable_seeds")
    if not edges:
        failures.append("zero_edges")
    if reachable_seeds and (unreachable_seeds or missing_seeds):
        failures.append("partial_seed_coverage")

    return {
        "case": case_name,
        "faults": faults,
        "fault_tuple": _fault_tuple(faults),
        "status": "ok",
        "passed": not failures,
        "failures": failures,
        "seeds": seeds,
        "seed_status": seed_status,
        "reachable_seeds": reachable_seeds,
        "missing_seeds": missing_seeds,
        "unreachable_seeds": sorted(set(unreachable_seeds) | set(workflow_unreachable_failures)),
        "workflow_unreachable_seeds": workflow_unreachable,
        "workflow_unreachable_warnings": workflow_unreachable_warnings,
        "entry_nodes": sorted(entry_nodes),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "seed_verdicts": seed_verdicts,
        "confirmed_seeds": sorted(confirmed_seed_records),
    }


def build_quality_report(
    run_dir: Path,
    dataset_dir: Path | None = None,
) -> dict[str, Any]:
    if (run_dir / "run_meta.json").exists() or (run_dir / "fpg_scenario.json").exists():
        case_dirs = [run_dir]
    else:
        case_dirs = [
            p
            for p in sorted(run_dir.iterdir())
            if p.is_dir() and not p.name.startswith(".")
        ]
    cases = [_case_quality(p, dataset_dir) for p in case_dirs]
    status_counts = Counter(c["status"] for c in cases)
    failure_counts: Counter[str] = Counter()
    unreachable_counts: Counter[str] = Counter()
    fault_counts: Counter[str] = Counter()
    failed_fault_counts: Counter[str] = Counter()
    failure_by_fault: Counter[tuple[str, str]] = Counter()
    unreachable_by_fault: Counter[tuple[str, str]] = Counter()
    for case in cases:
        fault_tuple = str(case.get("fault_tuple") or "unknown")
        fault_counts[fault_tuple] += 1
        if not case.get("passed"):
            failed_fault_counts[fault_tuple] += 1
        failure_counts.update(case.get("failures", []))
        unreachable_counts.update(case.get("unreachable_seeds", []))
        for failure in case.get("failures", []):
            failure_by_fault[(fault_tuple, str(failure))] += 1
        for seed in case.get("unreachable_seeds", []):
            unreachable_by_fault[(fault_tuple, str(seed))] += 1

    passed = sum(1 for c in cases if c.get("passed"))
    return {
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
        "summary": {
            "total_cases": len(cases),
            "passed": passed,
            "failed": len(cases) - passed,
            "status_counts": dict(status_counts),
            "failure_counts": dict(failure_counts),
            "fault_counts": dict(fault_counts),
            "failed_fault_counts": dict(failed_fault_counts),
            "top_failures_by_fault": [
                {
                    "fault_tuple": fault_tuple,
                    "failure": failure,
                    "count": count,
                }
                for (fault_tuple, failure), count
                in failure_by_fault.most_common()
            ],
            "top_unreachable_seeds": [
                {"seed": seed, "count": count}
                for seed, count in unreachable_counts.most_common()
            ],
            "top_unreachable_seeds_by_fault": [
                {
                    "fault_tuple": fault_tuple,
                    "seed": seed,
                    "count": count,
                }
                for (fault_tuple, seed), count
                in unreachable_by_fault.most_common()
            ],
        },
        "cases": cases,
    }


__all__ = ["build_quality_report"]
