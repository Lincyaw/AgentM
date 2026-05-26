#!/usr/bin/env python3
"""Producer-consumer fault propagation verifier.

Phase 0: build a relationship graph from traces (call graph,
bidirectional) and optionally deployment co-location.
Propagation: every confirmed node is a *producer* — it enqueues ALL
its neighbours. Each neighbour is a *consumer*: a hop-agent checks
whether it is genuinely degraded. If confirmed, it becomes a
producer itself. Edges between already-confirmed nodes are verified
by SQL alone (no agent). Same-round hops run in parallel.

Usage:
    uv run --no-sync python contrib/scenarios/verifier/eval/audit_case_propagate.py \\
        <case_dir> [--out <dir>] [--budget N] [--parallel N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FAULT_KINDS_DIR = REPO / "contrib" / "scenarios" / "verifier" / "fault_kinds"
SYNTHETIC = {
    "loadgenerator", "locust", "wrk2", "dsb-wrk2", "k6",
    "load-generator", "load_generator",
}


# ------------------------------------------------------------------
# Phase 0: relationship graph + injection parsing
# ------------------------------------------------------------------

def _duckdb_conn(data_dir: Path):  # noqa: ANN202
    import duckdb

    conn = duckdb.connect(":memory:")
    for f in sorted(data_dir.iterdir()):
        if f.is_file() and f.suffix == ".parquet" and f.name != "conclusion.parquet":
            conn.execute(
                f"CREATE VIEW {f.stem} AS "
                f"SELECT * FROM read_parquet('{f.as_posix()}')"
            )
    return conn


Rel = tuple[str, str, str, int]  # (service_a, service_b, rel_type, weight)


def get_relationships(data_dir: Path) -> list[Rel]:
    """Build bidirectional relationship list from call graph + deployment."""
    conn = _duckdb_conn(data_dir)
    rels: list[Rel] = []

    rows = conn.execute(
        "SELECT parent.service_name AS caller, "
        "       child.service_name AS callee, "
        "       COUNT(*) AS cnt "
        "FROM normal_traces child "
        "JOIN normal_traces parent "
        "  ON child.parent_span_id = parent.span_id "
        "WHERE child.service_name <> parent.service_name "
        "GROUP BY 1, 2 HAVING COUNT(*) >= 5 "
        "ORDER BY cnt DESC"
    ).fetchall()
    for caller, callee, cnt in rows:
        if caller not in SYNTHETIC and callee not in SYNTHETIC:
            rels.append((callee, caller, "callee_to_caller", int(cnt)))
            rels.append((caller, callee, "caller_to_callee", int(cnt)))

    try:
        dep_rows = conn.execute(
            "SELECT DISTINCT a.\"attr.k8s.node.name\" AS node, "
            "       a.service_name AS svc_a, b.service_name AS svc_b "
            "FROM normal_metrics a "
            "JOIN normal_metrics b "
            "  ON a.\"attr.k8s.node.name\" = b.\"attr.k8s.node.name\" "
            "WHERE a.service_name < b.service_name "
            "  AND a.\"attr.k8s.node.name\" IS NOT NULL"
        ).fetchall()
        for _node, svc_a, svc_b in dep_rows:
            if svc_a not in SYNTHETIC and svc_b not in SYNTHETIC:
                rels.append((svc_a, svc_b, "co_deployed", 1))
                rels.append((svc_b, svc_a, "co_deployed", 1))
    except Exception:  # noqa: BLE001
        pass

    conn.close()
    return rels


def _build_neighbor_graph(
    rels: list[Rel],
) -> dict[str, list[tuple[str, str, int]]]:
    """Return ``{service: [(neighbour, rel_type, weight)]}``."""
    graph: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for svc_a, svc_b, rel, weight in rels:
        key = (svc_a, svc_b, rel)
        if key not in seen:
            seen.add(key)
            graph[svc_a].append((svc_b, rel, weight))
    return dict(graph)


def get_injections(data_dir: Path) -> list[dict[str, str]]:
    """Extract ``[{target, chaos_type}]`` from injection.json."""
    injection = json.loads((data_dir / "injection.json").read_text())

    eng = injection.get("engine_config")
    if isinstance(eng, list) and eng and isinstance(eng[0], dict):
        return [
            {"target": e["app"], "chaos_type": e.get("chaos_type", "unknown")}
            for e in eng
        ]

    display = injection.get("display_config")
    if isinstance(display, str):
        try:
            display = json.loads(display)
        except Exception:
            display = {}
    if not isinstance(display, dict):
        display = {}

    point = display.get("injection_point", {})
    target = None
    if isinstance(point, dict):
        target = point.get("source_service") or point.get("app_label")
    if not target:
        gt = injection.get("ground_truth")
        if isinstance(gt, dict):
            svcs = gt.get("service") or []
            target = svcs[0] if svcs else None
        elif isinstance(gt, list) and gt and isinstance(gt[0], dict):
            svcs = gt[0].get("service") or []
            target = svcs[0] if svcs else None

    if not target:
        return []

    chaos_type = str(injection.get("fault_type", "unknown"))
    try:
        from rcabench_platform.v3.sdk.evaluation.v2.fault_kind import (
            chaos_type_from_index,
            map_chaos_type,
        )
        chaos_type = str(map_chaos_type(
            chaos_type_from_index(int(chaos_type))
        ).value)
    except Exception:
        pass
    return [{"target": target, "chaos_type": chaos_type}]


def get_target_evidence(data_dir: Path, target: str) -> dict[str, object]:
    """Quick SQL check: latency comparison for the injection target."""
    conn = _duckdb_conn(data_dir)
    rows = conn.execute(
        f"SELECT 'normal' AS win, AVG(duration)/1e6, COUNT(*) "
        f"FROM normal_traces WHERE service_name = '{target}' "
        f"UNION ALL "
        f"SELECT 'abnormal', AVG(duration)/1e6, COUNT(*) "
        f"FROM abnormal_traces WHERE service_name = '{target}'"
    ).fetchall()
    conn.close()

    if len(rows) == 2 and rows[0][1] and rows[1][1]:
        ratio = rows[1][1] / rows[0][1] if rows[0][1] > 0 else 0
        return {
            "normal_avg_ms": round(rows[0][1], 3),
            "abnormal_avg_ms": round(rows[1][1], 3),
            "ratio": round(ratio, 1),
        }
    return {}


def _load_fault_doc(fault_kind: str) -> str:
    """Read the per-fault-kind reference doc, or return empty."""
    p = FAULT_KINDS_DIR / f"{fault_kind}.md"
    if p.is_file():
        return p.read_text().strip()
    return ""


# ------------------------------------------------------------------
# Hop agent
# ------------------------------------------------------------------

_REL_DESCRIPTIONS = {
    "callee_to_caller": "{to} calls {frm} — if {frm} is slow/broken, "
                        "{to} (the caller) may block waiting for it.",
    "caller_to_callee": "{frm} calls {to} — if {frm} has a network fault, "
                        "its outgoing calls to {to} may carry corrupted "
                        "payloads or timeouts.",
    "co_deployed": "{frm} and {to} share a k8s node — resource contention "
                   "(CPU/memory/disk) from one can degrade the other.",
}


def _walk(obj: object):  # noqa: ANN202
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def extract_hop_verdict(obs_dir: Path) -> dict | None:
    """Walk observability JSONL to find the submitted HopVerdict."""
    best = None
    for f in sorted(obs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        for line in f.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            for s in _walk(row):
                t = s.strip()
                if (
                    t.startswith("{")
                    and '"verdict"' in t
                    and '"symptom_evidence"' in t
                ):
                    try:
                        obj = json.loads(t)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and "verdict" in obj:
                        best = obj
    return best


def run_hop(
    data_dir: Path,
    from_service: str,
    to_service: str,
    rel_type: str,
    fault_kind: str,
    fault_doc: str,
    injection_target: str,
    out_dir: Path,
    budget: int,
    timeout: int,
) -> dict | None:
    """Run one hop-agent and return its verdict dict (or None)."""
    hop_dir = out_dir / "hops" / f"{from_service}__{to_service}"
    hop_dir.mkdir(parents=True, exist_ok=True)

    rel_desc = _REL_DESCRIPTIONS.get(rel_type, "{frm} and {to} are related.")
    rel_text = rel_desc.format(frm=from_service, to=to_service)

    parts = [
        f"Confirmed degraded: **{from_service}**",
        f"Service to check: **{to_service}**",
        f"Relationship: {rel_text}",
        f"Fault injected: {fault_kind} on {injection_target}",
    ]
    if fault_doc:
        parts.append(f"\n## Fault reference ({fault_kind})\n{fault_doc}")
    parts.append(
        f"\nDetermine whether {to_service} is genuinely degraded due to "
        f"this relationship with {from_service}. Query normal_* vs "
        f"abnormal_* tables, verify the relationship, then submit."
    )
    prompt = "\n".join(parts)

    env = dict(os.environ)
    env["AGENTM_PROJECT_ROOT"] = str(REPO)
    env["AGENTM_RCA_DATA_DIR"] = str(data_dir)
    base = (
        ["agentm"]
        if shutil.which("agentm")
        else ["uv", "run", "--no-sync", "agentm"]
    )
    cmd = [
        *base, "--scenario", "verifier",
        "--provider", env.get("AGENTM_PROVIDER", "openai"),
        "--model", env.get("AGENTM_MODEL", "K2.6"),
        "--cwd", str(hop_dir),
        "--max-tool-calls", str(budget),
        prompt,
    ]

    try:
        with open(hop_dir / "stdout.log", "w") as fout, \
             open(hop_dir / "stderr.log", "w") as ferr:
            r = subprocess.run(
                cmd, env=env, stdout=fout, stderr=ferr, timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        (hop_dir / "stderr.log").write_text("TIMEOUT\n")
        return None

    obs_dir = hop_dir / ".agentm" / "observability"
    return extract_hop_verdict(obs_dir) if obs_dir.exists() else None


# ------------------------------------------------------------------
# Edge verification (SQL-only, no agent)
# ------------------------------------------------------------------

def _verify_edge_sql(data_dir: Path, from_svc: str, to_svc: str) -> bool:
    """Check that a direct relationship exists between the two services."""
    conn = _duckdb_conn(data_dir)
    row = conn.execute(
        f"SELECT COUNT(*) FROM normal_traces child "
        f"JOIN normal_traces parent "
        f"  ON child.parent_span_id = parent.span_id "
        f"WHERE (parent.service_name = '{to_svc}' "
        f"   AND child.service_name = '{from_svc}') "
        f"   OR (parent.service_name = '{from_svc}' "
        f"   AND child.service_name = '{to_svc}')"
    ).fetchone()
    conn.close()
    return bool(row and row[0] >= 5)


# ------------------------------------------------------------------
# Producer-consumer propagation
# ------------------------------------------------------------------

def propagate(
    data_dir: Path,
    injections: list[dict[str, str]],
    neighbor_graph: dict[str, list[tuple[str, str, int]]],
    fault_doc: str,
    out_dir: Path,
    *,
    budget: int,
    parallel: int,
    timeout: int,
) -> dict:
    """Producer-consumer fault propagation.

    Every confirmed node is a *producer*: it enqueues ALL its
    neighbours (callers, callees, co-deployed). For each neighbour:
      - already confirmed → SQL-verify the edge (no agent)
      - not yet confirmed → run a hop agent (may confirm the node)
    No edge is skipped; no node agent runs twice.
    """
    confirmed: set[str] = set()
    agent_checked: set[str] = set()
    checked_edges: set[tuple[str, str]] = set()
    queue: deque[str] = deque()
    edges: list[tuple[str, str]] = []
    node_evidence: dict[str, dict] = {}
    hop_log: list[dict] = []

    fault_kind = injections[0]["chaos_type"] if injections else "unknown"
    inj_target = injections[0]["target"] if injections else "unknown"

    for inj in injections:
        target = inj["target"]
        if target and target not in SYNTHETIC:
            ev = get_target_evidence(data_dir, target)
            confirmed.add(target)
            agent_checked.add(target)
            queue.append(target)
            node_evidence[target] = {"source": "injection_target", **ev}

    round_n = 0
    while queue:
        round_n += 1
        level_size = len(queue)

        sql_edges: list[tuple[str, str]] = []
        # to_service -> [(from_service, rel_type)]
        agent_hops: dict[str, list[tuple[str, str]]] = {}

        for _ in range(level_size):
            current = queue.popleft()
            for neighbor, rel_type, _weight in neighbor_graph.get(current, []):
                if neighbor in SYNTHETIC:
                    continue
                edge = (current, neighbor)
                if edge in checked_edges:
                    continue
                checked_edges.add(edge)

                if neighbor in confirmed:
                    sql_edges.append(edge)
                else:
                    agent_hops.setdefault(neighbor, []).append(
                        (current, rel_type)
                    )

        for from_svc, to_svc in sql_edges:
            if _verify_edge_sql(data_dir, from_svc, to_svc):
                edges.append((from_svc, to_svc))
                hop_log.append({
                    "round": round_n, "from": from_svc, "to": to_svc,
                    "verdict": "edge_sql",
                })
                print(f"  edge (SQL): {from_svc} -> {to_svc}")

        need_agent = {
            to: froms for to, froms in agent_hops.items()
            if to not in agent_checked
        }

        if need_agent:
            print(f"  round {round_n}: {len(need_agent)} agents, "
                  f"{len(sql_edges)} SQL edges")
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = {
                    pool.submit(
                        run_hop, data_dir,
                        froms[0][0], to_svc, froms[0][1],
                        fault_kind, fault_doc, inj_target,
                        out_dir, budget, timeout,
                    ): (to_svc, froms)
                    for to_svc, froms in need_agent.items()
                }
                for future in as_completed(futures):
                    to_svc, froms = futures[future]
                    agent_checked.add(to_svc)
                    result = future.result()
                    verdict = result.get("verdict") if result else None
                    hop_log.append({
                        "round": round_n, "from": froms[0][0],
                        "to": to_svc, "verdict": verdict,
                    })
                    print(f"    {froms[0][0]} -> {to_svc}: "
                          f"{verdict or 'no-result'}")
                    if verdict == "confirmed":
                        confirmed.add(to_svc)
                        node_evidence[to_svc] = {
                            "source": "hop_agent", **(result or {}),
                        }
                        queue.append(to_svc)
                        edges.append((froms[0][0], to_svc))
                        for other_from, _other_rel in froms[1:]:
                            if _verify_edge_sql(
                                data_dir, other_from, to_svc
                            ):
                                edges.append((other_from, to_svc))
                                print(f"    {other_from} -> "
                                      f"{to_svc}: edge_sql")

    return {
        "confirmed_nodes": sorted(confirmed),
        "edges": edges,
        "node_evidence": node_evidence,
        "hop_log": hop_log,
        "rounds": round_n,
    }


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

def build_report(
    result: dict, injections: list[dict[str, str]],
) -> dict:
    report: dict = {
        "injections": [
            {
                "target_service": inj["target"],
                "fault_kind": inj["chaos_type"],
                "verdict": "true",
                "rationale": "injection target (seed)",
            }
            for inj in injections
        ],
        "propagation_nodes": [],
        "propagation_edges": [],
    }
    for node in result["confirmed_nodes"]:
        ev = result["node_evidence"].get(node, {})
        report["propagation_nodes"].append({
            "service": node,
            "symptom_evidence": ev.get("symptom_evidence", []),
        })
    for from_svc, to_svc in result["edges"]:
        ev = result["node_evidence"].get(to_svc, {})
        report["propagation_edges"].append({
            "from_service": from_svc,
            "to_service": to_svc,
            "relationship_sql": ev.get("relationship_sql", ""),
            "claim": ev.get("claim", ""),
        })
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--budget", type=int, default=30,
                    help="tool-call budget per hop agent")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=180,
                    help="timeout per hop agent (seconds)")
    args = ap.parse_args()

    data_dir = args.case_dir.resolve()
    out = (args.out or data_dir / ".verify_propagate").resolve()
    out.mkdir(parents=True, exist_ok=True)

    injections = get_injections(data_dir)
    if not injections:
        print("ERROR: no injections found in injection.json")
        return 1

    fault_kind = injections[0]["chaos_type"]
    fault_doc = _load_fault_doc(fault_kind)

    rels = get_relationships(data_dir)
    neighbor_graph = _build_neighbor_graph(rels)
    (out / "relationships.json").write_text(json.dumps(
        [{"a": a, "b": b, "rel": r, "weight": w} for a, b, r, w in rels],
        indent=2, ensure_ascii=False,
    ))

    total_edges = sum(len(v) for v in neighbor_graph.values())
    print(f"Injections: {[(i['target'], i['chaos_type']) for i in injections]}")
    print(f"Fault doc: {'loaded' if fault_doc else 'not found'} "
          f"({fault_kind})")
    print(f"Relationship graph: {total_edges} directed edges "
          f"({len(neighbor_graph)} services)")

    result = propagate(
        data_dir, injections, neighbor_graph, fault_doc, out,
        budget=args.budget, parallel=args.parallel, timeout=args.timeout,
    )
    (out / "propagation_trace.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str)
    )

    report = build_report(result, injections)
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print(f"\nRounds: {result['rounds']}")
    print(f"Confirmed: {result['confirmed_nodes']}")
    print(f"Edges ({len(result['edges'])}): {result['edges']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
