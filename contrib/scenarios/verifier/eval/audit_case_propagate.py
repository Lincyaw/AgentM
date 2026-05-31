#!/usr/bin/env python3
"""Producer-consumer fault propagation verifier.

Phase 0: build a relationship graph from traces (call graph,
bidirectional) and optionally deployment co-location.
Propagation: every confirmed node is a *producer* — it enqueues ALL
its neighbours. Each neighbour is a *consumer*: a hop-agent checks
whether it is genuinely degraded. If confirmed, it becomes a
producer itself. Edges between already-confirmed nodes are verified
by SQL alone (no agent). Same-round hops run in parallel.

Usage (single case):
    uv run python audit_case_propagate.py run <case_dir> [--model X]

Usage (batch — ablation runs):
    uv run python audit_case_propagate.py batch <dataset_dir> \\
        --run-dir /tmp/verifier-seed2pro --model litellm --limit 10
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer

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
    """Build bidirectional relationship list from call graph + deployment.

    Uses the UNION of normal and abnormal windows so that edges only
    visible in one window (e.g. a short normal capture) are not lost.
    """
    conn = _duckdb_conn(data_dir)
    rels: list[Rel] = []

    rows = conn.execute(
        "WITH all_traces AS ("
        "  SELECT parent_span_id, span_id, service_name "
        "  FROM normal_traces "
        "  UNION ALL "
        "  SELECT parent_span_id, span_id, service_name "
        "  FROM abnormal_traces"
        ") "
        "SELECT parent.service_name AS caller, "
        "       child.service_name AS callee, "
        "       COUNT(*) AS cnt "
        "FROM all_traces child "
        "JOIN all_traces parent "
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
            "WITH all_metrics AS ("
            "  SELECT service_name, \"attr.k8s.node.name\" "
            "  FROM normal_metrics "
            "  UNION ALL "
            "  SELECT service_name, \"attr.k8s.node.name\" "
            "  FROM abnormal_metrics"
            ") "
            "SELECT DISTINCT a.\"attr.k8s.node.name\" AS node, "
            "       a.service_name AS svc_a, b.service_name AS svc_b "
            "FROM all_metrics a "
            "JOIN all_metrics b "
            "  ON a.\"attr.k8s.node.name\" = b.\"attr.k8s.node.name\" "
            "WHERE a.service_name < b.service_name "
            "  AND a.service_name <> '' AND b.service_name <> '' "
            "  AND a.service_name IS NOT NULL AND b.service_name IS NOT NULL "
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


# Uninstrumented backing components (DB / cache / broker) emit no spans
# of their own, so they never appear as a trace ``service_name`` — only
# in metrics. A node present in metrics but absent from traces in BOTH
# windows is infra; a node absent from *abnormal* traces only is a
# crashed-but-instrumented service (e.g. a PodFailure'd app), not infra.
_DB_INFRA = re.compile(r"mysql|mariadb|postgres|sqlserver|oracle|cockroach", re.I)
# Known uninstrumented backing components. A metrics-only node is treated as
# infra only if its name matches one of these — otherwise a real but idle
# (un-exercised in this case) service would be mislabelled as infra.
_INFRA_NAME = re.compile(
    r"mysql|mariadb|postgres|sqlserver|oracle|cockroach|mongo|redis|"
    r"memcached|rabbitmq|kafka|consul|etcd|zookeeper|nacos|elasticsearch|"
    r"cassandra|clickhouse|minio",
    re.I,
)
# SQL/ORM client span-name shapes that mark a span as a database call.
_DBOP_SPAN_SQL = (
    "(span_name LIKE 'SELECT%' OR span_name LIKE 'INSERT%' "
    "OR span_name LIKE 'UPDATE%' OR span_name LIKE 'DELETE%' "
    "OR span_name LIKE 'Transaction%' OR span_name LIKE 'COMMIT%' "
    "OR span_name LIKE 'Session%' OR span_name LIKE '%Repository%')"
)


def _trace_services(conn) -> set[str]:  # noqa: ANN001
    out: set[str] = set()
    for tbl in ("normal_traces", "abnormal_traces"):
        try:
            rows = conn.execute(f"SELECT DISTINCT service_name FROM {tbl}").fetchall()
        except Exception:  # noqa: BLE001 - table may be absent
            continue
        out.update(r[0] for r in rows if r[0])
    return out


def get_infra_nodes(data_dir: Path) -> set[str]:
    """Backing components present in metrics but never in traces.

    These are the nodes the trace-only relationship graph is structurally
    blind to (``mysql``, ``rabbitmq``, ``mongodb-*``, ``memcached-*`` …).
    """
    conn = _duckdb_conn(data_dir)
    try:
        metric_svcs = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT service_name FROM normal_metrics"
            ).fetchall()
            if r[0]
        }
    except Exception:  # noqa: BLE001
        conn.close()
        return set()
    infra = {
        s
        for s in metric_svcs - _trace_services(conn)
        if s not in SYNTHETIC and _INFRA_NAME.search(s)
    }
    conn.close()
    return infra


def get_infra_edges(data_dir: Path, infra_nodes: set[str]) -> list[Rel]:
    """Link each infra node to the services that depend on it.

    SQL databases are linked from every service emitting Client DB-op
    spans (the DB call lives inside the *caller*). Named per-service
    backends (``mongodb-profile``, ``memcached-rate-1``) are linked to the
    service whose name is a token of the backend name.
    """
    if not infra_nodes:
        return []
    conn = _duckdb_conn(data_dir)
    rels: list[Rel] = []

    db_callers: list[str] = []
    try:
        db_callers = [
            r[0]
            for r in conn.execute(
                "SELECT service_name, COUNT(*) AS c FROM normal_traces "
                f"WHERE \"attr.span_kind\" = 'Client' AND {_DBOP_SPAN_SQL} "
                "GROUP BY 1 HAVING COUNT(*) >= 5"
            ).fetchall()
            if r[0] and r[0] not in SYNTHETIC
        ]
    except Exception:  # noqa: BLE001
        pass

    trace_svcs = _trace_services(conn)
    conn.close()

    for node in infra_nodes:
        if _DB_INFRA.search(node):
            callers = db_callers
        else:
            # mongodb-profile / memcached-rate-1 -> match the owning service
            tokens = set(node.replace("_", "-").split("-"))
            callers = [s for s in trace_svcs if s in tokens or s in node]
        for svc in callers:
            rels.append((svc, node, "infra_dependency", 1))
            rels.append((node, svc, "infra_dependency", 1))
    return rels


def get_node_map(data_dir: Path) -> dict[str, str]:
    """Return ``{service_name: k8s_node_name}`` from metrics."""
    conn = _duckdb_conn(data_dir)
    try:
        rows = conn.execute(
            "SELECT DISTINCT service_name, \"attr.k8s.node.name\" "
            "FROM normal_metrics "
            "WHERE service_name IS NOT NULL AND service_name <> '' "
            "  AND \"attr.k8s.node.name\" IS NOT NULL"
        ).fetchall()
    except Exception:  # noqa: BLE001
        conn.close()
        return {}
    conn.close()
    return {svc: node for svc, node in rows}


def check_node_degraded(data_dir: Path, node_name: str) -> bool:
    """Quick check: does this k8s node show CPU or memory degradation?

    Compares average CPU/memory usage between normal and abnormal windows.
    Returns True if degradation exceeds 20%, meaning co_deployed services
    on this node may be genuinely affected.
    """
    conn = _duckdb_conn(data_dir)
    try:
        rows = conn.execute(
            "SELECT 'normal' AS win, "
            "  AVG(CASE WHEN metric_name LIKE '%cpu%' THEN value END), "
            "  AVG(CASE WHEN metric_name LIKE '%memory%' THEN value END) "
            "FROM normal_metrics "
            "WHERE \"attr.k8s.node.name\" = ? "
            "UNION ALL "
            "SELECT 'abnormal', "
            "  AVG(CASE WHEN metric_name LIKE '%cpu%' THEN value END), "
            "  AVG(CASE WHEN metric_name LIKE '%memory%' THEN value END) "
            "FROM abnormal_metrics "
            "WHERE \"attr.k8s.node.name\" = ?",
            [node_name, node_name],
        ).fetchall()
    except Exception:  # noqa: BLE001
        conn.close()
        return True  # err on the side of checking
    conn.close()

    if len(rows) != 2:
        return True
    normal_cpu, normal_mem = rows[0][1], rows[0][2]
    abnormal_cpu, abnormal_mem = rows[1][1], rows[1][2]

    threshold = 1.2  # 20% increase
    if normal_cpu and abnormal_cpu and abnormal_cpu > normal_cpu * threshold:
        return True
    if normal_mem and abnormal_mem and abnormal_mem > normal_mem * threshold:
        return True
    return False


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
    # duration is microseconds — convert to ms with /1e3
    rows = conn.execute(
        "SELECT 'normal' AS win, AVG(duration)/1e3, COUNT(*) "
        "FROM normal_traces WHERE service_name = ? "
        "UNION ALL "
        "SELECT 'abnormal', AVG(duration)/1e3, COUNT(*) "
        "FROM abnormal_traces WHERE service_name = ?",
        [target, target],
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


# chaos_type names whose doc filename abbreviates differently than a plain
# normalization would catch (same fault, shorter slug).
_FAULT_DOC_ALIAS = {
    "memorystress": "memstress",
    "jvmlatency": "jvmmethodlatency",
    "podkill": "podfailure",
    "containerkill": "podfailure",
}


def _norm_fault(name: str) -> str:
    """Lowercase, strip non-alphanumerics — so ``NetworkLoss`` matches
    ``network_loss``, ``PodFailure`` matches ``pod_failure``, etc."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return _FAULT_DOC_ALIAS.get(key, key)


def _load_fault_doc(fault_kind: str) -> str:
    """Read the per-fault-kind reference doc, or return empty.

    chaos_type arrives CamelCase (``NetworkLoss``) while the docs are
    snake_case (``network_loss.md``); match on a normalized name so the
    reference actually loads instead of silently missing.
    """
    if not fault_kind:
        return ""
    p = FAULT_KINDS_DIR / f"{fault_kind}.md"
    if p.is_file():
        return p.read_text().strip()
    target = _norm_fault(fault_kind)
    for doc in FAULT_KINDS_DIR.glob("*.md"):
        if _norm_fault(doc.stem) == target:
            return doc.read_text().strip()
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
    "infra_dependency": "{frm} depends on the backing component {to} "
                        "(database/cache/broker). {to} is uninstrumented: it "
                        "has NO spans of its own — its calls live inside {frm}.",
}


def extract_hop_verdict(obs_dir: Path) -> dict | None:
    """Extract the last accepted HopVerdict via ``agentm trace tools``.

    Shells out to the CLI rather than sniffing raw JSONL, so only accepted
    tool *results* are considered (rejected tool-call arguments are ignored).
    """
    base = (
        ["agentm"]
        if shutil.which("agentm")
        else ["uv", "run", "--no-sync", "agentm"]
    )
    best: dict | None = None
    for f in sorted(obs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        cmd = [
            *base, "trace", "tools",
            "--file", str(f),
            "--tool", "submit_hop_verdict",
            "--format", "ndjson",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
        except Exception:  # noqa: BLE001
            continue
        if proc.returncode != 0:
            continue
        for line in proc.stdout.splitlines():
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            result = row.get("result")
            if not result:
                continue
            # result is either a raw string or a ToolResult dict
            # {"content": [{"type":"text","text":"..."}], "is_error": ...}
            if isinstance(result, dict):
                if result.get("is_error"):
                    continue
                content = result.get("content", [])
                if content and isinstance(content[0], dict):
                    result = content[0].get("text", "")
                else:
                    continue
            if not isinstance(result, str) or not result:
                continue
            if '"error"' in result and '"verdict"' not in result:
                continue
            try:
                obj = json.loads(result)
            except Exception:  # noqa: BLE001
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
    is_infra: bool = False,
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
    if is_infra:
        parts.append(
            f"\n## {to_service} is an uninstrumented backing component\n"
            f"`{to_service}` has NO spans of its own — `service_name = "
            f"'{to_service}'` returns nothing in *_traces. Verify it via:\n"
            f"- (A) the Client DB/cache spans **inside {from_service}**: "
            f"`WHERE service_name = '{from_service}' AND "
            f"\"attr.span_kind\" = 'Client'` with SQL/ORM span_name shapes "
            f"(SELECT/INSERT/UPDATE/DELETE/Transaction/Session/%Repository%). "
            f"Compare normal vs abnormal latency and error rate.\n"
            f"- (B) `{to_service}`'s own resource metrics: `*_metrics` tables "
            f"`WHERE service_name = '{to_service}'`.\n"
            f"Judge by fault type: a JVM/JDBC fault leaves the DB itself "
            f"healthy (the wait is in {from_service}'s client code — do NOT "
            f"count {to_service} as degraded); a Network fault "
            f"(loss/partition/delay) on the {from_service}↔{to_service} "
            f"path genuinely breaks the dependency (DB-call spans "
            f"error/time out — count it degraded)."
        )
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
        "--model", env.get("AGENTM_MODEL", "doubao"),
        "--cwd", str(hop_dir),
        "--max-tool-calls", str(budget),
        "-p", prompt,
    ]

    with open(hop_dir / "stdout.log", "w") as fout, \
         open(hop_dir / "stderr.log", "w") as ferr:
        subprocess.run(
            cmd, env=env, stdout=fout, stderr=ferr,
        )

    obs_dir = hop_dir / ".agentm" / "observability"
    verdict = extract_hop_verdict(obs_dir) if obs_dir.exists() else None
    if verdict:
        (hop_dir / "verdict.json").write_text(
            json.dumps(verdict, ensure_ascii=False, indent=2)
        )
    return verdict


# ------------------------------------------------------------------
# Edge verification (SQL-only, no agent)
# ------------------------------------------------------------------

def _verify_edge_sql(data_dir: Path, from_svc: str, to_svc: str) -> bool:
    """Check that a direct relationship exists between the two services."""
    conn = _duckdb_conn(data_dir)
    row = conn.execute(
        "SELECT COUNT(*) FROM normal_traces child "
        "JOIN normal_traces parent "
        "  ON child.parent_span_id = parent.span_id "
        "WHERE (parent.service_name = ? "
        "   AND child.service_name = ?) "
        "   OR (parent.service_name = ? "
        "   AND child.service_name = ?)",
        [to_svc, from_svc, from_svc, to_svc],
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
    infra_nodes: set[str] | None = None,
    node_map: dict[str, str] | None = None,
) -> dict:
    """Producer-consumer fault propagation.

    Every confirmed node is a *producer*: it enqueues ALL its
    neighbours (callers, callees, co-deployed). For each neighbour:
      - already confirmed → SQL-verify the edge (no agent)
      - not yet confirmed → run a hop agent (may confirm the node)
    No edge is skipped; no node agent runs twice.
    """
    infra_nodes = infra_nodes or set()
    confirmed: set[str] = set()
    agent_checked: set[str] = set()
    checked_edges: set[tuple[str, str]] = set()
    queue: deque[str] = deque()
    edges: list[tuple[str, str]] = []
    node_evidence: dict[str, dict] = {}
    hop_log: list[dict] = []

    fault_kind = injections[0]["chaos_type"] if injections else "unknown"
    inj_target = injections[0]["target"] if injections else "unknown"

    # Per-branch fault context: each node carries the (fault_kind, target)
    # of the seed it descends from, so a hop reached from the NetworkLoss
    # seed is judged as NetworkLoss — not as injections[0]'s fault. Without
    # this, every hop in a multi-injection case is told the wrong fault.
    node_fault: dict[str, tuple[str, str]] = {}

    for inj in injections:
        target = inj["target"]
        if target and target not in SYNTHETIC:
            ev = get_target_evidence(data_dir, target)
            confirmed.add(target)
            agent_checked.add(target)
            queue.append(target)
            node_evidence[target] = {"source": "injection_target", **ev}
            node_fault[target] = (inj["chaos_type"], target)

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

        # co_deployed pre-check: skip neighbours on healthy nodes.
        # Group co_deployed targets by shared node, check node health once.
        if node_map:
            _checked_nodes: dict[str, bool] = {}
            skip_co: set[str] = set()
            for to_svc, froms in agent_hops.items():
                if all(rel == "co_deployed" for _, rel in froms):
                    node = node_map.get(to_svc)
                    if node:
                        if node not in _checked_nodes:
                            _checked_nodes[node] = check_node_degraded(
                                data_dir, node,
                            )
                        if not _checked_nodes[node]:
                            skip_co.add(to_svc)
            if skip_co:
                for svc in skip_co:
                    agent_checked.add(svc)
                    hop_log.append({
                        "round": round_n, "from": "node_health",
                        "to": svc, "verdict": "co_deployed_node_healthy",
                    })
                    print(f"    skip co_deployed {svc}: node healthy")
                agent_hops = {
                    k: v for k, v in agent_hops.items()
                    if k not in skip_co
                }

        need_agent = {
            to: froms for to, froms in agent_hops.items()
            if to not in agent_checked
        }

        if need_agent:
            print(f"  round {round_n}: {len(need_agent)} agents, "
                  f"{len(sql_edges)} SQL edges")
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = {}
                for to_svc, froms in need_agent.items():
                    hop_fk, hop_target = node_fault.get(
                        froms[0][0], (fault_kind, inj_target)
                    )
                    hop_doc = _load_fault_doc(hop_fk) or fault_doc
                    fut = pool.submit(
                        run_hop, data_dir,
                        froms[0][0], to_svc, froms[0][1],
                        hop_fk, hop_doc, hop_target,
                        out_dir, budget,
                        to_svc in infra_nodes,
                    )
                    futures[fut] = (to_svc, froms)
                for future in as_completed(futures):
                    to_svc, froms = futures[future]
                    agent_checked.add(to_svc)
                    result = future.result()
                    verdict = result.get("verdict") if result else None
                    hop_log.append({
                        "round": round_n, "from": froms[0][0],
                        "to": to_svc, "verdict": verdict or "no-result",
                    })
                    print(f"    {froms[0][0]} -> {to_svc}: "
                          f"{verdict or 'no-result'}")
                    if verdict == "confirmed":
                        confirmed.add(to_svc)
                        node_evidence[to_svc] = {
                            "source": "hop_agent", **(result or {}),
                        }
                        # inherit the producer's fault context for the next hop
                        if froms[0][0] in node_fault:
                            node_fault[to_svc] = node_fault[froms[0][0]]
                        # Infra nodes (mysql, redis…) are propagation SINKS: a
                        # localized fault on ONE service's path to the DB does
                        # not degrade every other consumer of that DB. Confirm
                        # the edge into it, but do not fan out from it (which
                        # would falsely implicate all DB users via starvation).
                        if to_svc not in infra_nodes:
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

def collect_all_verdicts(out_dir: Path) -> list[dict]:
    """Collect all hop verdicts (confirmed + rejected) from a case run.

    Walks the hops/ directory and extracts each hop's structured verdict
    from verdict.json (new runs) or the observability JSONL (legacy runs).
    Returns a list of dicts with from/to/verdict/rationale/claim.
    """
    hops_dir = out_dir / "hops"
    if not hops_dir.exists():
        return []

    verdicts: list[dict] = []
    for hop_dir in sorted(hops_dir.iterdir()):
        if not hop_dir.is_dir():
            continue
        parts = hop_dir.name.split("__", 1)
        if len(parts) != 2:
            continue
        from_svc, to_svc = parts

        verdict_data: dict | None = None
        vf = hop_dir / "verdict.json"
        if vf.exists():
            try:
                verdict_data = json.loads(vf.read_text())
            except Exception:  # noqa: BLE001
                pass

        if not verdict_data:
            obs_dir = hop_dir / ".agentm" / "observability"
            if obs_dir.exists():
                verdict_data = extract_hop_verdict(obs_dir)
                if verdict_data:
                    vf.write_text(json.dumps(
                        verdict_data, ensure_ascii=False, indent=2,
                    ))

        if verdict_data:
            verdicts.append({
                "from": from_svc,
                "to": to_svc,
                "verdict": verdict_data.get("verdict", "unknown"),
                "rationale": verdict_data.get("rationale", ""),
                "claim": verdict_data.get("claim", ""),
                "symptom_evidence": verdict_data.get("symptom_evidence", []),
            })
    return verdicts


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


def run_one_case(
    case_dir: Path,
    out_dir: Path,
    *,
    budget: int = 15,
    parallel: int = 4,
) -> dict:
    """Run propagation verification on a single case.

    Returns a summary dict with keys: case, seeds, confirmed, edges,
    rounds, error.  The full report and trace are written to *out_dir*.
    """
    data_dir = case_dir.resolve()
    out = out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    case_name = data_dir.name

    injections = get_injections(data_dir)
    if not injections:
        return {"case": case_name, "error": "no injections"}

    fault_kind = injections[0]["chaos_type"]
    fault_doc = _load_fault_doc(fault_kind)

    rels = get_relationships(data_dir)
    infra_nodes = get_infra_nodes(data_dir)
    rels.extend(get_infra_edges(data_dir, infra_nodes))
    neighbor_graph = _build_neighbor_graph(rels)
    node_map = get_node_map(data_dir)
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
    print(f"Infra nodes (metrics-only): {sorted(infra_nodes)}")

    result = propagate(
        data_dir, injections, neighbor_graph, fault_doc, out,
        budget=budget, parallel=parallel,
        infra_nodes=infra_nodes, node_map=node_map,
    )
    (out / "propagation_trace.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str)
    )

    report = build_report(result, injections)
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    all_verdicts = collect_all_verdicts(out)
    (out / "all_verdicts.json").write_text(
        json.dumps(all_verdicts, indent=2, ensure_ascii=False)
    )

    seeds = [i["target"] for i in injections]
    confirmed = result["confirmed_nodes"]
    propagated = [s for s in confirmed if s not in seeds]
    edge_count = len(result["edges"])

    print(f"\nRounds: {result['rounds']}")
    print(f"Confirmed: {confirmed}")
    print(f"Edges ({edge_count}): {result['edges']}")

    return {
        "case": case_name,
        "fault_kind": fault_kind,
        "seeds": seeds,
        "confirmed": confirmed,
        "propagated": propagated,
        "edges": edge_count,
        "rounds": result["rounds"],
    }


# ------------------------------------------------------------------
# Batch runner
# ------------------------------------------------------------------

def _read_cached_summary(out_dir: Path, case_name: str) -> dict | None:
    report_path = out_dir / "report.json"
    trace_path = out_dir / "propagation_trace.json"
    if not report_path.exists() or not trace_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text())
        trace = json.loads(trace_path.read_text())
    except Exception:  # noqa: BLE001
        return None
    seeds = [i["target_service"] for i in report.get("injections", [])]
    confirmed = trace.get("confirmed_nodes", [])
    return {
        "case": case_name,
        "fault_kind": report["injections"][0]["fault_kind"]
        if report.get("injections") else "unknown",
        "seeds": seeds,
        "confirmed": confirmed,
        "propagated": [s for s in confirmed if s not in seeds],
        "edges": len(report.get("propagation_edges", [])),
        "rounds": trace.get("rounds", 0),
        "cached": True,
    }


def _run_or_cache(
    dataset_dir: Path,
    run_dir: Path,
    name: str,
    idx: int,
    total: int,
    budget: int,
    parallel: int,
) -> dict:
    """Run one case or return cached result.  Thread-safe."""
    case_out = run_dir / name
    existing = _read_cached_summary(case_out, name)
    if existing:
        prop_str = (f"propagated={existing['propagated']}"
                    if existing["propagated"] else "no propagation")
        print(f"[{idx}/{total}] {name} CACHED: {prop_str}", flush=True)
        return existing

    print(f"[{idx}/{total}] {name} ...", flush=True)
    try:
        summary = run_one_case(
            dataset_dir / name, case_out,
            budget=budget, parallel=parallel,
        )
        if "error" in summary:
            print(f"  [{name}] ERROR: {summary['error']}", flush=True)
        else:
            prop_str = (f"propagated={summary['propagated']}"
                        if summary.get("propagated") else "no propagation")
            print(f"  [{name}] OK: {prop_str}", flush=True)
        return summary
    except Exception as exc:  # noqa: BLE001
        print(f"  [{name}] EXCEPTION: {exc}", flush=True)
        return {"case": name, "error": str(exc)}


def run_batch(
    dataset_dir: Path,
    run_dir: Path,
    *,
    budget: int = 15,
    parallel: int = 4,
    case_parallel: int = 1,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    """Run propagation on multiple cases under *dataset_dir*.

    *case_parallel* controls how many cases run concurrently (each case
    uses up to *parallel* hop agents internally).  A
    ``run_summary.jsonl`` in *run_dir* holds one JSON line per case,
    suitable for diff/analysis across ablation runs.
    """
    cases = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    cases = cases[offset:]
    if limit is not None:
        cases = cases[:limit]

    total = len(cases)
    run_dir.mkdir(parents=True, exist_ok=True)

    if case_parallel <= 1:
        summaries = [
            _run_or_cache(dataset_dir, run_dir, name, i, total, budget, parallel)
            for i, name in enumerate(cases, 1)
        ]
    else:
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=case_parallel) as pool:
            futures = {
                pool.submit(
                    _run_or_cache,
                    dataset_dir, run_dir, name, i, total, budget, parallel,
                ): name
                for i, name in enumerate(cases, 1)
            }
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
        summaries = [results[name] for name in cases]

    ok = sum(1 for s in summaries if "error" not in s)
    fail = sum(1 for s in summaries if "error" in s)
    cached = sum(1 for s in summaries if s.get("cached"))

    summary_path = run_dir / "run_summary.jsonl"
    with open(summary_path, "w") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"Total: {total}  OK: {ok}  Failed: {fail}  Cached: {cached}")
    prop_count = sum(1 for s in summaries if s.get("propagated"))
    print(f"Cases with propagation: {prop_count}/{ok}")
    print(f"Summary: {summary_path}")

    return summaries


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

app = typer.Typer(
    name="audit-propagate",
    help=__doc__,
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _set_model(model: str | None) -> None:
    if model:
        os.environ["AGENTM_MODEL"] = model


@app.command()
def run(
    case_dir: Annotated[Path, typer.Argument(help="single case directory")],
    out: Annotated[Path | None, typer.Option(help="output directory")] = None,
    budget: Annotated[int, typer.Option(help="tool-call budget per hop agent")] = 15,
    parallel: Annotated[int, typer.Option(help="concurrent hop agents")] = 4,
    model: Annotated[str | None, typer.Option(help="config.toml profile name")] = None,
) -> None:
    """Run propagation check on a single case."""
    _set_model(model)
    out_dir = out or case_dir.resolve() / ".verify_propagate"
    summary = run_one_case(case_dir, out_dir, budget=budget, parallel=parallel)
    if "error" in summary:
        raise typer.Exit(1)


@app.command()
def batch(
    dataset_dir: Annotated[Path, typer.Argument(help="directory containing case subdirectories")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="output directory for this run")],
    budget: Annotated[int, typer.Option(help="tool-call budget per hop agent")] = 15,
    parallel: Annotated[int, typer.Option(help="concurrent hop agents per case")] = 4,
    case_parallel: Annotated[int, typer.Option("--case-parallel", help="concurrent cases")] = 1,
    model: Annotated[str | None, typer.Option(help="config.toml profile name")] = None,
    limit: Annotated[int | None, typer.Option(help="max cases to run")] = None,
    offset: Annotated[int, typer.Option(help="skip first N cases")] = 0,
) -> None:
    """Run propagation on all cases in a dataset."""
    _set_model(model)
    run_batch(
        dataset_dir.resolve(), run_dir.resolve(),
        budget=budget, parallel=parallel,
        case_parallel=case_parallel,
        limit=limit, offset=offset,
    )


def _get_system_throughput(data_dir: Path) -> dict:
    """Compare load-generator root span counts between windows."""
    conn = _duckdb_conn(data_dir)
    result: dict = {}
    try:
        for w in ("normal", "abnormal"):
            row = conn.execute(
                f"SELECT COUNT(*) FROM {w}_traces "
                "WHERE (parent_span_id = '' OR parent_span_id IS NULL) "
                f"AND service_name IN {str(tuple(SYNTHETIC))}"
            ).fetchone()
            result[w] = row[0] if row else 0
    except Exception:  # noqa: BLE001
        pass
    conn.close()
    return result


def run_judge(
    case_dir: Path,
    run_dir: Path,
    *,
    budget: int = 20,
) -> dict:
    """Run a judge agent on a completed case to review all hop verdicts."""
    data_dir = case_dir.resolve()
    out = run_dir.resolve()

    # Collect inputs
    all_verdicts_path = out / "all_verdicts.json"
    if not all_verdicts_path.exists():
        all_verdicts = collect_all_verdicts(out)
        all_verdicts_path.write_text(
            json.dumps(all_verdicts, indent=2, ensure_ascii=False)
        )
    else:
        all_verdicts = json.loads(all_verdicts_path.read_text())

    trace = json.loads((out / "propagation_trace.json").read_text())
    confirmed = trace.get("confirmed_nodes", [])
    injections = get_injections(data_dir)
    throughput = _get_system_throughput(data_dir)

    # Build the judge prompt
    inj_lines = [
        f"- {i['target']} ({i['chaos_type']})" for i in injections
    ]
    confirmed_str = ", ".join(confirmed) if confirmed else "(none)"

    tp_normal = throughput.get("normal", 0)
    tp_abnormal = throughput.get("abnormal", 0)
    tp_drop = ((tp_normal - tp_abnormal) / tp_normal * 100
               if tp_normal > 0 else 0)

    rejected_entries = [v for v in all_verdicts if v["verdict"] == "rejected"]
    confirmed_entries = [v for v in all_verdicts if v["verdict"] == "confirmed"]

    verdict_lines: list[str] = []
    for v in all_verdicts:
        tag = "CONFIRMED" if v["verdict"] == "confirmed" else "REJECTED"
        verdict_lines.append(
            f"- [{tag}] {v['from']} → {v['to']}: {v['rationale']}"
        )

    prompt = f"""\
You are a judge reviewing the results of a fault-propagation verifier.
Individual hop agents each checked one edge in isolation — your job is to
look at the FULL picture and decide whether any verdicts should be
overturned.

## Fault injection
{chr(10).join(inj_lines)}

## System-wide throughput
- Normal window: {tp_normal} load-generator root spans
- Abnormal window: {tp_abnormal} load-generator root spans
- Drop: {tp_drop:.1f}%

## Current confirmed propagation
{confirmed_str}

## All hop verdicts ({len(confirmed_entries)} confirmed, {len(rejected_entries)} rejected)
{chr(10).join(verdict_lines)}

## Your task
1. Check causal consistency: does the propagation chain make sense for
   this fault type?  Are there contradictions between hop verdicts?
2. Check for system-wide cascade: if load-generator throughput dropped
   >80%, the system is in cascading failure.  In that regime,
   "throughput drop alone" is NOT a valid rejection reason — the
   service is unavailable because the whole system is collapsing.
   Flag any rejected hops that used this reasoning during a cascade.
3. Check cross-hop consistency: if service A calls B and C, and B is
   confirmed degraded but C is rejected despite similar evidence, flag
   the inconsistency.
4. For each flagged verdict, explain why it should be overturned and
   what evidence supports the override.

You also have access to `list_tables` and `query_sql` to independently
verify any evidence you question.

When done, call `submit_hop_verdict` with:
- verdict: "confirmed" if you found overrides, "rejected" if all
  original verdicts look correct
- rationale: your overall assessment
- claim: list of services whose verdicts should be overturned
  (comma-separated), or "none"
"""

    judge_dir = out / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)

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
        "--model", env.get("AGENTM_MODEL", "doubao"),
        "--cwd", str(judge_dir),
        "--max-tool-calls", str(budget),
        "-p", prompt,
    ]

    with open(judge_dir / "stdout.log", "w") as fout, \
         open(judge_dir / "stderr.log", "w") as ferr:
        subprocess.run(cmd, env=env, stdout=fout, stderr=ferr)

    # Extract judge verdict
    obs_dir = judge_dir / ".agentm" / "observability"
    verdict = extract_hop_verdict(obs_dir) if obs_dir.exists() else None
    if verdict:
        (judge_dir / "verdict.json").write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False)
        )

    # Merge judge overrides into a final propagation result
    overrides: set[str] = set()
    if verdict and verdict.get("verdict") == "confirmed":
        claim = verdict.get("claim", "")
        overrides = {
            s.strip() for s in claim.split(",")
            if s.strip() and s.strip() != "none"
        }

    seeds = {i["target"] for i in injections}
    final_confirmed = sorted(set(confirmed) | overrides)
    final_propagated = [s for s in final_confirmed if s not in seeds]

    # Index hop verdicts by target service for evidence lookup
    verdict_by_target: dict[str, dict] = {}
    for v in all_verdicts:
        verdict_by_target[v["to"]] = v

    # Build per-node entry with provenance + evidence
    nodes: list[dict] = []
    for svc in final_confirmed:
        if svc in seeds:
            nodes.append({
                "service": svc,
                "source": "injection_seed",
            })
        elif svc in overrides and svc not in confirmed:
            hop_v = verdict_by_target.get(svc, {})
            nodes.append({
                "service": svc,
                "source": "judge_override",
                "hop_verdict": hop_v.get("verdict", ""),
                "hop_rationale": hop_v.get("rationale", ""),
                "hop_evidence": hop_v.get("symptom_evidence", []),
                "judge_rationale": verdict.get("rationale", "")
                if verdict else "",
            })
        else:
            hop_v = verdict_by_target.get(svc, {})
            nodes.append({
                "service": svc,
                "source": "hop_agent",
                "rationale": hop_v.get("rationale", ""),
                "symptom_evidence": hop_v.get("symptom_evidence", []),
            })

    final = {
        "seeds": sorted(seeds),
        "confirmed_nodes": final_confirmed,
        "propagated": final_propagated,
        "nodes": nodes,
        "judge_verdict": verdict.get("verdict") if verdict else None,
        "judge_rationale": verdict.get("rationale", "") if verdict else "",
        "overrides": sorted(overrides - set(confirmed) - seeds),
    }
    (out / "final_propagation.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False)
    )

    return verdict or {}


@app.command()
def judge(
    case_dir: Annotated[Path, typer.Argument(help="single case directory")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="verifier run output for this case")],
    budget: Annotated[int, typer.Option(help="tool-call budget for judge agent")] = 20,
    model: Annotated[str | None, typer.Option(help="config.toml profile name")] = None,
) -> None:
    """Run judge agent on a completed case to review hop verdicts."""
    _set_model(model)
    result = run_judge(case_dir, run_dir, budget=budget)
    if result:
        print(f"\nJudge verdict: {result.get('verdict', '?')}")
        print(f"Rationale: {result.get('rationale', '?')}")
    else:
        print("Judge produced no verdict.")

    final_path = run_dir / "final_propagation.json"
    if final_path.exists():
        final = json.loads(final_path.read_text())
        overrides = final.get("overrides", [])
        print(f"\nFinal propagation: {len(final['confirmed_nodes'])} services "
              f"({len(overrides)} judge overrides)")
        print(f"Output: {final_path}")


def _gt_services(case_dir: Path) -> tuple[set[str], set[str]]:
    """Extract GT injection seeds and propagated services.

    Seeds come from injection.json ``engine_config``.  Propagated
    services come from causal_graph.json: fold the span-level graph to
    service granularity via ``component_to_service``, then subtract
    the seeds.
    """
    inj_path = case_dir / "injection.json"
    cg_path = case_dir / "causal_graph.json"
    if not inj_path.exists():
        return set(), set()

    inj = json.loads(inj_path.read_text())
    seeds: set[str] = set()
    raw_ec = inj.get("engine_config", [])
    if isinstance(raw_ec, list):
        for item in raw_ec:
            if isinstance(item, dict) and item.get("app"):
                seeds.add(item["app"])
    if not seeds:
        for item in inj.get("engine_config_summary", []):
            if isinstance(item, dict) and item.get("app"):
                seeds.add(item["app"])
    if not seeds:
        gt = inj.get("ground_truth")
        if isinstance(gt, dict):
            for s in gt.get("service", []):
                seeds.add(s)
        elif isinstance(gt, list):
            for g in gt:
                if isinstance(g, dict):
                    for s in g.get("service", []):
                        seeds.add(s)

    if not cg_path.exists():
        return seeds, set()

    cg = json.loads(cg_path.read_text())
    c2s = cg.get("component_to_service", {})
    gt_all = {svc for svc in c2s.values() if svc}
    return seeds, gt_all - seeds


@app.command()
def diff(
    dataset_dir: Annotated[Path, typer.Argument(help="dataset with GT labels")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="verifier run output")],
) -> None:
    """Compare verifier findings against GT labels."""
    summary_path = run_dir / "run_summary.jsonl"
    if not summary_path.exists():
        typer.echo("No run_summary.jsonl found — run batch first.")
        raise typer.Exit(1)

    with open(summary_path) as f:
        cases = [json.loads(line) for line in f]

    total = new_total = missed_total = match_total = 0
    diff_rows: list[dict] = []

    for s in cases:
        name = s["case"]
        if "error" in s:
            continue
        total += 1
        case_dir = dataset_dir / name
        gt_inj, gt_prop = _gt_services(case_dir)
        v_seeds = set(s.get("seeds", []))
        v_prop = set(s.get("propagated", []))

        new_finds = v_prop - gt_prop - v_seeds
        missed = gt_prop - v_prop

        row = {
            "case": name,
            "gt_inj": sorted(gt_inj), "gt_prop": sorted(gt_prop),
            "v_seeds": sorted(v_seeds), "v_prop": sorted(v_prop),
            "new": sorted(new_finds), "missed": sorted(missed),
        }
        diff_rows.append(row)

        tag = "MATCH" if (not new_finds and not missed) else ""
        if new_finds:
            new_total += len(new_finds)
            tag = f"NEW +{len(new_finds)}"
        if missed:
            missed_total += len(missed)
            tag += f"  MISSED -{len(missed)}"
        if not new_finds and not missed:
            match_total += 1

        typer.echo(f"{name}  {tag}")
        if new_finds:
            typer.echo(f"  new:    {sorted(new_finds)}")
        if missed:
            typer.echo(f"  missed: {sorted(missed)}")

    typer.echo(f"\n{'='*50}")
    typer.echo(f"Cases: {total}  Match: {match_total}  "
               f"New findings: {new_total}  Missed: {missed_total}")

    diff_path = run_dir / "gt_diff.jsonl"
    with open(diff_path, "w") as f:
        for row in diff_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    typer.echo(f"Diff: {diff_path}")


if __name__ == "__main__":
    app()
