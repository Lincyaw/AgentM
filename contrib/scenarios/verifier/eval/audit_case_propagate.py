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
    "callee_to_caller": "{to} calls {frm}, so {frm} is {to}'s downstream "
                        "dependency. A degraded callee propagates UP to its "
                        "caller {to}, which blocks on or fails with the bad "
                        "response. This is the usual direction for latency "
                        "and error faults.",
    "caller_to_callee": "{frm} calls {to}, so {to} is {frm}'s downstream "
                        "dependency. A caller affects its callee ONLY for "
                        "data-corruption / bad-request faults (it sends {to} "
                        "a wrong or corrupted request). A merely slow or "
                        "failing caller does NOT by itself degrade {to} — be "
                        "skeptical of confirming on this edge.",
    "co_deployed": "{frm} and {to} share a k8s node — ONLY a node-level "
                   "resource fault (CPU/memory/disk exhaustion) on one can "
                   "degrade the other. An app-logic, JVM, or network fault "
                   "does not cross to a co-located pod.",
    "infra_dependency": "{frm} depends on the backing component {to} "
                        "(database/cache/broker). {to} is uninstrumented: it "
                        "has NO spans of its own — its calls live inside {frm}.",
}


def extract_hop_verdict(
    obs_dir: Path,
    tool: str = "submit_hop_verdict",
    require_key: str = "verdict",
) -> dict | None:
    """Extract the last accepted tool result via ``agentm trace tools``.

    Shells out to the CLI rather than sniffing raw JSONL, so only accepted
    tool *results* are considered (rejected tool-call arguments are ignored).
    ``tool``/``require_key`` let the same path read the judge review tool
    (``submit_judge_review`` keyed on ``remove``).
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
            "--tool", tool,
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
            if '"error"' in result and f'"{require_key}"' not in result:
                continue
            try:
                obj = json.loads(result)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict) and require_key in obj:
                best = obj
    return best


def _format_upstream_evidence(evidence: dict) -> str:
    """Format upstream node evidence for the hop agent prompt."""
    lines: list[str] = []
    src = evidence.get("source", "")
    if src == "injection_target":
        n_ms = evidence.get("normal_avg_ms")
        a_ms = evidence.get("abnormal_avg_ms")
        ratio = evidence.get("ratio")
        if n_ms is not None and a_ms is not None:
            lines.append(
                f"Avg latency: normal {n_ms:.1f}ms → abnormal {a_ms:.1f}ms "
                f"({ratio}x)"
            )
    elif src == "hop_agent":
        rationale = evidence.get("rationale")
        if rationale:
            lines.append(f"Rationale: {rationale}")
        for ev in evidence.get("symptom_evidence", []):
            claim = ev.get("claim", "")
            sql = ev.get("sql", "")
            if claim:
                lines.append(f"- {claim}")
            if sql:
                lines.append(f"  ```sql\n  {sql}\n  ```")
    return "\n".join(lines)


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
    upstream_evidence: dict | None = None,
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
    if upstream_evidence:
        ev_text = _format_upstream_evidence(upstream_evidence)
        if ev_text:
            parts.append(
                f"\n## Observed symptoms on {from_service}\n{ev_text}\n\n"
                f"This is only a partial picture of the upstream's "
                f"degradation. Look for **different signals** on "
                f"{to_service} — do not just repeat the same queries. "
                f"The propagation may manifest differently on the "
                f"downstream (e.g. errors vs latency vs missing spans)."
            )
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
            f"The component is degraded ONLY if (B) its own metrics worsen, "
            f"or its DB/cache spans error/slow across MULTIPLE independent "
            f"callers. A single caller's slow or failing client spans is that "
            f"caller's egress problem — especially under a fault that lives on "
            f"`{from_service}` (a JVM/JDBC fault, or a `tc netem` delay/loss "
            f"that slows ALL of {from_service}'s packets). Do NOT count "
            f"`{to_service}` degraded from `{from_service}`'s client spans "
            f"alone — that double-counts {from_service}'s own degradation."
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
    """Edge-level fault propagation.

    Every confirmed node enqueues ALL its neighbours. Each directed edge
    (from, to) is independently evaluated — a service rejected via one
    relationship (e.g. co_deployed) can still be confirmed via another
    (e.g. caller_to_callee) from a different source in a later round.
    """
    infra_nodes = infra_nodes or set()
    confirmed: set[str] = set()
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
            queue.append(target)
            node_evidence[target] = {"source": "injection_target", **ev}
            node_fault[target] = (inj["chaos_type"], target)

    round_n = 0
    while queue:
        round_n += 1
        level_size = len(queue)

        sql_edges: list[tuple[str, str]] = []
        # Each entry is one edge to evaluate: (from_service, to_service, rel_type)
        pending_hops: list[tuple[str, str, str]] = []

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
                    pending_hops.append((current, neighbor, rel_type))

        for from_svc, to_svc in sql_edges:
            if _verify_edge_sql(data_dir, from_svc, to_svc):
                edges.append((from_svc, to_svc))
                hop_log.append({
                    "round": round_n, "from": from_svc, "to": to_svc,
                    "verdict": "edge_sql",
                })
                print(f"  edge (SQL): {from_svc} -> {to_svc}")

        # co_deployed pre-check: skip edges to services on healthy nodes
        # when ALL edges to that target are co_deployed.
        if node_map:
            _checked_nodes: dict[str, bool] = {}
            # Group by target to check if ALL edges are co_deployed
            target_rels: dict[str, list[tuple[str, str, str]]] = {}
            for from_svc, to_svc, rel in pending_hops:
                target_rels.setdefault(to_svc, []).append(
                    (from_svc, to_svc, rel)
                )
            skip_edges: set[tuple[str, str]] = set()
            for to_svc, hop_list in target_rels.items():
                if all(rel == "co_deployed" for _, _, rel in hop_list):
                    node = node_map.get(to_svc)
                    if node:
                        if node not in _checked_nodes:
                            _checked_nodes[node] = check_node_degraded(
                                data_dir, node,
                            )
                        if not _checked_nodes[node]:
                            for from_svc, _, _ in hop_list:
                                skip_edges.add((from_svc, to_svc))
                            hop_log.append({
                                "round": round_n, "from": "node_health",
                                "to": to_svc,
                                "verdict": "co_deployed_node_healthy",
                            })
                            print(f"    skip co_deployed {to_svc}: "
                                  f"node healthy")
            pending_hops = [
                h for h in pending_hops
                if (h[0], h[1]) not in skip_edges
            ]

        # Filter out edges whose target was confirmed earlier in this round
        # (by a parallel agent or a previous edge in this batch).
        pending_hops = [
            h for h in pending_hops if h[1] not in confirmed
        ]

        if pending_hops:
            print(f"  round {round_n}: {len(pending_hops)} agents, "
                  f"{len(sql_edges)} SQL edges")
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = {}
                for from_svc, to_svc, rel_type in pending_hops:
                    hop_fk, hop_target = node_fault.get(
                        from_svc, (fault_kind, inj_target)
                    )
                    hop_doc = _load_fault_doc(hop_fk) or fault_doc
                    fut = pool.submit(
                        run_hop, data_dir,
                        from_svc, to_svc, rel_type,
                        hop_fk, hop_doc, hop_target,
                        out_dir, budget,
                        to_svc in infra_nodes,
                        node_evidence.get(from_svc),
                    )
                    futures[fut] = (from_svc, to_svc, rel_type)
                for future in as_completed(futures):
                    from_svc, to_svc, rel_type = futures[future]
                    result = future.result()
                    verdict = result.get("verdict") if result else None
                    hop_log.append({
                        "round": round_n, "from": from_svc,
                        "to": to_svc, "verdict": verdict or "no-result",
                    })
                    print(f"    {from_svc} -> {to_svc}: "
                          f"{verdict or 'no-result'}")
                    if verdict == "confirmed" and to_svc not in confirmed:
                        confirmed.add(to_svc)
                        node_evidence[to_svc] = {
                            "source": "hop_agent", **(result or {}),
                        }
                        if from_svc in node_fault:
                            node_fault[to_svc] = node_fault[from_svc]
                        if to_svc not in infra_nodes:
                            queue.append(to_svc)
                        edges.append((from_svc, to_svc))

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
    seeds = {i["target"] for i in injections}
    inj_lines = [
        f"- {i['target']} ({i['chaos_type']})" for i in injections
    ]

    tp_normal = throughput.get("normal", 0)
    tp_abnormal = throughput.get("abnormal", 0)
    tp_drop = ((tp_normal - tp_abnormal) / tp_normal * 100
               if tp_normal > 0 else 0)

    # Index hop verdicts by target for evidence lookup.
    verdict_by_target: dict[str, dict] = {v["to"]: v for v in all_verdicts}

    def _ev_claims(svc: str) -> str:
        v = verdict_by_target.get(svc, {})
        claims = [
            e.get("claim", "") for e in v.get("symptom_evidence", [])
            if e.get("claim")
        ]
        return "; ".join(claims[:4])

    confirmed_nonseed = [s for s in confirmed if s not in seeds]
    confirmed_lines: list[str] = []
    for s in confirmed_nonseed:
        v = verdict_by_target.get(s, {})
        frm = v.get("from", "?")
        confirmed_lines.append(
            f"- {frm} → **{s}**: {v.get('rationale', '(no rationale)')}\n"
            f"    evidence: {_ev_claims(s) or '(none)'}"
        )
    confirmed_block = "\n".join(confirmed_lines) or "(none)"

    rejected_lines: list[str] = []
    for v in all_verdicts:
        if v["verdict"] == "rejected" and v["to"] not in confirmed:
            rejected_lines.append(
                f"- {v['from']} → {v['to']}: {v['rationale']}"
            )
    rejected_block = "\n".join(rejected_lines) or "(none)"

    prompt = f"""\
You are the lead auditor of a fault-propagation graph that independent
hop agents built one edge at a time. Each hop agent ran careful
per-edge analysis — checking error rate, latency magnitude, fault
mechanism, and relationship direction — and already rejected
throughput-only drops and noise. **Their confirmations are
authoritative; you do NOT remove them.**

Your ONLY job is to catch what no single edge could see: a system-wide
CASCADE in which services the hop agents rejected for "fewer calls /
throughput drop" are in fact genuinely unavailable because the whole
system is collapsing.

## Fault injection
{chr(10).join(inj_lines)}

## System-wide load (the cascade signal)
- load-generator root spans: normal {tp_normal} → abnormal {tp_abnormal} (drop {tp_drop:.1f}%)
- If drop > 80%: the system is in cascading collapse. Review the
  rejected list and ADD any service that is down because the whole
  system is down (was actively serving, now silent/erroring).
- If drop <= 80%: there is NO cascade. ADD nothing — a throughput drop
  is just the caller sending fewer requests.

## Confirmed services (context — do NOT change these) ({len(confirmed_nonseed)})
{confirmed_block}

## Rejected services — ADD only under a real cascade ({len(rejected_lines)})
{rejected_block}

## Decide
- Leave `remove` EMPTY. The per-edge analysis is authoritative for
  what is degraded; second-guessing it from rationale text alone
  removes genuinely-degraded services and corrupts the graph.
- ADD a rejected service only if a system-wide cascade (loadgen drop
  > 80%) makes it genuinely unavailable, not merely less-called. Use
  `list_tables` / `query_sql` to confirm; state latencies in ms/s
  (duration is nanoseconds).

Most reviews add nothing. Call `submit_judge_review` with `add` (and
`remove` empty) plus `rationale`.
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

    # Extract judge review (bidirectional: remove confirmed, add rejected)
    obs_dir = judge_dir / ".agentm" / "observability"
    verdict = (
        extract_hop_verdict(obs_dir, "submit_judge_review", "remove")
        if obs_dir.exists() else None
    )
    if verdict:
        (judge_dir / "verdict.json").write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False)
        )

    confirmed_set = set(confirmed)
    # Judge is promotion-only: hop confirmations are authoritative. LLM
    # pruning from rationale text alone is unreliable (it removes genuine
    # SLOW/ERROR services), so we record any suggested removal for audit
    # but do NOT apply it.
    suggested_remove = sorted(({
        s.strip() for s in (verdict.get("remove", []) if verdict else [])
        if isinstance(s, str) and s.strip()
    } & confirmed_set) - seeds)
    removed: set[str] = set()
    # add: rejected services to promote, not already confirmed, not seeds.
    added = {
        s.strip() for s in (verdict.get("add", []) if verdict else [])
        if isinstance(s, str) and s.strip()
    }
    added -= confirmed_set
    added -= seeds

    final_confirmed = sorted((confirmed_set - removed) | added)
    final_propagated = [s for s in final_confirmed if s not in seeds]
    judge_rationale = verdict.get("rationale", "") if verdict else ""

    # Build per-node entry with provenance + evidence
    nodes: list[dict] = []
    for svc in final_confirmed:
        if svc in seeds:
            nodes.append({"service": svc, "source": "injection_seed"})
        elif svc in added:
            hop_v = verdict_by_target.get(svc, {})
            nodes.append({
                "service": svc,
                "source": "judge_promoted",
                "hop_verdict": hop_v.get("verdict", ""),
                "hop_rationale": hop_v.get("rationale", ""),
                "hop_evidence": hop_v.get("symptom_evidence", []),
                "judge_rationale": judge_rationale,
            })
        else:
            hop_v = verdict_by_target.get(svc, {})
            nodes.append({
                "service": svc,
                "source": "hop_agent",
                "rationale": hop_v.get("rationale", ""),
                "symptom_evidence": hop_v.get("symptom_evidence", []),
            })

    # Build edges, dropping any whose target the judge removed.
    final_set = set(final_confirmed)
    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()
    for h in trace.get("hop_log", []):
        frm, to = h.get("from", ""), h.get("to", "")
        if not frm or not to or to not in final_set:
            continue
        v = h.get("verdict", "")
        if v in ("confirmed", "edge_sql") and (frm, to) not in seen_edges:
            seen_edges.add((frm, to))
            hop_v = verdict_by_target.get(to, {})
            edge_entry: dict = {
                "from": frm, "to": to,
                "source": "hop_agent" if v == "confirmed" else "edge_sql",
            }
            if v == "confirmed":
                edge_entry["rationale"] = hop_v.get("rationale", "")
                edge_entry["symptom_evidence"] = hop_v.get(
                    "symptom_evidence", [],
                )
            edges.append(edge_entry)

    # Edges for judge-promoted nodes
    for svc in added:
        promoted_v = verdict_by_target.get(svc)
        if promoted_v and (promoted_v["from"], svc) not in seen_edges:
            seen_edges.add((promoted_v["from"], svc))
            edges.append({
                "from": promoted_v["from"], "to": svc,
                "source": "judge_promoted",
                "hop_rationale": promoted_v.get("rationale", ""),
                "hop_evidence": promoted_v.get("symptom_evidence", []),
            })

    final = {
        "seeds": sorted(seeds),
        "confirmed_nodes": final_confirmed,
        "propagated": final_propagated,
        "edges": edges,
        "nodes": nodes,
        "judge_rationale": judge_rationale,
        "removed": sorted(removed),
        "suggested_remove": suggested_remove,
        "added": sorted(added),
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
        print(f"\nJudge rationale: {result.get('rationale', '?')}")
    else:
        print("Judge produced no review.")

    final_path = run_dir / "final_propagation.json"
    if final_path.exists():
        final = json.loads(final_path.read_text())
        removed = final.get("removed", [])
        added = final.get("added", [])
        print(f"\nFinal propagation: {len(final['confirmed_nodes'])} services "
              f"(-{len(removed)} pruned, +{len(added)} promoted)")
        print(f"Output: {final_path}")


def _run_judge_or_skip(
    dataset_dir: Path,
    run_dir: Path,
    name: str,
    idx: int,
    total: int,
    budget: int,
) -> dict:
    """Run judge on one case, skip if final_propagation.json already exists."""
    case_out = run_dir / name
    final_path = case_out / "final_propagation.json"
    if final_path.exists():
        try:
            fp = json.loads(final_path.read_text())
            removed = fp.get("removed", [])
            added = fp.get("added", [])
            print(f"[{idx}/{total}] {name} CACHED: "
                  f"{len(fp['confirmed_nodes'])} confirmed "
                  f"(-{len(removed)}/+{len(added)})", flush=True)
            return {"case": name, "cached": True,
                    "confirmed": len(fp["confirmed_nodes"]),
                    "removed": len(removed), "added": len(added)}
        except Exception:  # noqa: BLE001
            pass

    trace_path = case_out / "propagation_trace.json"
    if not trace_path.exists():
        print(f"[{idx}/{total}] {name} SKIP: no hop results", flush=True)
        return {"case": name, "error": "no hop results"}

    print(f"[{idx}/{total}] {name} judging...", flush=True)
    try:
        run_judge(dataset_dir / name, case_out, budget=budget)
        fp = json.loads(final_path.read_text()) if final_path.exists() else {}
        removed = fp.get("removed", [])
        added = fp.get("added", [])
        print(f"  [{name}] {len(fp.get('confirmed_nodes', []))} confirmed "
              f"(-{len(removed)}/+{len(added)})", flush=True)
        return {"case": name,
                "confirmed": len(fp.get("confirmed_nodes", [])),
                "removed": len(removed), "added": len(added)}
    except Exception as exc:  # noqa: BLE001
        print(f"  [{name}] EXCEPTION: {exc}", flush=True)
        return {"case": name, "error": str(exc)}


@app.command(name="judge-batch")
def judge_batch(
    dataset_dir: Annotated[Path, typer.Argument(help="dataset directory")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="verifier run output")],
    budget: Annotated[int, typer.Option(help="tool-call budget per judge")] = 20,
    parallel: Annotated[int, typer.Option(help="concurrent judge agents")] = 10,
    model: Annotated[str | None, typer.Option(help="config.toml profile name")] = None,
    limit: Annotated[int | None, typer.Option(help="max cases")] = None,
) -> None:
    """Run judge on all completed cases in a batch run."""
    _set_model(model)
    cases = sorted(
        p.name for p in run_dir.iterdir()
        if p.is_dir() and (p / "propagation_trace.json").exists()
    )
    if limit:
        cases = cases[:limit]
    total = len(cases)
    print(f"Judging {total} cases (parallel={parallel})")

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(
                _run_judge_or_skip,
                dataset_dir.resolve(), run_dir.resolve(),
                name, i, total, budget,
            ): name
            for i, name in enumerate(cases, 1)
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()

    summaries = [results[name] for name in cases if name in results]
    ok = sum(1 for s in summaries if "error" not in s)
    cached = sum(1 for s in summaries if s.get("cached"))
    pruned = sum(s.get("removed", 0) for s in summaries)
    promoted = sum(s.get("added", 0) for s in summaries)

    print(f"\n{'='*50}")
    print(f"Total: {total}  OK: {ok}  Cached: {cached}  "
          f"Pruned: {pruned}  Promoted: {promoted}")


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


# ------------------------------------------------------------------
# Raw-data oracle — adjudicates verifier vs GT from traces/logs/metrics
#
# GT's `state` field is unreliable (it marks throughput-drops as
# degraded/unavailable), so neither GT nor GT-matching can score the
# verifier. This oracle classifies each service's abnormal-vs-normal
# signature directly and is used only as a measurement yardstick — it is
# NOT part of the verifier runtime (keeping the agent's reasoning and the
# soundness metric independent).
# ------------------------------------------------------------------

_ORACLE_NS_MS = 1e6
_SLOW_FLOOR = 50 * _ORACLE_NS_MS      # 50ms absolute p95 increase floor
_SLOW_STRONG = 200 * _ORACLE_NS_MS    # 200ms = high-confidence slow
_ERR_DELTA = 0.05                     # +5 percentage points error rate
_INFRA_RE = re.compile(
    r"mysql|mariadb|postgres|redis|mongo|rabbitmq|kafka|memcached|"
    r"nacos|elasticsearch|zookeeper|consul|etcd", re.I,
)
_DEGRADED = {"ERROR", "SLOW", "INFRA_DEGRADED"}
_NOT_DEGRADED = {"THROUGHPUT", "FLAT", "INFRA_HEALTHY"}


def _oracle_sig(conn, svc: str) -> dict:  # noqa: ANN001
    sig: dict = {}
    for w in ("normal", "abnormal"):
        try:
            r = conn.execute(
                f"SELECT COUNT(*), quantile_cont(duration,0.95), "
                f"quantile_cont(duration,0.99), "
                f"SUM(CASE WHEN \"attr.status_code\"='STATUS_CODE_ERROR' "
                f"  OR TRY_CAST(\"attr.http.response.status_code\" AS INT)>=500 "
                f"  THEN 1 ELSE 0 END) "
                f"FROM {w}_traces WHERE service_name=?", [svc]).fetchone()
        except Exception:  # noqa: BLE001
            r = (0, None, None, 0)
        try:
            lr = conn.execute(
                f"SELECT COUNT(*), SUM(CASE WHEN level='ERROR' THEN 1 ELSE 0 END) "
                f"FROM {w}_logs WHERE service_name=?", [svc]).fetchone()
        except Exception:  # noqa: BLE001
            lr = (0, 0)
        sig[w] = {
            "spans": r[0] or 0, "p95": r[1], "p99": r[2],
            "err_spans": r[3] or 0, "logs": lr[0] or 0, "err_logs": lr[1] or 0,
        }
    return sig


def _oracle_infra_degraded(conn, svc: str) -> bool:  # noqa: ANN001
    try:
        rows = conn.execute(
            "SELECT 'n', AVG(CASE WHEN metric_name LIKE '%cpu%' THEN value END), "
            "  AVG(CASE WHEN metric_name LIKE '%memory%' THEN value END) "
            "FROM normal_metrics WHERE service_name=? "
            "UNION ALL "
            "SELECT 'a', AVG(CASE WHEN metric_name LIKE '%cpu%' THEN value END), "
            "  AVG(CASE WHEN metric_name LIKE '%memory%' THEN value END) "
            "FROM abnormal_metrics WHERE service_name=?", [svc, svc]).fetchall()
        if len(rows) == 2:
            nc, nm, ac, am = rows[0][1], rows[0][2], rows[1][1], rows[1][2]
            if nc and ac and ac > nc * 1.3:
                return True
            if nm and am and am > nm * 1.3:
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


def oracle_classify(conn, svc: str) -> str:  # noqa: ANN001
    """Label one service ERROR/SLOW/DOWN/THROUGHPUT/FLAT/INFRA_* from data."""
    if _INFRA_RE.search(svc):
        return "INFRA_DEGRADED" if _oracle_infra_degraded(conn, svc) \
            else "INFRA_HEALTHY"
    s = _oracle_sig(conn, svc)
    n, a = s["normal"], s["abnormal"]
    n_er = (n["err_logs"] / n["logs"]) if n["logs"] else 0.0
    a_er = (a["err_logs"] / a["logs"]) if a["logs"] else 0.0
    n_se = (n["err_spans"] / n["spans"]) if n["spans"] else 0.0
    a_se = (a["err_spans"] / a["spans"]) if a["spans"] else 0.0

    # Small-sample guard: <20 abnormal spans makes per-span stats noise.
    if a["spans"] < 20:
        if n["spans"] >= 50 and a["spans"] < n["spans"] * 0.5:
            return "DOWN"
        if a_er - n_er > _ERR_DELTA and a_er > 0.05 and a["logs"] >= 50:
            return "ERROR"
        return "FLAT"
    if (a_er - n_er > _ERR_DELTA and a_er > 0.05) or \
       (a_se - n_se > _ERR_DELTA and a_se > 0.05):
        return "ERROR"
    if n["spans"] >= 50 and a["spans"] < max(5, n["spans"] * 0.02):
        return "DOWN"
    if n["p95"] and a["p95"]:
        d = a["p95"] - n["p95"]
        if (d > _SLOW_STRONG and a["p95"] > n["p95"] * 1.3) or \
           (d > _SLOW_FLOOR and a["p95"] > n["p95"] * 1.5):
            return "SLOW"
    if n["p99"] and a["p99"] and a["p99"] > n["p99"] * 2 \
            and a["p99"] - n["p99"] > _SLOW_STRONG and a["p99"] > 500 * _ORACLE_NS_MS:
        return "SLOW"  # tail: a fraction of requests very slow even if p95 flat
    if n["spans"] and a["spans"] < n["spans"] * 0.8:
        if not a["p95"] or not n["p95"] or a["p95"] <= n["p95"] * 1.2:
            return "THROUGHPUT"
    return "FLAT"


@app.command()
def audit(
    dataset_dir: Annotated[Path, typer.Argument(help="dataset with GT labels")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="verifier run output")],
) -> None:
    """Score the verifier graph for SOUNDNESS against the raw-data oracle.

    Reports confirmation precision / recall (oracle = arbiter, not GT) and
    a verifier-vs-GT adjudication: on how many services the verifier
    corrects GT (over/under-labels) vs is itself wrong.
    """
    import duckdb  # noqa: F401  (ensures the dep is present)

    cases = sorted(
        p.name for p in run_dir.iterdir()
        if p.is_dir() and (p / "final_propagation.json").exists()
    )
    agg: dict[str, int] = defaultdict(int)
    rows: list[dict] = []
    for name in cases:
        cdir = dataset_dir / name
        if not (cdir / "causal_graph.json").exists():
            continue
        fp = json.loads((run_dir / name / "final_propagation.json").read_text())
        seeds = set(fp.get("seeds", []))
        ver = set(fp.get("propagated", []))
        gt_seeds, gt = _gt_services(cdir)
        seeds |= gt_seeds
        gt -= seeds
        ver -= seeds
        # hop-rejected = every hop target not in the final set
        rejected: set[str] = set()
        avp = run_dir / name / "all_verdicts.json"
        if avp.exists():
            rejected = {v["to"] for v in json.loads(avp.read_text())}
        rejected -= ver | seeds

        conn = _duckdb_conn(cdir.resolve())
        labels = {s: oracle_classify(conn, s) for s in (ver | gt | rejected)}
        conn.close()

        for s in ver:
            L = labels[s]
            agg["TP" if L in _DEGRADED else
                ("conf_DOWN" if L == "DOWN" else "FP")] += 1
        for s in rejected:
            L = labels[s]
            agg["FN" if L in _DEGRADED else
                ("rej_DOWN" if L == "DOWN" else "TN")] += 1
        for s in (ver | gt):
            L = labels[s]
            inv, ing = s in ver, s in gt
            if L == "DOWN":
                continue
            deg = L in _DEGRADED
            if inv and not ing and deg:
                agg["fix_gt_underlabel"] += 1
            elif inv and not ing and not deg:
                agg["verifier_FP"] += 1
            elif ing and not inv and not deg:
                agg["fix_gt_overlabel"] += 1
            elif ing and not inv and deg:
                agg["verifier_miss"] += 1
        rows.append({"case": name,
                     "verifier": sorted(ver), "gt": sorted(gt),
                     "labels": labels})

    tp, fp_, fn = agg["TP"], agg["FP"], agg["FN"]
    prec = tp / (tp + fp_) if (tp + fp_) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    typer.echo(f"\nCases: {len(rows)}   (oracle = arbiter, GT is NOT truth)")
    typer.echo(f"Confirmations: TP={tp} FP={fp_} DOWN={agg['conf_DOWN']}  "
               f"precision={prec:.3f}")
    typer.echo(f"Rejections:    FN={fn} TN={agg['TN']} DOWN={agg['rej_DOWN']}  "
               f"recall={rec:.3f}")
    typer.echo("\nVerifier vs GT (oracle-arbitrated):")
    typer.echo(f"  corrects GT over-label (GT had non-degraded): "
               f"{agg['fix_gt_overlabel']}")
    typer.echo(f"  corrects GT under-label (GT missed degraded): "
               f"{agg['fix_gt_underlabel']}")
    typer.echo(f"  verifier miss (GT right, verifier wrong):     "
               f"{agg['verifier_miss']}")
    typer.echo(f"  verifier false positive:                      "
               f"{agg['verifier_FP']}")
    out = run_dir / "audit.jsonl"
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    typer.echo(f"\nPer-service labels: {out}")


if __name__ == "__main__":
    app()
