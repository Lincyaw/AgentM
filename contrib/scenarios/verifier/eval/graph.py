"""Phase 0: relationship graph building from traces and metrics."""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

SYNTHETIC = {
    "loadgenerator", "locust", "wrk2", "dsb-wrk2", "k6",
    "load-generator", "load_generator",
}

Rel = tuple[str, str, str]  # (service_a, service_b, rel_type)

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


def _duckdb_conn(data_dir: Path):  # noqa: ANN202
    import duckdb

    conn = duckdb.connect(":memory:")
    _cap = os.environ.get("AGENTM_DUCKDB_THREADS")
    if _cap:
        try:
            conn.execute(f"SET threads={max(1, int(_cap))}")
        except (ValueError, duckdb.Error):
            pass
    for f in sorted(data_dir.iterdir()):
        if f.is_file() and f.suffix == ".parquet" and f.name != "conclusion.parquet":
            conn.execute(
                f"CREATE VIEW {f.stem} AS "
                f"SELECT * FROM read_parquet('{f.as_posix()}')"
            )
    return conn


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
        "GROUP BY 1, 2 "
        "ORDER BY cnt DESC"
    ).fetchall()
    for caller, callee, _cnt in rows:
        if caller not in SYNTHETIC and callee not in SYNTHETIC:
            rels.append((callee, caller, "callee_to_caller"))
            rels.append((caller, callee, "caller_to_callee"))

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
                rels.append((svc_a, svc_b, "co_deployed"))
                rels.append((svc_b, svc_a, "co_deployed"))
    except Exception:  # noqa: BLE001
        pass

    conn.close()
    return rels


def _build_neighbor_graph(
    rels: list[Rel],
) -> dict[str, list[tuple[str, str]]]:
    """Return ``{service: [(neighbour, rel_type)]}``."""
    graph: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for svc_a, svc_b, rel in rels:
        key = (svc_a, svc_b, rel)
        if key not in seen:
            seen.add(key)
            graph[svc_a].append((svc_b, rel))
    return dict(graph)


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
            rels.append((svc, node, "infra_dependency"))
            rels.append((node, svc, "infra_dependency"))
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
