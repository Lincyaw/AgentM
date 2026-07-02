"""Relationship graph building from traces and metrics."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from loguru import logger

from . import duckdb_conn

SYNTHETIC = {
    "loadgenerator", "locust", "wrk2", "dsb-wrk2", "k6",
    "load-generator", "load_generator",
}

Rel = tuple[str, str, str]  # (service_a, service_b, rel_type)

_DB_INFRA = re.compile(r"mysql|mariadb|postgres|sqlserver|oracle|cockroach", re.I)
_INFRA_NAME = re.compile(
    r"mysql|mariadb|postgres|sqlserver|oracle|cockroach|mongo|redis|"
    r"memcached|rabbitmq|kafka|consul|etcd|zookeeper|nacos|elasticsearch|"
    r"cassandra|clickhouse|minio",
    re.I,
)
_DBOP_SPAN_SQL = (
    "(span_name LIKE 'SELECT%' OR span_name LIKE 'INSERT%' "
    "OR span_name LIKE 'UPDATE%' OR span_name LIKE 'DELETE%' "
    "OR span_name LIKE 'Transaction%' OR span_name LIKE 'COMMIT%' "
    "OR span_name LIKE 'Session%' OR span_name LIKE '%Repository%')"
)


def get_relationships(data_dir: Path) -> list[Rel]:
    """Build bidirectional relationship list from call graph + deployment."""
    conn = duckdb_conn(data_dir)
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
    except Exception as exc:  # noqa: BLE001
        # Dependency table may be absent for this case — skip co-deploy edges.
        logger.debug("verifier graph: co-deploy relation query failed: {}", exc)

    conn.close()
    return rels


def build_neighbor_graph(
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
        except Exception as exc:  # noqa: BLE001
            # Trace table {tbl} may not exist for this case — try the next one.
            logger.debug("verifier graph: service query on {} failed: {}", tbl, exc)
            continue
        out.update(r[0] for r in rows if r[0])
    return out


def get_infra_nodes(data_dir: Path) -> set[str]:
    """Backing components present in metrics but never in traces."""
    conn = duckdb_conn(data_dir)
    try:
        metric_svcs = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT service_name FROM normal_metrics"
            ).fetchall()
            if r[0]
        }
    except Exception:  # noqa: BLE001
        logger.debug("Failed to query normal_metrics for infra nodes")
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
    """Link each infra node to the services that depend on it."""
    if not infra_nodes:
        return []
    conn = duckdb_conn(data_dir)
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
    except Exception as exc:  # noqa: BLE001
        # Source table absent/empty for this case — leave the list as-is.
        logger.debug("verifier graph: high-frequency caller query failed: {}", exc)

    trace_svcs = _trace_services(conn)
    conn.close()

    for node in infra_nodes:
        if _DB_INFRA.search(node):
            callers = db_callers
        else:
            tokens = set(node.replace("_", "-").split("-"))
            callers = [s for s in trace_svcs if s in tokens or s in node]
        for svc in callers:
            rels.append((svc, node, "infra_dependency"))
            rels.append((node, svc, "infra_dependency"))
    return rels


def profile_dataset(
    data_dir: Path, max_distinct: int = 25,
) -> dict[str, dict[str, object]]:
    """Column profiles + low-cardinality value distributions per table pair."""
    conn = duckdb_conn(data_dir)
    tables = {
        f.stem for f in data_dir.iterdir()
        if f.suffix == ".parquet" and f.name != "conclusion.parquet"
    }
    pairs = sorted(
        base for base in {t.removeprefix("normal_") for t in tables if t.startswith("normal_")}
        if f"abnormal_{base}" in tables
    )
    profile: dict[str, dict[str, object]] = {}
    for base in pairs:
        cols = [
            r[1] for r in conn.execute(
                f"PRAGMA table_info('normal_{base}')"
            ).fetchall()
        ]
        col_profile: dict[str, dict[str, dict[str, int]]] = {}
        for col in cols:
            qcol = '"' + col.replace('"', '""') + '"'
            try:
                n_card, a_card = (
                    conn.execute(
                        f"SELECT COUNT(DISTINCT {qcol}) FROM {win}_{base}"
                    ).fetchone()[0]
                    for win in ("normal", "abnormal")
                )
            except Exception as exc:  # noqa: BLE001
                # Cardinality query failed (missing column/table) — skip column.
                logger.debug("verifier graph: cardinality query failed: {}", exc)
                continue
            if max(n_card, a_card) > max_distinct:
                continue
            dist: dict[str, dict[str, int]] = {}
            for win in ("normal", "abnormal"):
                rows = conn.execute(
                    f"SELECT COALESCE(CAST({qcol} AS VARCHAR), 'NULL'), COUNT(*) "
                    f"FROM {win}_{base} GROUP BY 1 ORDER BY 2 DESC"
                ).fetchall()
                dist[win] = {str(v): int(c) for v, c in rows}
            col_profile[col] = dist
        profile[base] = {"columns": cols, "value_distributions": col_profile}
    conn.close()
    return profile


def vanished_endpoints(
    data_dir: Path, services: list[str], min_normal: int = 5,
) -> dict[str, list[dict[str, object]]]:
    """Endpoints present in normal window but absent from abnormal."""
    if not services:
        return {}
    conn = duckdb_conn(data_dir)
    out: dict[str, list[dict[str, object]]] = {}
    for svc in services:
        try:
            rows = conn.execute(
                "SELECT n.span_name, n.cnt, COALESCE(a.cnt, 0) "
                "FROM (SELECT span_name, COUNT(*) cnt FROM normal_traces "
                "      WHERE service_name = ? GROUP BY 1) n "
                "LEFT JOIN (SELECT span_name, COUNT(*) cnt FROM abnormal_traces "
                "      WHERE service_name = ? GROUP BY 1) a "
                "USING (span_name) "
                "WHERE n.cnt >= ? AND COALESCE(a.cnt, 0) = 0 "
                "ORDER BY n.cnt DESC",
                [svc, svc, min_normal],
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            # Per-service query failed for this case — skip to the next service.
            logger.debug("verifier graph: per-service edge query failed: {}", exc)
            continue
        if rows:
            out[svc] = [
                {"span_name": r[0], "normal": int(r[1]), "abnormal": int(r[2])}
                for r in rows
            ]
    conn.close()
    return out
