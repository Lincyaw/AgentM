"""Compact deterministic case profiling for verifier prompts and review packets."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from . import duckdb_conn
from .fpg import injection_node_id, injection_subject
from .graph import SYNTHETIC, profile_dataset, vanished_endpoints
from .schema import Injection

_MAX_SERVICES = 80
_MAX_RELATIONSHIPS = 200
_MAX_ANOMALIES = 80


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _columns(conn: Any, table: str) -> list[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        return [str(row[1]) for row in rows]
    except Exception as exc:  # noqa: BLE001
        logger.debug("verifier profile: column discovery failed for {}: {}", table, exc)
        return []


def _row_count(conn: Any, table: str) -> int | None:
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except Exception as exc:  # noqa: BLE001
        logger.debug("verifier profile: row count failed for {}: {}", table, exc)
        return None


def _first(cols: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(cols)
    lowered = {col.lower(): col for col in cols}
    for candidate in candidates:
        if candidate in available:
            return candidate
        hit = lowered.get(candidate.lower())
        if hit:
            return hit
    return None


def _service_col(cols: Iterable[str]) -> str | None:
    return _first(cols, ("service_name", "service", "svc"))


def _table_services(conn: Any, table: str) -> set[str]:
    cols = _columns(conn, table)
    service_col = _service_col(cols)
    if not service_col:
        return set()
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {_quote(service_col)} FROM {table} "
            f"WHERE {_quote(service_col)} IS NOT NULL"
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.debug("verifier profile: service discovery failed for {}: {}", table, exc)
        return set()
    return {str(row[0]) for row in rows if row[0] and str(row[0]) not in SYNTHETIC}


def _status_error_expr(cols: Iterable[str]) -> str:
    parts: list[str] = []
    for col in cols:
        lowered = col.lower()
        if "status" not in lowered and "error" not in lowered:
            continue
        qcol = _quote(col)
        parts.append(
            f"lower(CAST({qcol} AS VARCHAR)) IN "
            "('error', 'err', 'status_code_error', 'true', 'failed', 'failure')"
        )
        if "http" in lowered or "status" in lowered:
            parts.append(f"TRY_CAST({qcol} AS INTEGER) >= 500")
    return " OR ".join(f"({part})" for part in parts) or "FALSE"


def _window_tables(conn: Any, base: str) -> tuple[str, str] | None:
    normal = f"normal_{base}"
    abnormal = f"abnormal_{base}"
    if _columns(conn, normal) and _columns(conn, abnormal):
        return normal, abnormal
    return None


def _trace_stats(conn: Any, data_dir: Path) -> dict[str, Any]:
    tables = _window_tables(conn, "traces")
    if not tables:
        return {"services": {}, "vanished_endpoints": {}}
    normal_table, abnormal_table = tables
    common_cols = [
        col for col in _columns(conn, normal_table)
        if col in set(_columns(conn, abnormal_table))
    ]
    service_col = _service_col(common_cols)
    if not service_col:
        return {"services": {}, "vanished_endpoints": {}}
    span_col = _first(common_cols, ("span_name", "name", "operation_name"))
    duration_col = _first(
        common_cols,
        ("duration", "duration_ns", "duration_nano", "duration_ms", "latency"),
    )
    span_expr = (
        f"COUNT(DISTINCT {_quote(span_col)}) AS distinct_span_names"
        if span_col
        else "0 AS distinct_span_names"
    )
    p95_expr = (
        f"quantile_cont(TRY_CAST({_quote(duration_col)} AS DOUBLE), 0.95) AS p95"
        if duration_col
        else "NULL AS p95"
    )
    services: dict[str, dict[str, Any]] = {}
    for win, table in (("normal", normal_table), ("abnormal", abnormal_table)):
        error_expr = _status_error_expr(_columns(conn, table))
        try:
            rows = conn.execute(
                f"SELECT {_quote(service_col)} AS svc, COUNT(*) AS spans, "
                f"{span_expr}, "
                f"SUM(CASE WHEN {error_expr} THEN 1 ELSE 0 END) AS errors, "
                f"{p95_expr} FROM {table} "
                f"WHERE {_quote(service_col)} IS NOT NULL GROUP BY 1"
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("verifier profile: trace stats failed for {}: {}", table, exc)
            continue
        for svc, spans, distinct_spans, errors, p95 in rows:
            if not svc or str(svc) in SYNTHETIC:
                continue
            services.setdefault(str(svc), {})[win] = {
                "span_count": int(spans or 0),
                "distinct_span_names": int(distinct_spans or 0),
                "error_count": int(errors or 0),
                "error_rate": round(float(errors or 0) / float(spans or 1), 6),
                "p95_duration": float(p95) if p95 is not None else None,
            }
    try:
        vanished = vanished_endpoints(data_dir, sorted(services)[:_MAX_SERVICES])
    except Exception as exc:  # noqa: BLE001
        logger.debug("verifier profile: vanished endpoint query failed: {}", exc)
        vanished = {}
    return {"services": services, "vanished_endpoints": vanished}


def _sample_stats(
    conn: Any,
    base: str,
    *,
    name_columns: Sequence[str],
    elevated_level: bool = False,
) -> dict[str, Any]:
    tables = _window_tables(conn, base)
    if not tables:
        return {"services": {}}
    normal_table, abnormal_table = tables
    common_cols = [
        col for col in _columns(conn, normal_table)
        if col in set(_columns(conn, abnormal_table))
    ]
    service_col = _service_col(common_cols)
    if not service_col:
        return {"services": {}}
    name_col = _first(common_cols, name_columns)
    distinct_expr = (
        f"COUNT(DISTINCT {_quote(name_col)}) AS distinct_names"
        if name_col
        else "0 AS distinct_names"
    )
    level_col = _first(common_cols, ("severity_text", "level", "log_level", "severity"))
    elevated_expr = "0 AS elevated_count"
    if elevated_level and level_col:
        qlevel = _quote(level_col)
        elevated_expr = (
            f"SUM(CASE WHEN lower(CAST({qlevel} AS VARCHAR)) LIKE '%error%' "
            f"OR lower(CAST({qlevel} AS VARCHAR)) LIKE '%fatal%' "
            f"OR lower(CAST({qlevel} AS VARCHAR)) LIKE '%warn%' "
            "THEN 1 ELSE 0 END) AS elevated_count"
        )
    services: dict[str, dict[str, Any]] = {}
    for win, table in (("normal", normal_table), ("abnormal", abnormal_table)):
        try:
            rows = conn.execute(
                f"SELECT {_quote(service_col)} AS svc, COUNT(*) AS rows, "
                f"{distinct_expr}, {elevated_expr} FROM {table} "
                f"WHERE {_quote(service_col)} IS NOT NULL GROUP BY 1"
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("verifier profile: {} stats failed for {}: {}", base, table, exc)
            continue
        for svc, row_count, distinct_names, elevated_count in rows:
            if not svc or str(svc) in SYNTHETIC:
                continue
            key = "sample_count" if base == "metrics" else "row_count"
            services.setdefault(str(svc), {})[win] = {
                key: int(row_count or 0),
                "distinct_names": int(distinct_names or 0),
                "elevated_level_count": int(elevated_count or 0),
            }
    return {"services": services}


def _relationships(graph: Mapping[str, Sequence[Sequence[str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for src, neighbors in graph.items():
        for info in neighbors:
            if len(info) < 2:
                continue
            key = (src, str(info[0]), str(info[1]))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"src": key[0], "dst": key[1], "rel_type": key[2]})
    return rows[:_MAX_RELATIONSHIPS]


def build_data_profile(
    data_dir: Path,
    graph: Mapping[str, Sequence[Sequence[str]]],
    infra_nodes: Sequence[str],
) -> dict[str, Any]:
    """Build a bounded JSON-serializable profile before any LLM calls."""
    tables = sorted(
        file.stem for file in data_dir.iterdir()
        if file.is_file() and file.suffix == ".parquet" and file.name != "conclusion.parquet"
    )
    conn = duckdb_conn(data_dir)
    try:
        table_info = {
            table: {"columns": _columns(conn, table), "row_count": _row_count(conn, table)}
            for table in tables
        }
        observed_services = set(infra_nodes)
        for table in tables:
            observed_services.update(_table_services(conn, table))
        graph_services = set(graph)
        for neighbors in graph.values():
            graph_services.update(str(info[0]) for info in neighbors if info)
        services = sorted((observed_services | graph_services) - SYNTHETIC)
        statistics = {
            "traces": _trace_stats(conn, data_dir),
            "metrics": _sample_stats(
                conn,
                "metrics",
                name_columns=("metric_name", "name", "metric", "__name__"),
            ),
            "logs": _sample_stats(
                conn,
                "logs",
                name_columns=("body", "message", "template", "event_name"),
                elevated_level=True,
            ),
        }
    finally:
        conn.close()
    try:
        low_cardinality = profile_dataset(data_dir, max_distinct=12)
    except Exception as exc:  # noqa: BLE001
        logger.debug("verifier profile: low-cardinality profile failed: {}", exc)
        low_cardinality = {}
    return {
        "structure": {
            "tables": table_info,
            "modalities": {
                "traces": any(table.endswith("traces") for table in tables),
                "metrics": any("metrics" in table for table in tables),
                "logs": any(table.endswith("logs") for table in tables),
            },
            "services": services[:_MAX_SERVICES],
            "infra_nodes": sorted(infra_nodes),
            "relationships": _relationships(graph),
            "low_cardinality_columns": low_cardinality,
        },
        "statistics": statistics,
    }


def _ratio(after: float, before: float) -> float | None:
    return None if before <= 0 else after / before


def _append_anomaly(
    records: list[dict[str, Any]],
    *,
    subject: str,
    modality: str,
    signal: str,
    normal: Any,
    abnormal: Any,
    status: str,
    summary: str,
) -> None:
    records.append(
        {
            "id": f"{modality}:{signal}:{subject}:{len(records)}",
            "subject": subject,
            "modality": modality,
            "signal": signal,
            "status": status,
            "normal": normal,
            "abnormal": abnormal,
            "summary": summary,
        }
    )


def _service_windows(
    stats: Mapping[str, Any],
    modality: str,
) -> Mapping[str, Any]:
    block = stats.get(modality, {})
    if isinstance(block, Mapping):
        services = block.get("services", {})
        if isinstance(services, Mapping):
            return services
    return {}


def build_anomaly_inventory(data_profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Summarize visible normal/abnormal differences without assigning cause."""
    records: list[dict[str, Any]] = []
    stats = data_profile.get("statistics", {}) if isinstance(data_profile, Mapping) else {}
    if not isinstance(stats, Mapping):
        return records

    for service, windows in _service_windows(stats, "traces").items():
        if not isinstance(windows, Mapping):
            continue
        normal = windows.get("normal", {}) if isinstance(windows.get("normal"), Mapping) else {}
        abnormal = windows.get("abnormal", {}) if isinstance(windows.get("abnormal"), Mapping) else {}
        subject = f"svc:{service}"
        n_spans = float(normal.get("span_count") or 0)
        a_spans = float(abnormal.get("span_count") or 0)
        if n_spans >= 5 and (a_spans == 0 or (_ratio(a_spans, n_spans) or 1) <= 0.2):
            _append_anomaly(
                records,
                subject=subject,
                modality="trace",
                signal="span_volume_drop",
                normal=int(n_spans),
                abnormal=int(a_spans),
                status="changed",
                summary="trace volume drops sharply in abnormal",
            )
        n_rate = float(normal.get("error_rate") or 0)
        a_rate = float(abnormal.get("error_rate") or 0)
        n_err = float(normal.get("error_count") or 0)
        a_err = float(abnormal.get("error_count") or 0)
        if a_err >= 3 and a_rate - n_rate >= 0.05:
            _append_anomaly(
                records,
                subject=subject,
                modality="trace",
                signal="error_rate_increase",
                normal=n_rate,
                abnormal=a_rate,
                status="changed",
                summary="trace error rate increases in abnormal",
            )
        elif n_err >= 3 and a_err >= 3 and abs(a_rate - n_rate) <= 0.03:
            _append_anomaly(
                records,
                subject=subject,
                modality="trace",
                signal="stable_error_background",
                normal=n_rate,
                abnormal=a_rate,
                status="stable_preexisting_or_noisy",
                summary="error signal is present before and after with little shift",
            )
        n_p95, a_p95 = normal.get("p95_duration"), abnormal.get("p95_duration")
        if isinstance(n_p95, (int, float)) and isinstance(a_p95, (int, float)):
            if (_ratio(float(a_p95), float(n_p95)) or 0) >= 2:
                _append_anomaly(
                    records,
                    subject=subject,
                    modality="trace",
                    signal="p95_latency_increase",
                    normal=n_p95,
                    abnormal=a_p95,
                    status="changed",
                    summary="trace p95 latency increases materially in abnormal",
                )

    trace_block = stats.get("traces", {})
    vanished = trace_block.get("vanished_endpoints", {}) if isinstance(trace_block, Mapping) else {}
    if isinstance(vanished, Mapping):
        for service, endpoints in vanished.items():
            if not isinstance(endpoints, list):
                continue
            for endpoint in endpoints[:5]:
                _append_anomaly(
                    records,
                    subject=f"svc:{service}",
                    modality="trace",
                    signal="vanished_endpoint",
                    normal=endpoint.get("normal"),
                    abnormal=endpoint.get("abnormal"),
                    status="changed",
                    summary=f"span {endpoint.get('span_name')} vanished in abnormal",
                )

    for modality, count_key, signal in (
        ("metrics", "sample_count", "metric_sample_shift"),
        ("logs", "row_count", "log_volume_shift"),
    ):
        for service, windows in _service_windows(stats, modality).items():
            if not isinstance(windows, Mapping):
                continue
            normal = windows.get("normal", {}) if isinstance(windows.get("normal"), Mapping) else {}
            abnormal = windows.get("abnormal", {}) if isinstance(windows.get("abnormal"), Mapping) else {}
            n_count = float(normal.get(count_key) or 0)
            a_count = float(abnormal.get(count_key) or 0)
            ratio = _ratio(a_count, n_count)
            if n_count >= 5 and (a_count == 0 or (ratio is not None and ratio <= 0.2)):
                _append_anomaly(
                    records,
                    subject=f"svc:{service}",
                    modality=modality.rstrip("s"),
                    signal=signal,
                    normal=int(n_count),
                    abnormal=int(a_count),
                    status="changed",
                    summary=f"{modality} volume drops sharply in abnormal",
                )
            elif ratio is not None and ratio >= 5 and a_count >= 10:
                _append_anomaly(
                    records,
                    subject=f"svc:{service}",
                    modality=modality.rstrip("s"),
                    signal=signal,
                    normal=int(n_count),
                    abnormal=int(a_count),
                    status="changed",
                    summary=f"{modality} volume increases sharply in abnormal",
                )

    priority = {"changed": 0, "stable_preexisting_or_noisy": 1}
    records.sort(
        key=lambda row: (
            priority.get(str(row.get("status")), 9),
            str(row.get("subject")),
            str(row.get("signal")),
        )
    )
    return records[:_MAX_ANOMALIES]


def _service_from_subject(subject: object) -> str | None:
    text = str(subject or "")
    return text.removeprefix("svc:") if text.startswith("svc:") else None


def _related_services(
    target: str,
    graph: Mapping[str, Sequence[Sequence[str]]],
    extra: Iterable[str] = (),
) -> list[str]:
    related = {target, *extra}
    for src, neighbors in graph.items():
        if src == target or src in related:
            related.update(str(info[0]) for info in neighbors if info)
        for info in neighbors:
            if info and str(info[0]) == target:
                related.add(src)
    return sorted(service for service in related if service and service not in SYNTHETIC)[:30]


def build_seed_observation_surfaces(
    injections: Sequence[Injection],
    graph: Mapping[str, Sequence[Sequence[str]]],
    data_profile: Mapping[str, Any],
    anomaly_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build compact per-seed maps of nearby entities and visible anomalies."""
    structure = data_profile.get("structure", {})
    structure = structure if isinstance(structure, Mapping) else {}
    relationships = structure.get("relationships", [])
    relationships = relationships if isinstance(relationships, list) else []
    out: dict[str, dict[str, Any]] = {}
    for inj in injections:
        seed_id = injection_node_id(inj)
        target = inj.get("effect_target") or inj.get("target") or seed_id
        endpoints = [value for value in (inj.get("edge_source"), inj.get("edge_target")) if value]
        services = _related_services(str(target), graph, endpoints)
        service_set = set(services)
        surface: dict[str, Any] = {
            "seed": seed_id,
            "subject": injection_subject(inj),
            "target": inj.get("target"),
            "effect_target": target,
            "nearby_services": services,
            "available_modalities": structure.get("modalities", {}),
            "relationships": [
                rel for rel in relationships[:_MAX_RELATIONSHIPS]
                if isinstance(rel, Mapping)
                and (rel.get("src") in service_set or rel.get("dst") in service_set)
            ][:40],
            "visible_anomalies": [
                dict(record) for record in anomaly_inventory
                if _service_from_subject(record.get("subject")) in service_set
            ][:12],
        }
        if endpoints:
            surface["link"] = {
                "source": inj.get("edge_source"),
                "target": inj.get("edge_target"),
                "note": "Verify the actually exercised direction before confirming.",
            }
        out[seed_id] = surface
    return out


__all__ = [
    "build_anomaly_inventory",
    "build_data_profile",
    "build_seed_observation_surfaces",
]
