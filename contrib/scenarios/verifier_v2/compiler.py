"""Deterministic evidence compiler — no LLM.

Re-executes all SQL from the searcher's dossier, extracts values,
checks coverage, and produces a CompiledDossier for the judge.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.core.lib import cap_duckdb_threads
from loguru import logger

from .schema import (
    CompiledDossier,
    CoverageGap,
    EvidenceDossier,
    EvidenceItem,
    SQLResult,
    VerificationTask,
)


def compile_evidence(
    dossier: EvidenceDossier,
    task: VerificationTask,
    data_dir: str,
    data_profile: dict[str, Any] | None = None,
) -> CompiledDossier:
    """Re-execute all SQL and produce a verified compiled dossier."""
    conn = _connect(data_dir)

    relationship_results = _execute_items(conn, dossier.relationship_queries, "relationship")
    target_results = _execute_items(conn, dossier.target_observations, "target")
    control_results = _execute_items(conn, dossier.control_observations, "control")
    counter_results = _execute_items(conn, dossier.counter_evidence, "counter")

    # Inject baseline stats from data_profile as deterministic evidence
    # The judge always sees these regardless of what the searcher queried
    baseline_results = _baseline_from_profile(data_profile, task)
    if baseline_results:
        target_results = baseline_results + target_results

    if conn is not None:
        conn.close()

    coverage_gaps = _check_coverage(
        dossier=dossier,
        task=task,
        relationship_results=relationship_results,
        target_results=target_results,
        control_results=control_results,
        counter_results=counter_results,
    )

    return CompiledDossier(
        task=task,
        relationship_results=relationship_results,
        target_results=target_results,
        control_results=control_results,
        counter_results=counter_results,
        observed_relationship=dossier.observed_relationship,
        affected_endpoints=dossier.affected_endpoints,
        modalities_checked=dossier.modalities_checked,
        coverage_gaps=coverage_gaps,
    )


def _connect(data_dir: str) -> Any:
    """Connect to DuckDB with case parquet files as views."""
    try:
        import duckdb
    except ImportError:
        return None

    conn = duckdb.connect(":memory:")
    cap_duckdb_threads(conn)

    # Register common macros
    for name, quantile in [("p50", "0.5"), ("p90", "0.9"), ("p95", "0.95"), ("p99", "0.99")]:
        try:
            conn.execute(f"CREATE OR REPLACE MACRO {name}(x) AS quantile_cont(x, {quantile})")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not register DuckDB macro {}: {}", name, exc)

    # Register parquet files as views
    data_path = Path(data_dir)
    if data_path.is_dir():
        for f in sorted(data_path.iterdir()):
            if f.is_file() and f.suffix == ".parquet" and f.name != "conclusion.parquet":
                path_str = f.as_posix().replace("'", "''")
                try:
                    conn.execute(
                        f"CREATE OR REPLACE VIEW {f.stem} AS "
                        f"SELECT * FROM read_parquet('{path_str}')"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not expose parquet view {}: {}", f, exc)

    return conn


def _baseline_from_profile(
    data_profile: dict[str, Any] | None,
    task: VerificationTask,
) -> list[SQLResult]:
    """Extract deterministic baseline stats from the pre-computed data_profile.

    These stats are always available to the judge regardless of what the
    searcher happens to query. They provide: span_count, p95_duration,
    error_rate for the target service in normal vs abnormal windows.
    """
    if not data_profile:
        return []

    services = {task.from_entity, task.to_entity} - {""}
    # Also extract services from link seeds
    for svc in list(services):
        if svc.startswith("link:") and "->" in svc:
            parts = svc.removeprefix("link:").split("->", 1)
            services.update(p.strip() for p in parts)

    stats = data_profile.get("statistics", {})
    trace_services = stats.get("traces", {}).get("services", {})
    log_services = stats.get("logs", {}).get("services", {})

    results: list[SQLResult] = []
    for svc in sorted(services):
        trace_data = trace_services.get(svc, {})
        normal = trace_data.get("normal", {})
        abnormal = trace_data.get("abnormal", {})
        if not normal and not abnormal:
            continue

        summary = {
            "service": svc,
            "normal_span_count": normal.get("span_count"),
            "abnormal_span_count": abnormal.get("span_count"),
            "normal_p95_duration_ns": normal.get("p95_duration"),
            "abnormal_p95_duration_ns": abnormal.get("p95_duration"),
            "normal_error_rate": normal.get("error_rate"),
            "abnormal_error_rate": abnormal.get("error_rate"),
            "normal_error_count": normal.get("error_count"),
            "abnormal_error_count": abnormal.get("error_count"),
        }
        # Compute ratios for the judge
        n_spans = normal.get("span_count", 0)
        a_spans = abnormal.get("span_count", 0)
        n_p95 = normal.get("p95_duration")
        a_p95 = abnormal.get("p95_duration")

        interpretation = []
        if n_spans and a_spans:
            ratio = a_spans / n_spans
            interpretation.append(f"span count: {n_spans} → {a_spans} ({ratio:.2f}x)")
        elif n_spans and not a_spans:
            interpretation.append(f"span count: {n_spans} → 0 (service vanished)")

        if n_p95 is not None and a_p95 is not None and n_p95 > 0:
            lat_ratio = a_p95 / n_p95
            n_ms = n_p95 / 1e6
            a_ms = a_p95 / 1e6
            interpretation.append(f"p95 latency: {n_ms:.1f}ms → {a_ms:.1f}ms ({lat_ratio:.1f}x)")

        n_err = normal.get("error_rate", 0)
        a_err = abnormal.get("error_rate", 0)
        if a_err > n_err + 0.01:
            interpretation.append(f"error rate: {n_err:.3f} → {a_err:.3f}")

        results.append(SQLResult(
            location=f"baseline[{svc}]",
            sql=f"(pre-computed from data_profile for {svc})",
            explanation="Deterministic baseline: " + "; ".join(interpretation) if interpretation else f"Baseline stats for {svc}",
            success=True,
            row_count=1,
            sample_values=[{k: v for k, v in summary.items() if v is not None}],
        ))

        # Also add log baseline if available
        log_data = log_services.get(svc, {})
        l_normal = log_data.get("normal", {})
        l_abnormal = log_data.get("abnormal", {})
        if l_normal or l_abnormal:
            log_summary = {
                "service": svc,
                "normal_row_count": l_normal.get("row_count"),
                "abnormal_row_count": l_abnormal.get("row_count"),
                "normal_elevated_count": l_normal.get("elevated_level_count"),
                "abnormal_elevated_count": l_abnormal.get("elevated_level_count"),
            }
            n_elevated = l_normal.get("elevated_level_count", 0)
            a_elevated = l_abnormal.get("elevated_level_count", 0)
            log_interp = []
            if l_normal.get("row_count") and l_abnormal.get("row_count"):
                log_interp.append(
                    f"logs: {l_normal['row_count']} → {l_abnormal['row_count']} rows"
                )
            if a_elevated > n_elevated:
                log_interp.append(
                    f"elevated logs: {n_elevated} → {a_elevated}"
                )
            if log_interp:
                results.append(SQLResult(
                    location=f"baseline_logs[{svc}]",
                    sql=f"(pre-computed log stats for {svc})",
                    explanation="Log baseline: " + "; ".join(log_interp),
                    success=True,
                    row_count=1,
                    sample_values=[{k: v for k, v in log_summary.items() if v is not None}],
                ))

    return results


def _execute_items(
    conn: Any,
    items: list[EvidenceItem],
    category: str,
) -> list[SQLResult]:
    """Execute each evidence item's SQL and capture results."""
    results: list[SQLResult] = []
    for i, item in enumerate(items):
        location = f"{category}[{i}]"
        if conn is None:
            results.append(SQLResult(
                location=location,
                sql=item.sql,
                explanation=item.explanation,
                success=False,
                error="duckdb not available",
            ))
            continue

        try:
            cursor = conn.execute(item.sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            sample = [
                dict(zip(columns, row, strict=True))
                for row in rows[:5]
            ]
            results.append(SQLResult(
                location=location,
                sql=item.sql,
                explanation=item.explanation,
                success=True,
                row_count=len(rows),
                sample_values=sample,
            ))
        except Exception as exc:
            results.append(SQLResult(
                location=location,
                sql=item.sql,
                explanation=item.explanation,
                success=False,
                error=str(exc).splitlines()[0][:300],
            ))

    return results


def _check_coverage(
    *,
    dossier: EvidenceDossier,
    task: VerificationTask,
    relationship_results: list[SQLResult],
    target_results: list[SQLResult],
    control_results: list[SQLResult],
    counter_results: list[SQLResult],
) -> list[CoverageGap]:
    """Identify coverage deficiencies in the dossier."""
    gaps: list[CoverageGap] = []

    # SQL execution failures
    all_results = relationship_results + target_results + control_results + counter_results
    failed = [r for r in all_results if not r.success]
    if failed:
        gaps.append(CoverageGap(
            category="sql_error",
            description=f"{len(failed)} SQL queries failed: "
            + "; ".join(f"{r.location}: {r.error}" for r in failed[:3]),
        ))

    # Empty results (SQL ran but returned nothing)
    empty = [r for r in all_results if r.success and r.row_count == 0]
    if empty:
        gaps.append(CoverageGap(
            category="empty_results",
            description=f"{len(empty)} queries returned 0 rows: "
            + ", ".join(r.location for r in empty[:5]),
        ))

    # Modality coverage
    checked = set(dossier.modalities_checked)
    unavailable = set(dossier.modalities_unavailable)
    required = {"traces", "metrics", "logs"}
    missing = required - checked - unavailable
    if missing:
        gaps.append(CoverageGap(
            category="modality",
            description="Modalities not checked and not marked unavailable: "
            + ", ".join(sorted(missing)),
        ))

    # Control path (required for hop tasks)
    if task.kind == "hop" and not control_results:
        gaps.append(CoverageGap(
            category="control",
            description="No control path comparison provided. Compare the "
            "target's change against sibling endpoints or unaffected services "
            "to establish selectivity.",
        ))

    # Counter evidence (always required)
    if not counter_results and not dossier.counter_evidence:
        gaps.append(CoverageGap(
            category="counter_evidence",
            description="No counter-evidence provided. Actively search for "
            "reasons the observed change might NOT be caused by the upstream "
            "fault (workload shifts, other faults, timing mismatch).",
        ))

    # Relationship evidence (required for hop tasks)
    if task.kind == "hop" and not relationship_results:
        gaps.append(CoverageGap(
            category="relationship",
            description="No relationship evidence provided. Show how "
            f"{task.from_entity} and {task.to_entity} are connected "
            "(calls, shared resources, co-deployment, etc.).",
        ))

    return gaps
