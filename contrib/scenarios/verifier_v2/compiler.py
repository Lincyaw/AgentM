"""Deterministic evidence compiler — no LLM.

Re-executes all SQL from the searcher's dossier, extracts values,
checks coverage, and produces a CompiledDossier for the judge.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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
) -> CompiledDossier:
    """Re-execute all SQL and produce a verified compiled dossier."""
    conn = _connect(data_dir)

    relationship_results = _execute_items(conn, dossier.relationship_queries, "relationship")
    target_results = _execute_items(conn, dossier.target_observations, "target")
    control_results = _execute_items(conn, dossier.control_observations, "control")
    counter_results = _execute_items(conn, dossier.counter_evidence, "counter")

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
    cap = os.environ.get("AGENTM_DUCKDB_THREADS")
    if cap:
        try:
            conn.execute(f"SET threads={max(1, int(cap))}")
        except (ValueError, Exception):  # noqa: S110
            pass

    # Register common macros
    for name, quantile in [("p50", "0.5"), ("p90", "0.9"), ("p95", "0.95"), ("p99", "0.99")]:
        try:
            conn.execute(f"CREATE OR REPLACE MACRO {name}(x) AS quantile_cont(x, {quantile})")
        except Exception:  # noqa: BLE001, S110
            pass

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
                except Exception:  # noqa: BLE001, S110
                    pass

    return conn


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
                dict(zip(columns, row))
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
