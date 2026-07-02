"""Actionable feedback payloads for verifier submit tools."""
from __future__ import annotations

import re
from typing import Any

from loguru import logger


def _annotate_sql_failure(failure: dict[str, str]) -> dict[str, str]:
    annotated = dict(failure)
    error = failure.get("error", "")
    if error == "0 rows":
        annotated["why"] = (
            "The SQL executed successfully but returned no rows, so the "
            "evidence is not inspectable by the verifier."
        )
        annotated["fix"] = (
            "If you intended to prove absence, rewrite the query as an "
            "aggregate that always returns one row, for example: WITH q AS "
            "(<your query>) SELECT count(*) AS matching_rows FROM q. A "
            "returned row with matching_rows=0 is valid evidence."
        )
    elif "only query.language='sql'" in error:
        annotated["why"] = "The verifier can only re-execute SQL evidence."
        annotated["fix"] = (
            "Use query.language='sql' and put the DuckDB statement in "
            "query.statement."
        )
    elif 'near "window"' in error or "near 'window'" in error:
        annotated["why"] = (
            "`window` is parsed as a reserved DuckDB keyword in this "
            "statement, so the evidence SQL cannot be re-executed."
        )
        annotated["fix"] = (
            "Rename the output alias from `window` to a non-reserved name "
            "such as `win`, `phase`, or `sample_window`, then rerun the SQL "
            "with query_sql before resubmitting."
        )
    elif error.startswith("missing required modality:"):
        modality = error.rsplit(":", 1)[-1].strip()
        annotated["why"] = (
            f"The case has {modality} tables, but the submitted evidence does "
            f"not include normal-vs-abnormal {modality} SQL."
        )
        annotated["fix"] = (
            f"Add a compact SQL evidence item that queries both normal and "
            f"abnormal {modality} data. If the modality is uninformative, "
            f"return a count/summary row proving that instead of omitting it."
        )
    elif error == "duration divided by 1000":
        annotated["why"] = (
            "Trace duration is stored in nanoseconds. Dividing by 1000 "
            "produces microseconds, not milliseconds."
        )
        annotated["fix"] = (
            "Use duration/1e6 (or duration / 1000000.0) for millisecond "
            "latency evidence, then rerun the SQL before resubmitting."
        )
    elif error == "multiple SQL statements":
        annotated["why"] = (
            "Each evidence item must be one re-executable read-only DuckDB "
            "statement. Multiple semicolon-separated statements cannot be "
            "replayed safely as a single evidence query."
        )
        annotated["fix"] = (
            "Rewrite the evidence as one SELECT/WITH statement, for example "
            "by using UNION ALL or CTEs to compare both windows in one result."
        )
    elif error == "non-read-only SQL statement":
        annotated["why"] = (
            "Evidence SQL must be read-only. Mutating or DDL statements are "
            "not valid verifier evidence."
        )
        annotated["fix"] = (
            "Replace it with a SELECT/WITH/EXPLAIN/DESCRIBE/SHOW/SUMMARIZE "
            "query that returns the same evidence without modifying state."
        )
    else:
        annotated["why"] = (
            "The SQL could not be re-executed against this case's DuckDB views."
        )
        annotated["fix"] = (
            "Run or simplify the SQL with query_sql first, then resubmit using "
            "the corrected table names, column names, and syntax."
        )
    return annotated


def sql_validation_error_payload(failures: list[dict[str, str]]) -> dict[str, Any]:
    """Return SQL validation feedback with concrete repair instructions."""
    annotated = [_annotate_sql_failure(failure) for failure in failures]
    hints = [
        "Every evidence SQL is re-executed before the verdict is accepted.",
    ]
    if any(failure.get("error") == "0 rows" for failure in failures):
        hints.append(
            "A 0-row result is not accepted as evidence. For negative or "
            "absence evidence, return a count/summary row instead of returning "
            "no rows."
        )
    if any(failure.get("error") != "0 rows" for failure in failures):
        hints.append(
            "At least one evidence item is not executable as written; fix it "
            "with query_sql before resubmitting."
        )
    if any("near \"window\"" in failure.get("error", "") for failure in failures):
        hints.append(
            "`window` is a poor SQL alias here; use `win` or `phase` instead."
        )
    if any(failure.get("error") == "duration divided by 1000" for failure in failures):
        hints.append(
            "Trace duration is nanoseconds in this dataset; use /1e6 for "
            "milliseconds in final evidence SQL."
        )
    if any(failure.get("error") == "multiple SQL statements" for failure in failures):
        hints.append(
            "Put each evidence query in a single read-only statement; use "
            "UNION ALL or CTEs instead of semicolon-separated SELECTs."
        )
    hints.append(
        "Resubmit the same verdict after replacing only the failing evidence "
        "items; keep the payload compact to avoid tool-call truncation."
    )
    return {
        "error": "sql_validation_failed",
        "failures": annotated,
        "hint": "Fix the failing evidence SQLs and resubmit.",
        "actionable_hints": hints,
    }


def _mentions_any_table(sql: str, tables: set[str]) -> bool:
    lowered = sql.lower()
    for table in tables:
        pattern = rf"(?<![a-z0-9_]){re.escape(table.lower())}(?![a-z0-9_])"
        if re.search(pattern, lowered):
            return True
    return False


_BAD_DURATION_DIVISOR = re.compile(
    r"\bduration\b\s*(?:\)|\s)*\/\s*(?:1000(?:\.0+)?|1e3)\b",
    re.IGNORECASE,
)


def duration_unit_failures(
    statements: list[tuple[str, str]],
) -> list[dict[str, str]]:
    """Reject evidence that converts trace duration to the wrong unit."""
    failures: list[dict[str, str]] = []
    for location, sql in statements:
        if not _BAD_DURATION_DIVISOR.search(sql):
            continue
        failures.append(
            {
                "location": location,
                "error": "duration divided by 1000",
                "sql": sql,
            }
        )
    return failures


def sql_statement_shape_failure(location: str, sql: str) -> dict[str, str] | None:
    """Reject evidence SQL that is not exactly one read-only statement."""
    try:
        import duckdb
    except ImportError:
        return None

    try:
        parsed = duckdb.extract_statements(sql)
    except Exception as exc:  # noqa: BLE001
        logger.debug("SQL parse failed at {}: {}", location, exc)
        return {
            "location": location,
            "error": str(exc).splitlines()[0][:300],
            "sql": sql,
        }

    if len(parsed) != 1:
        return {
            "location": location,
            "error": "multiple SQL statements",
            "sql": sql,
        }

    statement_type = parsed[0].type
    if statement_type not in {duckdb.StatementType.SELECT, duckdb.StatementType.EXPLAIN}:
        return {
            "location": location,
            "error": "non-read-only SQL statement",
            "sql": sql,
        }

    return None


def modality_coverage_failures(
    statements: list[tuple[str, str]],
    view_names: set[str],
) -> list[dict[str, str]]:
    """Require trace, metric, and log evidence when those views exist."""
    normal_views = {name.lower() for name in view_names if name.startswith("normal_")}
    abnormal_views = {
        name.lower() for name in view_names if name.startswith("abnormal_")
    }
    families = {
        "trace": (
            {name for name in normal_views if name == "normal_traces"},
            {name for name in abnormal_views if name == "abnormal_traces"},
        ),
        "metrics": (
            {name for name in normal_views if name.startswith("normal_metrics")},
            {name for name in abnormal_views if name.startswith("abnormal_metrics")},
        ),
        "logs": (
            {name for name in normal_views if name == "normal_logs"},
            {name for name in abnormal_views if name == "abnormal_logs"},
        ),
    }

    failures: list[dict[str, str]] = []
    for modality, (normal_tables, abnormal_tables) in families.items():
        if not normal_tables or not abnormal_tables:
            continue
        normal_seen = any(
            _mentions_any_table(sql, normal_tables) for _, sql in statements
        )
        abnormal_seen = any(
            _mentions_any_table(sql, abnormal_tables) for _, sql in statements
        )
        if normal_seen and abnormal_seen:
            continue
        failures.append(
            {
                "location": "evidence",
                "error": f"missing required modality: {modality}",
                "sql": "",
            }
        )
    return failures
