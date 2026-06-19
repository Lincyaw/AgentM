"""Actionable feedback payloads for verifier submit tools."""
from __future__ import annotations

from typing import Any


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
