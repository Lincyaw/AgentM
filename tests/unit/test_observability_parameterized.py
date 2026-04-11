"""Focused tests for parameterized SQL construction in observability tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agentm.tools.observability._builders import _build_filter_clauses, _build_time_clause


def test_filter_clauses_parameterize_values_and_resist_injection() -> None:
    clause, params = _build_filter_clauses('{"service_name": "ts-order", "status_code": "500"}')
    assert clause.count("?") == 2
    assert params == ["ts-order", "500"] or params == ["500", "ts-order"]

    malicious = "'; DROP TABLE users; --"
    clause2, params2 = _build_filter_clauses(f'{{"name": "{malicious}"}}')
    assert "DROP" not in clause2
    assert malicious in params2


def test_filter_clauses_invalid_or_empty_input_returns_no_clauses() -> None:
    assert _build_filter_clauses(None) == ("", [])
    assert _build_filter_clauses("not json") == ("", [])


def test_time_clause_parameterization_for_range_and_start_only() -> None:
    clause, params = _build_time_clause("2026-03-08T10:00:00", "2026-03-08T11:00:00")
    assert clause.count("?") == 2
    assert len(params) == 2

    clause2, params2 = _build_time_clause("2026-03-08T10:00:00", None)
    assert clause2.count("?") == 1
    assert len(params2) == 1


def test_query_executes_with_parameter_list() -> None:
    from agentm.tools.observability._core import _query

    with patch("agentm.tools.observability._core.duckdb") as mock_duckdb:
        conn = MagicMock()
        mock_duckdb.connect.return_value = conn
        conn.execute.return_value.fetchall.return_value = []
        conn.execute.return_value.description = []

        _query("SELECT * FROM t WHERE x = ?", ["test_value"])
        conn.execute.assert_called_once_with("SELECT * FROM t WHERE x = ?", ["test_value"])
