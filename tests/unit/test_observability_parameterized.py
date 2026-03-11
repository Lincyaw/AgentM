"""Tests for parameterized SQL in observability tools.

Bug prevented: SQL injection via LLM-controlled filter values passed
through string interpolation instead of parameterized queries.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from agentm.tools.observability._builders import (
    _build_filter_clauses,
    _build_time_clause,
)


class TestBuildFilterClauses:
    """_build_filter_clauses should return parameterized clauses."""

    def test_returns_params_for_valid_filters(self):
        clause, params = _build_filter_clauses('{"service_name": "ts-order"}')
        assert "?" in clause
        assert params == ["ts-order"]
        assert "ts-order" not in clause

    def test_empty_input_returns_no_params(self):
        clause, params = _build_filter_clauses(None)
        assert clause == ""
        assert params == []

    def test_invalid_json_returns_no_params(self):
        clause, params = _build_filter_clauses("not json")
        assert clause == ""
        assert params == []

    def test_multiple_filters_return_multiple_params(self):
        clause, params = _build_filter_clauses(
            '{"service_name": "ts-order", "status_code": "500"}'
        )
        assert clause.count("?") == 2
        assert len(params) == 2

    def test_sql_injection_attempt_is_parameterized(self):
        """Values with SQL injection patterns must be passed as params, not interpolated."""
        malicious = "'; DROP TABLE users; --"
        clause, params = _build_filter_clauses(f'{{"name": "{malicious}"}}')
        assert "DROP" not in clause
        assert malicious in params


class TestBuildTimeClause:
    """_build_time_clause should return parameterized clauses."""

    def test_returns_params_for_time_range(self):
        clause, params = _build_time_clause(
            "2026-03-08T10:00:00", "2026-03-08T11:00:00"
        )
        assert clause.count("?") == 2
        assert len(params) == 2

    def test_start_only_returns_one_param(self):
        clause, params = _build_time_clause("2026-03-08T10:00:00", None)
        assert clause.count("?") == 1
        assert len(params) == 1

    def test_none_returns_empty(self):
        clause, params = _build_time_clause(None, None)
        assert clause == ""
        assert params == []


class TestQueryWithParams:
    """_query should pass params to DuckDB execute."""

    @patch("agentm.tools.observability._core.duckdb")
    def test_execute_called_with_params(self, mock_duckdb):
        from agentm.tools.observability._core import _query

        mock_conn = MagicMock()
        mock_duckdb.connect.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_conn.execute.return_value.description = []

        _query("SELECT * FROM t WHERE x = ?", ["test_value"])

        mock_conn.execute.assert_called_once_with(
            "SELECT * FROM t WHERE x = ?", ["test_value"]
        )
