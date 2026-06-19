"""ClickHouse session resolution contracts."""

from __future__ import annotations

from typing import Any

from agentm.core.observability import clickhouse


def test_most_recent_session_id_filters_by_cwd(monkeypatch: Any) -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def fake_query(
        url: str,
        sql: str,
        params: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[dict[str, str]]:
        del url
        calls.append((sql, params))
        return [{"sid": "cwd-session"}]

    monkeypatch.setattr(clickhouse, "_query", fake_query)

    assert clickhouse.most_recent_session_id("http://ch", "/workspace") == "cwd-session"
    sql, params = calls[0]
    assert "LogAttributes['agentm.session.cwd'] = {cwd:String}" in sql
    assert params == {"cwd": "/workspace"}


def test_most_recent_session_id_falls_back_for_legacy_rows(monkeypatch: Any) -> None:
    calls: list[dict[str, Any] | None] = []

    def fake_query(
        url: str,
        sql: str,
        params: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[dict[str, str]]:
        del url, sql
        calls.append(params)
        if params:
            return []
        return [{"sid": "legacy-session"}]

    monkeypatch.setattr(clickhouse, "_query", fake_query)

    assert clickhouse.most_recent_session_id("http://ch", "/workspace") == "legacy-session"
    assert calls == [{"cwd": "/workspace"}, {}]
