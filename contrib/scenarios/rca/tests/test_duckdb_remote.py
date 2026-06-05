"""Fail-stop tests for the duckdb_sql atom's REMOTE mode.

Remote mode turns ``list_tables`` / ``query_sql`` into HTTP clients against
the aegis blob query endpoint, with the agent's prompt and tool schema
unchanged. The transport is stubbed (no live aegis); the load-bearing
positions locked down here are:

1. ``query_sql`` builds the correct ``{prefix, sql}`` POST body + Arrow
   Accept header, and an Arrow IPC response decodes into the same JSON shape
   local mode produces (the prompt-facing contract).
2. ``list_tables`` reshapes the schema response into the existing
   ``{data_dir, tables:[...]}`` payload.
3. The endpoint and bearer token are redacted from tool outputs — a leak
   would land secrets in the agent context and the observability trace.
4. With no endpoint configured, install still selects LOCAL mode (the
   backward-compatibility guarantee).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

# Remote mode needs the optional ``duckdb-remote`` extra. Build the Arrow
# fixture with the same pyarrow the atom decodes with; skip cleanly when the
# extra is not installed (local-only environments).
pa = pytest.importorskip("pyarrow")

from agentm_rca.tools import duckdb_sql  # noqa: E402 - after importorskip guard

_ENDPOINT = "https://aegis.example:8082"
_TOKEN = "supersecret-bearer-xyz"  # noqa: S105 - test fixture, not a real secret
_BUCKET = "my-bucket"
_PREFIX = "cases/batch-01KQ/"


class _Api:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def register_tool(self, t: Any) -> None:
        self.tools[t.name] = t


class _Captured:
    """Records the last request the atom issued so the test can assert on it."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []


def _arrow_stream(rows: list[dict[str, Any]]) -> bytes:
    table = pa.Table.from_pylist(rows)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _install_remote(
    monkeypatch: Any,
    *,
    responder: Any,
    captured: _Captured,
    env: dict[str, str] | None = None,
) -> _Api:
    monkeypatch.setenv("AGENTM_DUCKDB_TOKEN", _TOKEN)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)

    def _fake_request(
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> duckdb_sql._HttpResponse:
        captured.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "params": params,
                "json_body": json_body,
            }
        )
        return responder(method, url, json_body)

    monkeypatch.setattr(duckdb_sql, "_http_request", _fake_request)

    api = _Api()
    # _Api is a structural test double for ExtensionAPI (register_tool only).
    cfg = {"endpoint": _ENDPOINT, "bucket": _BUCKET, "dataset": _PREFIX}
    duckdb_sql.install(api, cfg)  # type: ignore[arg-type]
    return api


def test_remote_query_request_shape_and_arrow_decode(monkeypatch: Any) -> None:
    captured = _Captured()
    rows = [
        {"service_name": "svc-a", "errors": 12, "p99": 1234.5},
        {"service_name": "svc-b", "errors": 0, "p99": 88.0},
    ]

    def responder(method: str, url: str, body: Any) -> duckdb_sql._HttpResponse:
        assert method == "POST"
        return duckdb_sql._HttpResponse(200, _arrow_stream(rows))

    api = _install_remote(monkeypatch, responder=responder, captured=captured)
    result = asyncio.run(
        api.tools["query_sql"].fn({"sql": "SELECT service_name, errors, p99 FROM t"})
    )

    # Request: correct endpoint, {prefix, sql} body, Arrow Accept header.
    call = captured.calls[-1]
    assert call["url"] == f"{_ENDPOINT}/api/v2/blob/buckets/{_BUCKET}/query"
    assert call["json_body"]["prefix"] == _PREFIX
    assert "service_name" in call["json_body"]["sql"]
    assert call["headers"]["Accept"] == "application/vnd.apache.arrow.stream"
    assert call["headers"]["Authorization"] == f"Bearer {_TOKEN}"

    # Response: same JSON shape local mode emits (row_count + rows list).
    payload = json.loads(result.content[0].text)
    assert payload["row_count"] == 2
    assert payload["rows"][0]["service_name"] == "svc-a"
    assert payload["rows"][0]["errors"] == 12
    assert not result.is_error


def test_remote_list_tables_reshapes_schema(monkeypatch: Any) -> None:
    captured = _Captured()
    schema_resp = {
        "tables": [
            {
                "table": "abnormal_traces",
                "row_count": 4096,
                "columns": [
                    {"name": "trace_id", "type": "VARCHAR"},
                    {"name": "duration", "type": "BIGINT"},
                ],
            }
        ]
    }

    def responder(method: str, url: str, body: Any) -> duckdb_sql._HttpResponse:
        assert method == "GET"
        return duckdb_sql._HttpResponse(
            200, json.dumps(schema_resp).encode("utf-8")
        )

    api = _install_remote(monkeypatch, responder=responder, captured=captured)
    result = asyncio.run(api.tools["list_tables"].fn({}))

    call = captured.calls[-1]
    assert call["url"] == f"{_ENDPOINT}/api/v2/blob/buckets/{_BUCKET}/schema"
    assert call["params"] == {"prefix": _PREFIX}

    payload = json.loads(result.content[0].text)
    assert payload["tables"] == schema_resp["tables"]
    assert payload["data_dir"] == f"blob://{_BUCKET}/{_PREFIX}"
    assert not result.is_error


def test_remote_redacts_endpoint_and_token_on_error(monkeypatch: Any) -> None:
    captured = _Captured()

    def responder(method: str, url: str, body: Any) -> duckdb_sql._HttpResponse:
        # Server echoes the URL+token in the error body (worst case for leakage).
        leak = f"upstream error reaching {_ENDPOINT} with Bearer {_TOKEN}"
        return duckdb_sql._HttpResponse(500, leak.encode("utf-8"))

    api = _install_remote(monkeypatch, responder=responder, captured=captured)
    result = asyncio.run(api.tools["query_sql"].fn({"sql": "SELECT 1"}))

    text = result.content[0].text
    assert result.is_error
    assert _ENDPOINT not in text
    assert _TOKEN not in text
    assert "<remote>" in text


def test_unset_endpoint_selects_local_mode(monkeypatch: Any, tmp_path: Any) -> None:
    # No endpoint + a valid data_dir ⇒ local mode, unchanged. If remote mode
    # leaked in, install would try to build an HTTP client and the local
    # data_dir contract would not be honoured.
    monkeypatch.delenv("AGENTM_DUCKDB_ENDPOINT", raising=False)
    monkeypatch.delenv("AGENTM_DUCKDB_BUCKET", raising=False)
    (tmp_path / "abnormal_traces.parquet").write_bytes(b"")  # presence only

    import duckdb

    conn = duckdb.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES ('svc-a', 1)) AS v(name, n)"
        )
        conn.execute(
            f"COPY t TO '{tmp_path / 'abnormal_traces.parquet'}' (FORMAT PARQUET)"
        )
    finally:
        conn.close()

    api = _Api()
    duckdb_sql.install(api, {"data_dir": str(tmp_path)})  # type: ignore[arg-type]
    result = asyncio.run(api.tools["list_tables"].fn({}))
    payload = json.loads(result.content[0].text)
    # Local payload reports the on-disk directory, not a remote handle.
    assert payload["data_dir"] == str(tmp_path)
    assert any(t["table"] == "abnormal_traces" for t in payload["tables"])
