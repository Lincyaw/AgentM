"""Fail-stop tests for the duckdb_sql atom's REMOTE mode.

Remote mode turns ``list_tables`` / ``query_sql`` into clients over the
rcabench SDK's blob query endpoint (``BlobApi.blob_query_bucket``), with the
agent's prompt and tool schema unchanged. The SDK call is stubbed (no live
aegis, no rcabench import), so these run anywhere. The load-bearing
positions locked down here are:

1. ``query_sql`` builds the correct ``{prefix, sql}`` request body against
   the right bucket, and the SDK's decoded JSON rows surface in the same
   ``{row_count, rows}`` shape local mode produces (the prompt-facing
   contract).
2. ``list_tables`` reshapes the schema discovery rows into the existing
   ``{data_dir, tables:[...]}`` payload.
3. The endpoint and bearer token are redacted from tool outputs — a leak
   would land secrets in the agent context and the observability trace.
4. With no endpoint configured, install still selects LOCAL mode (the
   backward-compatibility guarantee).
5. The SDK client is wired with the bearer token under the ``BearerAuth``
   api-key field (the exact field a wrong guess 401s on).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agentm_rca.tools import duckdb_sql

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


class _FakeResp:
    """Stand-in for the SDK's ``GenericResponseAny`` (only ``.data`` is read)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data


class _FakeApiException(Exception):
    """Shape-compatible with rcabench ``ApiException`` (``.status`` / ``.body``)."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body


def _install_remote(
    monkeypatch: Any,
    *,
    responder: Any,
    captured: _Captured,
    env: dict[str, str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> _Api:
    monkeypatch.setenv("AGENTM_DUCKDB_TOKEN", _TOKEN)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)

    def _fake_invoke(state: Any, request_body: dict[str, Any]) -> Any:
        captured.calls.append({"bucket": state._bucket, "request_body": request_body})
        return responder(state, request_body)

    # Stubbing the SDK seam means state.blob() is never reached — no rcabench.
    monkeypatch.setattr(duckdb_sql, "_invoke_blob_query", _fake_invoke)

    api = _Api()
    # _Api is a structural test double for ExtensionAPI (register_tool only).
    cfg = cfg or {"endpoint": _ENDPOINT, "bucket": _BUCKET, "dataset": _PREFIX}
    duckdb_sql.install(api, cfg)  # type: ignore[arg-type]
    return api


def test_remote_query_request_shape_and_json_decode(monkeypatch: Any) -> None:
    captured = _Captured()
    rows = [
        {"service_name": "svc-a", "errors": 12, "p99": 1234.5},
        {"service_name": "svc-b", "errors": 0, "p99": 88.0},
    ]

    def responder(state: Any, body: dict[str, Any]) -> _FakeResp:
        return _FakeResp({"row_count": len(rows), "rows": rows})

    api = _install_remote(monkeypatch, responder=responder, captured=captured)
    result = asyncio.run(
        api.tools["query_sql"].fn({"sql": "SELECT service_name, errors, p99 FROM t"})
    )

    # Request: correct bucket + {prefix, sql} body (URL/headers/auth are the
    # SDK's job, exercised end-to-end against live aegis, not here).
    call = captured.calls[-1]
    assert call["bucket"] == _BUCKET
    assert call["request_body"]["prefix"] == _PREFIX
    assert "service_name" in call["request_body"]["sql"]

    # Response: same JSON shape local mode emits (row_count + rows list).
    payload = json.loads(result.content[0].text)
    assert payload["row_count"] == 2
    assert payload["rows"][0]["service_name"] == "svc-a"
    assert payload["rows"][0]["errors"] == 12
    assert not result.is_error


def test_remote_list_tables_via_discovery_query(monkeypatch: Any) -> None:
    # There is ONE server endpoint (/query). list_tables is a discovery query
    # over information_schema, reshaped into the local-mode {tables:[...]}.
    captured = _Captured()
    discovery_rows = [
        {"table_name": "abnormal_traces", "column_name": "trace_id", "data_type": "VARCHAR"},
        {"table_name": "abnormal_traces", "column_name": "duration", "data_type": "BIGINT"},
        {"table_name": "normal_logs", "column_name": "message", "data_type": "VARCHAR"},
    ]

    def responder(state: Any, body: dict[str, Any]) -> _FakeResp:
        return _FakeResp({"row_count": len(discovery_rows), "rows": discovery_rows})

    api = _install_remote(monkeypatch, responder=responder, captured=captured)
    result = asyncio.run(api.tools["list_tables"].fn({}))

    call = captured.calls[-1]
    assert call["bucket"] == _BUCKET
    assert "information_schema.columns" in call["request_body"]["sql"]
    assert call["request_body"]["prefix"] == _PREFIX

    payload = json.loads(result.content[0].text)
    assert payload["data_dir"] == f"blob://{_BUCKET}/{_PREFIX}"
    tables = {t["table"]: t["columns"] for t in payload["tables"]}
    assert tables["abnormal_traces"] == [
        {"name": "trace_id", "type": "VARCHAR"},
        {"name": "duration", "type": "BIGINT"},
    ]
    assert tables["normal_logs"] == [{"name": "message", "type": "VARCHAR"}]
    assert not result.is_error


def test_remote_list_tables_keys_mode_sends_keys(monkeypatch: Any) -> None:
    # keys-mode selector flows into the /query body for list_tables too.
    captured = _Captured()
    keys = ["a/x.parquet", "b/y.parquet"]
    row = [{"table_name": "x", "column_name": "c", "data_type": "VARCHAR"}]

    def responder(state: Any, body: dict[str, Any]) -> _FakeResp:
        return _FakeResp({"row_count": 1, "rows": row})

    api = _install_remote(
        monkeypatch,
        responder=responder,
        captured=captured,
        cfg={"endpoint": _ENDPOINT, "bucket": _BUCKET, "keys": keys},
    )
    asyncio.run(api.tools["list_tables"].fn({}))

    body = captured.calls[-1]["request_body"]
    assert body["keys"] == keys
    assert "prefix" not in body


def test_remote_redacts_endpoint_and_token_on_error(monkeypatch: Any) -> None:
    captured = _Captured()

    def responder(state: Any, body: dict[str, Any]) -> _FakeResp:
        # Server echoes the URL+token in the error body (worst case for leakage).
        leak = f"upstream error reaching {_ENDPOINT} with Bearer {_TOKEN}"
        raise _FakeApiException(500, leak)

    api = _install_remote(monkeypatch, responder=responder, captured=captured)
    result = asyncio.run(api.tools["query_sql"].fn({"sql": "SELECT 1"}))

    text = result.content[0].text
    assert result.is_error
    assert _ENDPOINT not in text
    assert _TOKEN not in text
    assert "<remote>" in text


def test_remote_binder_error_attaches_hint(monkeypatch: Any) -> None:
    # A server-side DuckDB binder error must surface the same dotted-column
    # recovery hint local mode gives, parsed from the SDK's JSON error body.
    captured = _Captured()

    def responder(state: Any, body: dict[str, Any]) -> _FakeResp:
        envelope = json.dumps(
            {"code": 400, "message": 'Binder Error: Referenced column "x" not found'}
        )
        raise _FakeApiException(400, envelope)

    api = _install_remote(monkeypatch, responder=responder, captured=captured)
    result = asyncio.run(api.tools["query_sql"].fn({"sql": "SELECT x FROM t"}))

    payload = json.loads(result.content[0].text)
    assert result.is_error
    assert "Binder Error" in payload["error"]
    assert "double-quote" in payload["hint"].lower()


def test_remote_blob_client_wires_bearer_auth(monkeypatch: Any) -> None:
    # The bearer token must land under the BearerAuth api-key field with a
    # "Bearer" prefix — the exact wiring a wrong guess 401s on. Needs the SDK.
    pytest.importorskip("rcabench")
    monkeypatch.setenv("AGENTM_DUCKDB_TOKEN", _TOKEN)
    state = duckdb_sql._resolve_remote(
        {"endpoint": _ENDPOINT, "bucket": _BUCKET, "dataset": _PREFIX}
    )
    assert state is not None
    blob = state.blob()
    cfg = blob.api_client.configuration
    assert cfg.api_key["BearerAuth"] == _TOKEN
    assert cfg.api_key_prefix["BearerAuth"] == "Bearer"


def test_unset_endpoint_selects_local_mode(monkeypatch: Any, tmp_path: Any) -> None:
    # No endpoint + a valid data_dir ⇒ local mode, unchanged. If remote mode
    # leaked in, install would try to build an SDK client and the local
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
