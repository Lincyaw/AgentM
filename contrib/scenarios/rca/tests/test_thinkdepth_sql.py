"""Smoke tests for the ThinkDepthAI-parity tool atom.

Covers the load-bearing parts of the contract: tool surface (4 tools,
exact names), the telemetry-only file allow-list (label-leakage guard),
the bare-filename resolution against ``AGENTM_RCA_DATA_DIR``, and
``think_tool`` echoing reasoning back. Real DuckDB execution is exercised
through a tiny on-disk parquet so the SQL plumbing is also covered.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import duckdb
import pytest

from agentm_rca.tools import thinkdepth_sql


class _Api:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def register_tool(self, t: Any) -> None:
        self.tools[t.name] = t


def _install(data_dir: Path | None = None) -> _Api:
    api = _Api()
    if data_dir is not None:
        os.environ["AGENTM_RCA_DATA_DIR"] = str(data_dir)
    thinkdepth_sql.install(api, {})
    return api


def _make_telemetry_parquet(tmp: Path, name: str) -> Path:
    path = tmp / name
    conn = duckdb.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES "
            "('svc-a', 100, '2024-01-01'), ('svc-b', 200, '2024-01-02')) "
            "AS v(service_name, value, time)"
        )
        conn.execute(f"COPY t TO '{path}' (FORMAT PARQUET)")
    finally:
        conn.close()
    return path


def test_registers_exactly_four_tools() -> None:
    api = _install()
    assert sorted(api.tools) == [
        "get_schema",
        "list_tables_in_directory",
        "query_parquet_files",
        "think_tool",
    ]


def test_think_tool_echoes_reasoning() -> None:
    api = _install()
    result = asyncio.run(api.tools["think_tool"].fn({"reasoning": "next: query traces"}))
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "recorded"
    assert payload["reasoning"] == "next: query traces"


def test_label_leakage_guard_blocks_conclusion_parquet(tmp_path: Path) -> None:
    # The allow-list ("log|trace|metric") explicitly excludes
    # conclusion.parquet — the file that holds the ground-truth fault. A
    # baseline that could read it would silently leak labels and inflate
    # accuracy. Lock the guard down so an accidental rename doesn't
    # disable it.
    api = _install(tmp_path)
    leak = tmp_path / "conclusion.parquet"
    leak.write_bytes(b"")  # path existence is enough to exercise the guard
    result = asyncio.run(
        api.tools["get_schema"].fn({"parquet_file": "conclusion.parquet"})
    )
    payload = json.loads(result.content[0].text)
    assert "Access denied" in payload["error"]
    assert result.is_error


def test_bare_filename_resolves_against_data_dir(tmp_path: Path) -> None:
    api = _install(tmp_path)
    _make_telemetry_parquet(tmp_path, "abnormal_traces.parquet")
    result = asyncio.run(
        api.tools["query_parquet_files"].fn(
            {
                "parquet_files": "abnormal_traces.parquet",
                "query": "SELECT count(*) AS n FROM abnormal_traces",
            }
        )
    )
    payload = json.loads(result.content[0].text)
    assert payload[0]["n"] == 2


def test_list_tables_skips_non_telemetry_files(tmp_path: Path) -> None:
    api = _install(tmp_path)
    _make_telemetry_parquet(tmp_path, "abnormal_traces.parquet")
    (tmp_path / "conclusion.parquet").write_bytes(b"")
    result = asyncio.run(
        api.tools["list_tables_in_directory"].fn({"directory": str(tmp_path)})
    )
    payload = json.loads(result.content[0].text)
    names = [entry["filename"] for entry in payload]
    assert "abnormal_traces.parquet" in names
    assert "conclusion.parquet" not in names


@pytest.mark.parametrize(
    "tool,args",
    [
        ("get_schema", {"parquet_file": "missing_metric.parquet"}),
        ("list_tables_in_directory", {"directory": "/tmp/__nonexistent__"}),
    ],
)
def test_missing_inputs_return_structured_errors(
    tool: str, args: dict[str, Any], tmp_path: Path
) -> None:
    api = _install(tmp_path)
    result = asyncio.run(api.tools[tool].fn(args))
    payload = json.loads(result.content[0].text)
    assert "error" in payload
    assert result.is_error
