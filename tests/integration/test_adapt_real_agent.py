"""Real-model E2E smoke test for the builtin adapt event tools.

Opt in with ``AGENTM_RUN_REAL_LLM_TESTS=1``. The test intentionally drives
the public ``agentm`` CLI and verifies behavior through ``agentm trace`` so
the assertion surface matches how operators debug real sessions.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


_RUN_REAL_LLM = os.environ.get("AGENTM_RUN_REAL_LLM_TESTS") == "1"
_SESSION_RE = re.compile(r"session id:\s*([0-9a-f]{32})")


def _agentm_cmd(*args: str) -> list[str]:
    return [sys.executable, "-c", "from agentm import main; main()", *args]


def _run_agentm(
    *args: str,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _agentm_cmd(*args),
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _parse_trace_tools(stdout: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        records.append(json.loads(line))
    return records


def _tool_json(record: dict[str, Any]) -> dict[str, Any]:
    result = record["result"]
    assert result.get("is_error") is False, record
    content = result["content"]
    assert content and content[0]["type"] == "text", record
    return json.loads(content[0]["text"])


@pytest.mark.slow
@pytest.mark.requires_api_key
@pytest.mark.skipif(
    not _RUN_REAL_LLM,
    reason="set AGENTM_RUN_REAL_LLM_TESTS=1 to run real-LLM AgentM E2E tests",
)
def test_real_agent_adapt_event_tools_emit_expected_trace(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[2]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    observability_dir = tmp_path / "observability"
    observability_dir.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    env = os.environ.copy()
    env["AGENTM_OBSERVABILITY_DIR"] = str(observability_dir)
    env["AGENTM_CLICKHOUSE_URL"] = ""
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = ""
    model = env.get("AGENTM_REAL_LLM_MODEL", "azure-gpt")

    prompt = (
        "Call exactly these tools in order and no other tools: "
        "(1) adapt_list_events with visibility='recommended' and "
        "include_observed=true. "
        "(2) adapt_get_event with channel='tool_result' and include_observed=true. "
        "(3) adapt_event_scaffold with name='event_probe', channel='tool_result', "
        "goal='Observe tool_result events during this smoke test', "
        "tool_name='event_probe_summary'. "
        "(4) adapt_status with {}. "
        "(5) adapt_events with limit=10. "
        "After the five tool calls, briefly report whether each call returned ok=true."
    )

    run = _run_agentm(
        "--cwd",
        str(workspace),
        "--scenario",
        "terminal_bench:arl_adapt",
        "--model",
        model,
        "--set",
        "operations.backend=local",
        "--tools",
        (
            "adapt_list_events,adapt_get_event,adapt_event_scaffold,"
            "adapt_status,adapt_events"
        ),
        "--max-turns",
        "8",
        "--max-tool-calls",
        "5",
        "-p",
        prompt,
        cwd=repo_root,
        env=env,
        timeout=180,
    )
    (log_dir / "agent.stdout.log").write_text(run.stdout, encoding="utf-8")
    (log_dir / "agent.stderr.log").write_text(run.stderr, encoding="utf-8")
    assert run.returncode == 0, (
        f"agentm failed with rc={run.returncode}; logs in {log_dir}\n"
        f"stdout:\n{run.stdout[-4000:]}\n"
        f"stderr:\n{run.stderr[-4000:]}"
    )

    session_match = _SESSION_RE.search(f"{run.stdout}\n{run.stderr}")
    assert session_match is not None, f"session id not found; logs in {log_dir}"
    session_id = session_match.group(1)

    trace = _run_agentm(
        "trace",
        "tools",
        "--session",
        session_id,
        "--format",
        "ndjson",
        cwd=repo_root,
        env=env,
        timeout=60,
    )
    (log_dir / "trace-tools.ndjson").write_text(trace.stdout, encoding="utf-8")
    (log_dir / "trace-tools.stderr.log").write_text(trace.stderr, encoding="utf-8")
    assert trace.returncode == 0, (
        f"trace tools failed with rc={trace.returncode}; logs in {log_dir}\n"
        f"stdout:\n{trace.stdout[-4000:]}\n"
        f"stderr:\n{trace.stderr[-4000:]}"
    )

    records = _parse_trace_tools(trace.stdout)
    tools = [record["tool"] for record in records]
    assert tools == [
        "adapt_list_events",
        "adapt_get_event",
        "adapt_event_scaffold",
        "adapt_status",
        "adapt_events",
    ], f"unexpected tool trace for session {session_id}; logs in {log_dir}: {tools}"

    payloads = {record["tool"]: _tool_json(record) for record in records}

    listed = payloads["adapt_list_events"]
    assert listed["ok"] is True
    assert listed["count"] >= 17
    listed_channels = {event["channel"] for event in listed["events"]}
    assert {"tool_call", "tool_result", "before_agent_start"} <= listed_channels
    tool_call_summary = next(
        event for event in listed["events"] if event["channel"] == "tool_call"
    )
    assert tool_call_summary["observed"]["count"] >= 1
    assert "fields" not in tool_call_summary

    detail = payloads["adapt_get_event"]["event"]
    assert detail["channel"] == "tool_result"
    assert detail["event_type"] == "ToolResultEvent"
    assert detail["hook"]["return_contract"] == "ToolResult | None"
    assert detail["hook"]["notes"][0] == detail["doc"]
    assert detail["observed"]["count"] >= 1
    assert detail["observed"]["last_event_type"] == "ToolResultEvent"

    scaffold = payloads["adapt_event_scaffold"]
    assert scaffold["ok"] is True
    assert scaffold["channel"] == "tool_result"
    assert scaffold["tool_name"] == "event_probe_summary"
    assert "ToolResultEvent" in scaffold["source"]
    assert "api.on(ToolResultEvent.CHANNEL, _on_event)" in scaffold["source"]

    status = payloads["adapt_status"]
    assert status["ok"] is True
    assert status["workspace_root"] == str(workspace)
    assert status["scenario_local_extensions"] is True
    assert any(atom["name"] == "adapt" for atom in status["loaded_atoms"])

    events = payloads["adapt_events"]
    assert events["ok"] is True
    assert isinstance(events["events"], list)
