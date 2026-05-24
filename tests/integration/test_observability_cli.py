from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any


class _OpenAIStubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        length = int(self.headers.get("content-length", "0"))
        if length:
            self.rfile.read(length)
        chunks = [
            {
                "id": "chatcmpl-observability-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "obs-stub",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "observed"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-observability-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "obs-stub",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
            {
                "id": "chatcmpl-observability-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "obs-stub",
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ]
        body = b"".join(
            b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"
            for chunk in chunks
        ) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def test_cli_observability_trace_contains_identity_events(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    scenario_dir = sandbox / "contrib" / "scenarios" / "obs_cli"
    scenario_dir.mkdir(parents=True)
    manifest = scenario_dir / "manifest.yaml"
    manifest.write_text(
        "name: obs_cli\n"
        "extensions:\n  - module: agentm.extensions.builtin.operations_local\n"
        "  - module: agentm.extensions.builtin.observability\n"
        "    config:\n"
        "      path: .agentm/observability/{session_id}.jsonl\n"
        "      include_handler_records: false\n",
        encoding="utf-8",
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIStubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = os.environ.copy()
        env.update(
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from agentm.cli import main; main()",
                "record an observability trace",
                "--cwd",
                str(sandbox),
                "--scenario",
                str(manifest),
                "--provider",
                "openai",
                "--model",
                "obs-stub",
                "--quiet",
                "--no-skills",
                "--no-prompt-templates",
            ],
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert completed.returncode == 0, completed.stderr
    trace_files = sorted((sandbox / ".agentm" / "observability").glob("*.jsonl"))
    assert len(trace_files) == 1
    rows = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_names: set[str] = set()
    span_names: set[str] = set()
    dispatch_channels: set[str] = set()
    for row in rows:
        for scope in row.get("scopeLogs", []) or []:
            for record in scope.get("logRecords", []) or []:
                name = record.get("eventName")
                if isinstance(name, str):
                    event_names.add(name)
                if name == "agentm.event.dispatch":
                    for attr in record.get("attributes", []) or []:
                        if attr.get("key") == "agentm.event.channel":
                            v = attr.get("value", {}).get("stringValue")
                            if isinstance(v, str):
                                dispatch_channels.add(v)
        for scope in row.get("scopeSpans", []) or []:
            for span in scope.get("spans", []) or []:
                name = span.get("name")
                if isinstance(name, str):
                    span_names.add(name)
    assert {
        "agentm.session.start",
        "agentm.session.ready",
        "agentm.turn.summary",
        "agentm.session.end",
    } <= event_names
    assert any(name.startswith("chat ") for name in span_names), span_names
    assert "turn_end" in dispatch_channels


def test_cli_retry_policy_composition_is_visible_in_observability(
    tmp_path: Path,
) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    scenario_dir = sandbox / "contrib" / "scenarios" / "retry_obs"
    scenario_dir.mkdir(parents=True)
    manifest = scenario_dir / "manifest.yaml"
    manifest.write_text(
        "name: retry_obs\n"
        "extensions:\n  - module: agentm.extensions.builtin.operations_local\n"
        "  - module: agentm.extensions.builtin.observability\n"
        "    config:\n"
        "      path: .agentm/observability/{session_id}.jsonl\n"
        "      include_handler_records: false\n"
        "  - module: agentm.extensions.builtin.retry_policy\n"
        "    config:\n"
        "      max_retries: 1\n"
        "      base_delay: 0\n",
        encoding="utf-8",
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIStubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = os.environ.copy()
        env.update(
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                "OPENAI_VERIFY_SSL": "false",
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from agentm.cli import main; main()",
                "record retry policy composition",
                "--cwd",
                str(sandbox),
                "--scenario",
                str(manifest),
                "--provider",
                "openai",
                "--model",
                "obs-stub",
                "--quiet",
                "--no-skills",
                "--no-prompt-templates",
            ],
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert completed.returncode == 0, completed.stderr
    trace_files = sorted((sandbox / ".agentm" / "observability").glob("*.jsonl"))
    assert len(trace_files) == 1
    rows = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    log_records: list[dict[str, Any]] = []
    for row in rows:
        for scope in row.get("scopeLogs", []) or []:
            log_records.extend(scope.get("logRecords", []) or [])

    def _attr(record: dict[str, Any], key: str) -> Any:
        for attr in record.get("attributes", []) or []:
            if attr.get("key") == key:
                v = attr.get("value") or {}
                if "stringValue" in v:
                    return v["stringValue"]
                if "intValue" in v:
                    try:
                        return int(v["intValue"])
                    except (TypeError, ValueError):
                        return v["intValue"]
                if "boolValue" in v:
                    return v["boolValue"]
        return None

    assert any(
        r.get("eventName") == "agentm.extension.install"
        and _attr(r, "agentm.extension.module_path")
        == "agentm.extensions.builtin.retry_policy"
        for r in log_records
    )
    assert any(
        r.get("eventName") == "agentm.api.register"
        and _attr(r, "agentm.api.kind") == "provider"
        and _attr(r, "agentm.api.name") == "openai"
        for r in log_records
    )
    # The diagnostic dispatch carries the payload as a JSON-encoded
    # attribute (since arbitrary dicts can't ride directly in OTel
    # attribute slots). Decode and probe for the message substring.
    found_diagnostic = False
    for r in log_records:
        if r.get("eventName") != "agentm.event.dispatch":
            continue
        if _attr(r, "agentm.event.channel") != "diagnostic":
            continue
        payload_str = _attr(r, "agentm.event.payload")
        if not isinstance(payload_str, str):
            continue
        try:
            payload = json.loads(payload_str)
        except (TypeError, ValueError):
            continue
        if "verify_ssl=False" in str(payload.get("message", "")):
            found_diagnostic = True
            break
    assert found_diagnostic
