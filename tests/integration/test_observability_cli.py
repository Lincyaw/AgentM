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
    rows = [json.loads(line) for line in trace_files[0].read_text(encoding="utf-8").splitlines()]
    kinds = {row.get("kind") for row in rows}
    assert {
        "session.start",
        "session.ready",
        "llm.request.start",
        "llm.request.end",
        "turn.summary",
        "session.end",
    } <= kinds
    assert any(
        row.get("kind") == "event.dispatch"
        and row.get("attributes", {}).get("channel") == "turn_end"
        for row in rows
    )


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
    ]
    assert any(
        row.get("kind") == "extension.install"
        and row.get("attributes", {}).get("module_path")
        == "agentm.extensions.builtin.retry_policy"
        for row in rows
    )
    assert any(
        row.get("kind") == "api.register"
        and row.get("attributes", {}).get("kind") == "provider"
        and row.get("attributes", {}).get("name") == "openai"
        for row in rows
    )
    assert any(
        row.get("kind") == "event.dispatch"
        and row.get("attributes", {}).get("channel") == "diagnostic"
        and "verify_ssl=False"
        in row.get("attributes", {}).get("event", {}).get("message", "")
        for row in rows
    )
