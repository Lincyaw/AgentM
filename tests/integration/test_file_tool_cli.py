from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, ClassVar


class _FileToolOpenAIStub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    request_count: ClassVar[int] = 0

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        length = int(self.headers.get("content-length", "0"))
        if length:
            self.rfile.read(length)
        type(self).request_count += 1
        body = _stream_body(type(self).request_count)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _stream_body(call_no: int) -> bytes:
    if call_no == 1:
        chunks = [
            _tool_chunk(
                call_id="call-find",
                name="find",
                arguments={"pattern": "*.py", "path": "."},
            ),
            _finish_chunk("tool_calls"),
            _usage_chunk(),
        ]
    elif call_no == 2:
        chunks = [
            _tool_chunk(
                call_id="call-grep",
                name="grep",
                arguments={"pattern": "needle", "path": "."},
            ),
            _finish_chunk("tool_calls"),
            _usage_chunk(),
        ]
    else:
        chunks = [
            {
                "id": "chatcmpl-file-tool-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "file-tool-stub",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "done"},
                        "finish_reason": None,
                    }
                ],
            },
            _finish_chunk("stop"),
            _usage_chunk(),
        ]
    return b"".join(
        b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"
        for chunk in chunks
    ) + b"data: [DONE]\n\n"


def _tool_chunk(*, call_id: str, name: str, arguments: dict[str, str]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-file-tool-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "file-tool-stub",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
    }


def _finish_chunk(reason: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-file-tool-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "file-tool-stub",
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
    }


def _usage_chunk() -> dict[str, Any]:
    return {
        "id": "chatcmpl-file-tool-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "file-tool-stub",
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_cli_file_tools_honor_gitignore_in_trace(tmp_path: Path) -> None:
    """E2E: drive CLI file tools and inspect trajectory, not harness internals."""

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (sandbox / "target.py").write_text("needle\n", encoding="utf-8")
    (sandbox / "ignored.py").write_text("needle hidden\n", encoding="utf-8")
    scenario_dir = sandbox / "contrib" / "scenarios" / "file_tools"
    scenario_dir.mkdir(parents=True)
    manifest = scenario_dir / "manifest.yaml"
    manifest.write_text(
        "name: file_tools\n"
        "extensions:\n"
        "  - module: agentm.extensions.builtin.observability\n"
        "    config:\n"
        "      path: .agentm/observability/{session_id}.jsonl\n"
        "      include_handler_records: false\n"
        "  - module: agentm.extensions.builtin.tool_find\n"
        "  - module: agentm.extensions.builtin.tool_grep\n",
        encoding="utf-8",
    )

    _FileToolOpenAIStub.request_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FileToolOpenAIStub)
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
                "find and grep python files",
                "--cwd",
                str(sandbox),
                "--scenario",
                str(manifest),
                "--provider",
                "openai",
                "--model",
                "file-tool-stub",
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

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert _FileToolOpenAIStub.request_count >= 3

    trace_files = sorted((sandbox / ".agentm" / "observability").glob("*.jsonl"))
    assert len(trace_files) == 1
    rows = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tool_results = [
        row.get("attributes", {}).get("event", {})
        for row in rows
        if row.get("kind") == "event.dispatch"
        and row.get("attributes", {}).get("channel") == "tool_result"
    ]
    assert {event.get("tool_name") for event in tool_results} >= {"find", "grep"}
    rendered_results = json.dumps(tool_results)
    assert "target.py" in rendered_results
    assert "ignored.py" not in rendered_results
