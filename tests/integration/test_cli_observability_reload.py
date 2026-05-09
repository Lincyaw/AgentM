from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


_OLD_ATOM_SOURCE = '''
from __future__ import annotations

from pathlib import Path

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="flaky_atom",
    description="CLI reload rollback regression atom",
    registers=("tool:flaky_demo",),
)


def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    marker = Path(__file__).with_suffix(".install_count")
    count = int(marker.read_text(encoding="utf-8")) if marker.exists() else 0
    if count >= 1:
        raise RuntimeError("old rollback install exploded")
    marker.write_text(str(count + 1), encoding="utf-8")

    async def _execute(args: dict[str, object]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="old-live")])

    api.register_tool(
        FunctionTool(
            name="flaky_demo",
            description="demo tool",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            fn=_execute,
        )
    )
'''


_NEW_ATOM_SOURCE = '''
from __future__ import annotations

from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="flaky_atom",
    description="CLI reload rollback regression atom",
    registers=("tool:flaky_demo",),
)


def install(api: ExtensionAPI, config: dict[str, object]) -> None:
    raise RuntimeError("new install exploded")
'''


class _OpenAIStub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    calls = 0
    reload_source = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return None

    def do_POST(self) -> None:  # noqa: N802
        type(self).calls += 1
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        if type(self).calls == 1:
            chunks = _tool_call_chunks(type(self).reload_source)
        else:
            chunks = _final_answer_chunks()
        body = _sse(chunks)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)


def _sse(chunks: Iterable[dict[str, Any] | str]) -> bytes:
    lines: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, str):
            lines.append(f"data: {chunk}\n\n")
        else:
            lines.append(f"data: {json.dumps(chunk)}\n\n")
    return "".join(lines).encode("utf-8")


def _chunk(delta: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "agentm-test",
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }


def _tool_call_chunks(source: str) -> list[dict[str, Any] | str]:
    args = json.dumps(
        {
            "name": "flaky_atom",
            "source": source,
            "rationale": "exercise CLI rollback double failure",
        }
    )
    return [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-reload",
                        "type": "function",
                        "function": {"name": "reload_atom", "arguments": args},
                    }
                ],
            },
            None,
        ),
        _chunk({}, "tool_calls"),
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "agentm-test",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        "[DONE]",
    ]


def _final_answer_chunks() -> list[dict[str, Any] | str]:
    return [
        _chunk({"role": "assistant", "content": "done"}, None),
        _chunk({}, "stop"),
        "[DONE]",
    ]


def _write_cli_package(tmp_path: Path) -> str:
    pkg = "cli_reload_pkg"
    pkg_dir = tmp_path / pkg
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "flaky_atom.py").write_text(_OLD_ATOM_SOURCE, encoding="utf-8")
    return pkg


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_cli_repo(tmp_path: Path, pkg: str) -> None:
    (tmp_path / "core-manifest.yaml").write_text(
        "version: 1\n"
        "constitution:\n"
        "  paths:\n"
        "    - core-manifest.yaml\n"
        "managed:\n"
        "  globs:\n"
        f"    - {pkg}/**.py\n"
        "extension_api:\n"
        "  current: 1\n"
        "  semver_rules: {major: x, minor: x, patch: x}\n"
        "  deprecation:\n"
        "    grace: 1\n"
        "reload:\n"
        "  tier_2_atoms: []\n",
        encoding="utf-8",
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "add", "core-manifest.yaml", pkg)
    _git(tmp_path, "commit", "-m", "seed cli reload fixture", "--quiet")


def _trace_records(tmp_path: Path) -> list[dict[str, Any]]:
    traces = list((tmp_path / ".agentm" / "observability").glob("*.jsonl"))
    assert traces, "CLI run did not produce an observability trace"
    records: list[dict[str, Any]] = []
    for line in traces[0].read_text(encoding="utf-8").splitlines():
        records.append(json.loads(line))
    return records


def test_cli_trace_records_rollback_double_failure(tmp_path: Path) -> None:
    pkg = _write_cli_package(tmp_path)
    _init_cli_repo(tmp_path, pkg)
    _OpenAIStub.calls = 0
    _OpenAIStub.reload_source = _NEW_ATOM_SOURCE
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = os.environ.copy()
        env.update(
            {
                "AGENTM_PROVIDER": "openai",
                "AGENTM_MODEL": "agentm-test",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
            }
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from agentm.cli import main; main()",
                "--cwd",
                str(tmp_path),
                "--extension",
                f"{pkg}.flaky_atom",
                "--extension",
                "_agentm_contrib__tool_catalog",
                "reload flaky atom",
            ],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert result.returncode == 0, result.stderr
    atom_reload_records = [
        record for record in _trace_records(tmp_path)
        if record.get("kind") == "atom.reload"
    ]
    assert atom_reload_records
    assert atom_reload_records[-1]["status"]["code"] == "ERROR"
    assert (
        atom_reload_records[-1]["status"]["message"]
        == "rollback_failure_state_preserved"
    )
