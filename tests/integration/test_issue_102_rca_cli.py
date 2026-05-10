from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, ClassVar


class _RcaSubagentOpenAIStub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    parent_calls: ClassVar[int] = 0
    child_calls: ClassVar[int] = 0

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) if length else b"{}")
        tool_names = {
            tool.get("function", {}).get("name")
            for tool in payload.get("tools", [])
            if isinstance(tool, dict)
        }
        if "dispatch_agent" in tool_names:
            type(self).parent_calls += 1
            body = _parent_stream(type(self).parent_calls)
        else:
            type(self).child_calls += 1
            body = _child_stream(type(self).child_calls)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _parent_stream(call_no: int) -> bytes:
    if call_no == 1:
        chunks = [
            _tool_chunk(
                call_id="parent-query",
                name="query_sql",
                arguments={"sql": "SELECT 1"},
            ),
            _finish_chunk("tool_calls"),
            _usage_chunk(),
        ]
    elif call_no == 2:
        chunks = [
            _tool_chunk(
                call_id="dispatch-critic",
                name="dispatch_agent",
                arguments={
                    "purpose": "verify inherited RCA data-tool config",
                    "prompt": "Confirm the inherited duckdb_sql config is available.",
                    "subagent_type": "critic",
                },
            ),
            _finish_chunk("tool_calls"),
            _usage_chunk(),
        ]
    else:
        chunks = [_text_chunk("done"), _finish_chunk("stop"), _usage_chunk()]
    return _sse(chunks)


def _child_stream(call_no: int) -> bytes:
    if call_no == 1:
        chunks = [
            _tool_chunk(
                call_id="child-query",
                name="query_sql",
                arguments={"sql": "SELECT 1"},
            ),
            _finish_chunk("tool_calls"),
            _usage_chunk(),
        ]
    else:
        chunks = [_text_chunk("critic inherited config"), _finish_chunk("stop"), _usage_chunk()]
    return _sse(chunks)


def _sse(chunks: list[dict[str, Any]]) -> bytes:
    return b"".join(
        b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"
        for chunk in chunks
    ) + b"data: [DONE]\n\n"


def _tool_chunk(*, call_id: str, name: str, arguments: dict[str, str]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-issue-102",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "issue-102-stub",
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


def _text_chunk(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-issue-102",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "issue-102-stub",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": None,
            }
        ],
    }


def _finish_chunk(reason: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-issue-102",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "issue-102-stub",
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
    }


def _usage_chunk() -> dict[str, Any]:
    return {
        "id": "chatcmpl-issue-102",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "issue-102-stub",
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _write_stub_rca_modules(root: Path) -> Path:
    package = root / "stubs" / "agentm_rca"
    tools = package / "tools"
    tools.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n",
        encoding="utf-8",
    )
    (tools / "__init__.py").write_text("", encoding="utf-8")
    for module_name, tool_name in [
        ("duckdb_sql", "query_sql"),
        ("hypothesis_tools", "record_hypothesis"),
        ("finalize", "submit_final_report"),
        ("worker_finalize", "return_response"),
    ]:
        (tools / f"{module_name}.py").write_text(
            "from __future__ import annotations\n"
            "import json\n"
            "from typing import Any\n"
            "from agentm.core.abi import FunctionTool, TextContent, ToolResult\n"
            "from agentm.extensions import ExtensionManifest\n\n"
            f"MANIFEST = ExtensionManifest(name='{module_name}', description='stub {module_name}', registers=('tool:{tool_name}',))\n\n"
            "def install(api: Any, _config: dict[str, Any]) -> None:\n"
            "    async def _tool(_args: dict[str, Any]) -> ToolResult:\n"
            "        payload = json.dumps({'config': _config}, sort_keys=True)\n"
            "        return ToolResult(content=[TextContent(type='text', text=payload)])\n\n"
            "    api.register_tool(\n"
            "        FunctionTool(\n"
            f"            name='{tool_name}',\n"
            f"            description='stub {tool_name}',\n"
            "            parameters={'type': 'object', 'properties': {}, 'additionalProperties': False},\n"
            "            fn=_tool,\n"
            "        )\n"
            "    )\n",
            encoding="utf-8",
        )
    (package / "worker_skills.py").write_text(
        "from agentm.extensions import ExtensionManifest\n"
        "MANIFEST = ExtensionManifest(name='worker_skills', description='stub worker_skills', registers=())\n"
        "def install(api, config):\n"
        "    return None\n",
        encoding="utf-8",
    )
    return root / "stubs"


def _read_trace_rows(sandbox: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in sorted((sandbox / ".agentm" / "observability").glob("*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in trace.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _query_sql_result_configs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for row in rows:
        attrs = row.get("attributes", {})
        if row.get("kind") != "event.dispatch" or attrs.get("channel") != "tool_result":
            continue
        event = attrs.get("event", {})
        if event.get("tool_name") != "query_sql":
            continue
        result = event.get("result", {})
        content = result.get("content", [])
        if not content or not isinstance(content[0], dict):
            continue
        payload = json.loads(str(content[0].get("text", "{}")))
        config = payload.get("config")
        call_id = event.get("tool_call_id")
        if isinstance(call_id, str) and isinstance(config, dict):
            configs[call_id] = config
    return configs


def test_rca_cli_critic_inherits_parent_duckdb_config_in_trace(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    data_dir = sandbox / "data"
    data_dir.mkdir()
    stub_path = _write_stub_rca_modules(sandbox)
    source_scenario = Path.cwd() / "contrib" / "scenarios" / "rca"
    scenario_dir = sandbox / "contrib" / "scenarios" / "rca"
    scenario_dir.mkdir(parents=True)
    shutil.copy2(source_scenario / "manifest.yaml", scenario_dir / "manifest.yaml")
    shutil.copytree(source_scenario / "agents", scenario_dir / "agents")
    shutil.copytree(source_scenario / "prompts", scenario_dir / "prompts")
    scenario = scenario_dir / "manifest.yaml"

    _RcaSubagentOpenAIStub.parent_calls = 0
    _RcaSubagentOpenAIStub.child_calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RcaSubagentOpenAIStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = os.environ.copy()
        env.update(
            {
                "AGENTM_RCA_DATA_DIR": str(data_dir),
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                "PYTHONPATH": (
                    f"{stub_path}{os.pathsep}{Path.cwd()}"
                    f"{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
                ),
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from agentm.cli import main; main()",
                "dispatch the RCA critic once",
                "--cwd",
                str(sandbox),
                "--scenario",
                str(scenario),
                "--provider",
                "openai",
                "--model",
                "issue-102-stub",
                "--quiet",
                "--no-skills",
                "--no-prompt-templates",
            ],
            cwd=sandbox,
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
    assert _RcaSubagentOpenAIStub.parent_calls >= 1
    assert _RcaSubagentOpenAIStub.child_calls >= 1

    rows = _read_trace_rows(sandbox)
    configs = _query_sql_result_configs(rows)
    expected = {"exclude": ["conclusion.parquet"]}
    assert configs["parent-query"] == expected
    assert configs["child-query"] == expected
