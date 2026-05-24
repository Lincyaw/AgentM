from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_slash_command_wins_over_prompt_template_collision(tmp_path: Path) -> None:
    """E2E: drive the CLI, then inspect trajectory records for slash dispatch."""

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    prompts_dir = sandbox / ".agentm" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "ship.md").write_text("template should not run\n", encoding="utf-8")

    (tmp_path / "e2e_command.py").write_text(
        "from __future__ import annotations\n"
        "from agentm.core.abi.extension import CommandSpec\n"
        "\n"
        "def install(api, config):\n"
        "    del config\n"
        "    async def _ship(args, owner_api):\n"
        "        result = await owner_api.get_resource_writer().write(\n"
        "            'command_won.txt', args.encode('utf-8'), rationale='slash command output'\n"
        "        )\n"
        "        if result.error is not None:\n"
        "            raise RuntimeError(result.error)\n"
        "    api.register_command('ship', CommandSpec(description='ship', handler=_ship))\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    agentm_bin = Path(sys.executable).with_name("agentm")
    completed = subprocess.run(
        [
            str(agentm_bin),
            "--cwd",
            str(sandbox),
            "--extension",
            "e2e_command",
            "--quiet",
            "/ship now",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert (sandbox / "command_won.txt").read_text(encoding="utf-8") == "now"

    trace_paths = sorted((sandbox / ".agentm" / "observability").glob("*.jsonl"))
    assert trace_paths, "CLI run did not write an observability trace"
    rows = [
        json.loads(line)
        for line in trace_paths[-1].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Walk OTLP log records: a slash-command dispatch shows up as an
    # ``agentm.event.dispatch`` on the ``command_dispatched`` channel
    # whose JSON-encoded payload names the command.
    found_ship = False
    saw_chat_span = False
    for row in rows:
        for scope in row.get("scopeLogs", []) or []:
            for record in scope.get("logRecords", []) or []:
                if record.get("eventName") != "agentm.event.dispatch":
                    continue
                attrs = {a.get("key"): a.get("value", {}) for a in record.get("attributes", []) or []}
                channel = attrs.get("agentm.event.channel", {}).get("stringValue")
                if channel != "command_dispatched":
                    continue
                payload_str = attrs.get("agentm.event.payload", {}).get("stringValue", "")
                if not isinstance(payload_str, str) or not payload_str:
                    continue
                try:
                    payload = json.loads(payload_str)
                except (TypeError, ValueError):
                    continue
                if payload.get("name") == "ship":
                    found_ship = True
        for scope in row.get("scopeSpans", []) or []:
            for span in scope.get("spans", []) or []:
                if str(span.get("name", "")).startswith("chat "):
                    saw_chat_span = True
    assert found_ship
    assert not saw_chat_span, "slash command should short-circuit before any LLM call"
