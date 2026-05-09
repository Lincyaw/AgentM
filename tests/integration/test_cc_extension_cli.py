from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_cc_package_mount_records_extension_and_resource_events(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "core-manifest.yaml").write_text(
        Path("core-manifest.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    plugin = tmp_path / "plugin"
    commands_dir = plugin / "commands"
    commands_dir.mkdir(parents=True)
    (commands_dir / "review.md").write_text(
        "---\nname: review\ndescription: Review target\n---\nReview the provided target.\n",
        encoding="utf-8",
    )
    registry = tmp_path / "installed_plugins.json"
    registry.write_text(
        json.dumps({"plugins": {"reviewer@local": [{"scope": "user", "installPath": str(plugin)}]}}),
        encoding="utf-8",
    )

    config = {
        "plugins": {"registry_path": str(registry)},
        "commands": {"inherit_claude": False},
        "agents": {"inherit_claude": False},
    }
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "test-key")
    env["PYTHONPATH"] = f"{Path.cwd()}{os.pathsep}{env.get('PYTHONPATH', '')}"
    agentm_bin = Path(sys.executable).with_name("agentm")
    completed = subprocess.run(
        [
            str(agentm_bin),
            "--cwd",
            str(sandbox),
            "--extension",
            f"contrib.extensions.cc:{json.dumps(config)}",
            "--provider",
            "openai",
            "--model",
            "cc-e2e-stub",
            "--quiet",
            "--no-skills",
            "--no-prompt-templates",
            "/review target",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    trace_paths = sorted((sandbox / ".agentm" / "observability").glob("*.jsonl"))
    assert trace_paths, "CLI run did not write an observability trace"
    records = [
        json.loads(line)
        for line in trace_paths[-1].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ready = next(record for record in records if record.get("kind") == "session.ready")
    assert "contrib.extensions.cc" in ready.get("attributes", {}).get("extension_module_paths", [])
    assert any(
        record.get("kind") == "event.dispatch"
        and record.get("name") == "emit:resources_discover"
        for record in records
    )
    assert any(
        record.get("kind") == "event.dispatch"
        and record.get("name") == "emit:command_dispatched"
        and record.get("attributes", {}).get("event", {}).get("name") == "review"
        for record in records
    )
    assert any(
        record.get("kind") == "api.register"
        and record.get("attributes", {}).get("kind") == "command"
        and record.get("attributes", {}).get("name") == "review"
        for record in records
    )
    assert not any(record.get("kind") == "llm.request.start" for record in records)
