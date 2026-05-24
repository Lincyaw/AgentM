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
    rows = [
        json.loads(line)
        for line in trace_paths[-1].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    log_records: list[dict] = []
    span_names: set[str] = set()
    for row in rows:
        for scope in row.get("scopeLogs", []) or []:
            log_records.extend(scope.get("logRecords", []) or [])
        for scope in row.get("scopeSpans", []) or []:
            for span in scope.get("spans", []) or []:
                name = span.get("name")
                if isinstance(name, str):
                    span_names.add(name)

    def _attr(record: dict, key: str):
        for attr in record.get("attributes", []) or []:
            if attr.get("key") == key:
                v = attr.get("value") or {}
                if "stringValue" in v:
                    return v["stringValue"]
        return None

    def _payload(record: dict) -> dict:
        raw = _attr(record, "agentm.event.payload")
        if not isinstance(raw, str) or not raw:
            return {}
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return {}

    # session.ready body carries the loaded module paths inside the
    # OTLP kvlist body; serialise the whole record and substring-check
    # for the canonical path (the indented JSON is deterministic for
    # short strings).
    ready = next(
        r for r in log_records if r.get("eventName") == "agentm.session.ready"
    )
    assert "contrib.extensions.cc" in json.dumps(ready)
    assert any(
        r.get("eventName") == "agentm.event.dispatch"
        and _attr(r, "agentm.event.channel") == "resources_discover"
        for r in log_records
    )
    assert any(
        r.get("eventName") == "agentm.event.dispatch"
        and _attr(r, "agentm.event.channel") == "command_dispatched"
        and _payload(r).get("name") == "review"
        for r in log_records
    )
    assert any(
        r.get("eventName") == "agentm.api.register"
        and _attr(r, "agentm.api.kind") == "command"
        and _attr(r, "agentm.api.name") == "review"
        for r in log_records
    )
    assert not any(name.startswith("chat ") for name in span_names)
