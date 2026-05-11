from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agentm.core.abi import EventBus
from agentm.core.abi.events import ResourcesDiscoverEvent, SessionReadyEvent
from contrib.extensions import cc
from contrib.extensions.cc import agents, commands, plugins
from contrib.extensions.cc._md_skills import parse_md_skill_records


class _Api:
    def __init__(self, cwd: Path) -> None:
        self.cwd = str(cwd)
        self.events = EventBus()
        self.registered_commands: dict[str, Any] = {}

    def on(self, channel: str, handler: Any, *, priority: int = 500) -> Any:
        return self.events.on(channel, handler, priority=priority)

    def register_command(self, name: str, spec: Any) -> None:
        self.registered_commands[name] = spec

    def send_user_message(self, content: str | list[Any]) -> None:
        self.sent = content


def test_cc_package_exports_tier2_atoms_and_no_flat_compat_files() -> None:
    assert cc.MANIFEST.tier == 2
    assert [manifest.tier for manifest in cc.MANIFESTS] == [2, 2, 2]
    assert agents.MANIFEST.name == "agents"
    assert commands.MANIFEST.name == "commands"
    assert plugins.MANIFEST.name == "plugins"
    assert not list(Path("contrib/extensions/cc").glob("cc_*.py"))


def test_parse_md_skill_records_is_shared_parser(tmp_path: Path) -> None:
    commands_dir = tmp_path / "commands"
    commands_dir.mkdir()
    (commands_dir / "ship.md").write_text(
        "---\nname: ship\ndescription: Ship it\n---\nDo the ship flow.\n",
        encoding="utf-8",
    )

    records = parse_md_skill_records(commands_dir)

    assert [record.name for record in records] == ["ship"]
    assert records[0].description == "Ship it"
    assert records[0].source == "claude-command"


def test_plugins_feed_command_paths_to_commands_via_resources_event(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    plugin = tmp_path / "plugin"
    (plugin / "commands").mkdir(parents=True)
    (plugin / "commands" / "review.md").write_text(
        "---\ndescription: Review patch\n---\nReview this patch.\n",
        encoding="utf-8",
    )
    registry = tmp_path / "installed_plugins.json"
    registry.write_text(
        json.dumps(
            {
                "plugins": {
                    "reviewer@local": [
                        {"scope": "user", "installPath": str(plugin)}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    api = _Api(sandbox)

    async def run() -> list[Any]:
        plugins.install(api, {"registry_path": str(registry)})  # type: ignore[arg-type]
        await commands.install(api, {"inherit_claude": False})  # type: ignore[arg-type]
        await api.events.emit(
            SessionReadyEvent.CHANNEL,
            SessionReadyEvent(
                cwd=str(sandbox),
                session_id="s",
                tool_names=(),
                command_names=(),
                extension_module_paths=(),
                model=None,
                root_session_id="s",
            ),
        )
        return await api.events.emit(
            ResourcesDiscoverEvent.CHANNEL,
            ResourcesDiscoverEvent(cwd=str(sandbox), reason="startup"),
        )

    responses = asyncio.run(run())

    assert "review" in api.registered_commands
    extra_skills = [
        skill
        for response in responses
        if isinstance(response, dict)
        for skill in response.get("extra_skills", [])
    ]
    assert [skill.name for skill in extra_skills] == ["review"]
