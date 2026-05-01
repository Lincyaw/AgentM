"""Builtin atom that mirrors Claude Code's ``.claude/commands/*.md`` files
as AgentM slash commands AND as model-invocable skills.

Each markdown file under the discovered command roots becomes:

1. A :class:`CommandSpec` registered against ``/<name>`` so user-typed slash
   commands inject the body via :meth:`ExtensionAPI.send_user_message`.
2. A :class:`SkillRecord` contributed back to ``skill_loader`` via the
   ``resources_discover`` event, so the same command shows up inside the
   ``<available_skills>`` system block — matching Claude Code's semantic
   where commands and skills are the same surface for the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.core.frontmatter import parse_frontmatter
from agentm.core.skills import SkillRecord
from agentm.extensions import ExtensionManifest
from agentm.harness.events import ResourcesDiscoverEvent
from agentm.harness.extension import CommandSpec, ExtensionAPI


MANIFEST = ExtensionManifest(
    name="claude_commands",
    description=(
        "Register .claude/commands/*.md files as slash commands AND surface "
        "them to the model via the <available_skills> block."
    ),
    registers=("event:resources_discover",),
    config_schema={
        "type": "object",
        "properties": {
            "command_paths": {"type": "array", "items": {"type": "string"}},
            "inherit_claude": {"type": "boolean"},
        },
        "additionalProperties": True,
    },
)


def _default_roots(cwd: str) -> list[Path]:
    return [
        Path.home() / ".claude" / "commands",
        Path(cwd) / ".claude" / "commands",
    ]


def _make_handler(body: str):  # type: ignore[no-untyped-def]
    def _handler(rest: str, api: ExtensionAPI) -> None:
        rest = rest.strip()
        message = f"{body}\n\n{rest}" if rest else body
        api.send_user_message(message)

    return _handler


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    inherit_claude = bool(config.get("inherit_claude", True))
    configured_paths = [Path(str(p)) for p in config.get("command_paths", [])]

    roots: list[Path] = list(configured_paths)
    if inherit_claude:
        roots.extend(_default_roots(api.cwd))

    skill_records: list[SkillRecord] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for md_path in sorted(root.glob("*.md")):
            name = md_path.stem
            if name in seen:
                continue
            try:
                text = md_path.read_text(encoding="utf-8")
            except OSError:
                continue
            metadata, body = parse_frontmatter(text)
            description = metadata.get("description")
            if not isinstance(description, str) or not description.strip():
                description = f"Claude command from {md_path}"
            if not body.strip():
                continue
            api.register_command(
                name,
                CommandSpec(
                    description=description.strip(),
                    handler=_make_handler(body),
                ),
            )
            seen.add(name)
            skill_records.append(
                SkillRecord(
                    name=name,
                    description=description.strip(),
                    file_path=str(md_path.resolve()),
                    base_dir=str(md_path.parent.resolve()),
                    disable_model_invocation=False,
                    source="claude-command",
                )
            )

    def _contribute(_: ResourcesDiscoverEvent) -> dict[str, Any] | None:
        if not skill_records:
            return None
        return {"extra_skills": list(skill_records)}

    api.on("resources_discover", _contribute)
