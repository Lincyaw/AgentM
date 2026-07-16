"""Discover Claude Code agent definitions and resolve them for ``sub_agent``.

Scans ``.claude/agents/*.md`` in the project directory, the user's home
``~/.claude/agents/``, and installed Claude Code plugins
(``~/.claude/plugins/cache/<source>/<plugin>/<version>/agents/``).
Parsed agent personas are injected into the system prompt as an
``<available_agents>`` block and resolved on ``ResolveSubagentEvent``
so :mod:`agentm.extensions.builtin.sub_agent` can dispatch them via
``subagent_type``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from pydantic import BaseModel

from agentm.core.abi import (
    BeforeAgentStartEvent,
    ExtensionAPI,
    ResolveSubagentEvent,
    SessionReadyEvent,
)
from agentm.core.lib import expand_path_from_cwd, parse_frontmatter
from agentm.extensions import ExtensionManifest


class ClaudeAgentsConfig(BaseModel):
    extra_paths: list[str] = []
    inherit_claude: bool = True


MANIFEST = ExtensionManifest(
    name="claude_agents",
    description=(
        "Discover .claude/agents/*.md agent definitions and resolve them "
        "for dispatch_agent via the resolve_subagent event."
    ),
    registers=(
        f"event:{BeforeAgentStartEvent.CHANNEL}",
        f"event:{ResolveSubagentEvent.CHANNEL}",
        f"event:{SessionReadyEvent.CHANNEL}",
    ),
    config_schema=ClaudeAgentsConfig,
)


# ---------------------------------------------------------------------------
# Frontmatter parsing helpers
# ---------------------------------------------------------------------------


def _parse_tools(raw: Any) -> list[str] | None:
    if isinstance(raw, list):
        tools = [str(item).strip() for item in raw if str(item).strip()]
        return tools or None
    if isinstance(raw, str) and raw.strip():
        tools = [t.strip() for t in raw.split(",") if t.strip()]
        return tools or None
    return None


def _parse_string_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _parse_budget_defaults(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    budget: dict[str, int] = {}
    for key in ("max_tool_calls", "max_turns"):
        value = raw.get(key)
        if isinstance(value, int) and value > 0:
            budget[key] = value
    return budget or None


def _parse_input_schema(raw: Any) -> dict[str, list[str]] | None:
    if not isinstance(raw, dict):
        return None
    required = _parse_string_list(raw.get("required"))
    optional = _parse_string_list(raw.get("optional"))
    if not required and not optional:
        return None
    return {"required": required, "optional": optional}


# ---------------------------------------------------------------------------
# Agent discovery
# ---------------------------------------------------------------------------


def _version_sort_key(path: Path) -> tuple[int, ...]:
    """Parse ``path.name`` as a dotted version for numeric comparison.

    Non-numeric segments (e.g. ``unknown``, git hashes) sort *below* any
    real version so the newest semver wins.
    """
    parts: list[int] = []
    for segment in path.name.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            return (-1,)
    return tuple(parts)


def _discover_dirs(cwd: str, inherit_claude: bool) -> list[Path]:
    dirs: list[Path] = []
    dirs.append(expand_path_from_cwd(".claude/agents", cwd))
    if not inherit_claude:
        return dirs
    dirs.append(Path.home() / ".claude" / "agents")
    plugin_cache = Path.home() / ".claude" / "plugins" / "cache"
    if plugin_cache.is_dir():
        for source_dir in plugin_cache.iterdir():
            if not source_dir.is_dir():
                continue
            for plugin_dir in sorted(source_dir.iterdir()):
                if not plugin_dir.is_dir():
                    continue
                for version_dir in sorted(
                    plugin_dir.iterdir(),
                    key=_version_sort_key,
                    reverse=True,
                ):
                    if not version_dir.is_dir():
                        continue
                    agents_dir = version_dir / "agents"
                    if agents_dir.is_dir():
                        dirs.append(agents_dir)
    return dirs


def _load_agents(dirs: list[Path]) -> dict[str, dict[str, Any]]:
    agents: dict[str, dict[str, Any]] = {}
    seen_files: set[str] = set()
    for agents_dir in dirs:
        if not agents_dir.is_dir():
            continue
        for md_path in sorted(agents_dir.glob("*.md")):
            real = str(md_path.resolve())
            if real in seen_files:
                continue
            seen_files.add(real)
            try:
                text = md_path.read_text(encoding="utf-8")
            except OSError:
                continue
            metadata, body = parse_frontmatter(text)
            if not body.strip():
                continue
            meta = metadata if isinstance(metadata, dict) else {}
            name = meta.get("name")
            if not isinstance(name, str) or not name.strip():
                name = md_path.stem
            name = name.strip()
            if name in agents:
                continue
            agents[name] = {
                "body": body.strip(),
                "tools": _parse_tools(meta.get("tools")),
                "description": str(meta.get("description", "")).strip(),
                "file_path": str(md_path),
                "input_schema": _parse_input_schema(meta.get("input_schema")),
                "budget_defaults": _parse_budget_defaults(meta.get("budget_defaults")),
                "artifact_kinds": _parse_string_list(meta.get("artifact_kinds")),
            }
    return agents


# ---------------------------------------------------------------------------
# available_agents XML block (inlined per §11)
# ---------------------------------------------------------------------------


def _field(persona: Any, key: str, default: Any = "") -> Any:
    if isinstance(persona, Mapping):
        return persona.get(key, default)
    return getattr(persona, key, default)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _tools_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return ""


def _available_agents_block(agents: dict[str, dict[str, Any]]) -> str:
    if not agents:
        return ""
    lines = ["<available_agents>"]
    for name in sorted(agents):
        persona = agents[name]
        lines.append("  <agent>")
        lines.append(f"    <name>{escape(name)}</name>")
        description = _text(_field(persona, "description"))
        if description:
            lines.append(f"    <description>{escape(description)}</description>")
        tools = _tools_str(_field(persona, "tools"))
        if tools:
            lines.append(f"    <tools>{escape(tools)}</tools>")
        lines.append("  </agent>")
    lines.append("</available_agents>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Atom install
# ---------------------------------------------------------------------------


class _ClaudeAgentsRuntime:
    def __init__(self, api: ExtensionAPI, config: ClaudeAgentsConfig) -> None:
        self._api = api
        self._inherit_claude = config.inherit_claude
        self._extra_paths = [
            expand_path_from_cwd(path, api.cwd) / "agents"
            for path in config.extra_paths
            if path.strip()
        ]
        self._agents: dict[str, dict[str, Any]] = {}
        self._cached_block = ""

    def install(self) -> None:
        self._api.on(SessionReadyEvent.CHANNEL, self.load)
        self._api.on(BeforeAgentStartEvent.CHANNEL, self.inject)
        self._api.on(ResolveSubagentEvent.CHANNEL, self.resolve)

    async def load(self, _event: SessionReadyEvent) -> None:
        dirs = _discover_dirs(self._api.cwd, self._inherit_claude)
        dirs.extend(self._extra_paths)
        self._agents = _load_agents(dirs)
        self._cached_block = _available_agents_block(self._agents)

    def inject(self, event: BeforeAgentStartEvent) -> None:
        if not self._cached_block:
            return
        existing = event.system or ""
        merged = (
            f"{existing}\n\n{self._cached_block}" if existing else self._cached_block
        )
        event.system = merged

    def resolve(self, payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, ResolveSubagentEvent):
            name = payload.name
        elif isinstance(payload, dict):
            raw_name = payload.get("name")
            if not isinstance(raw_name, str):
                return None
            name = raw_name
        else:
            return None
        persona = self._agents.get(name.strip())
        if persona is None:
            return None
        return {
            "body": persona["body"],
            "tools": None,
            "input_schema": persona["input_schema"],
            "budget_defaults": persona["budget_defaults"],
            "artifact_kinds": persona["artifact_kinds"],
        }


async def install(api: ExtensionAPI, config: ClaudeAgentsConfig) -> None:
    _ClaudeAgentsRuntime(api, config).install()
