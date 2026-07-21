# code-health: ignore-file[AM025] -- validates tool-call JSON values at the event boundary
"""Single owner for tool catalog visibility and call-time policy gates."""

from __future__ import annotations

import fnmatch
import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    BeforeSendEvent,
    JsonValue,
    Tool,
    ToolCallEvent,
)
from agentm.extensions import ExtensionManifest


class ToolPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    protect_file_mutations: bool = False


MANIFEST = ExtensionManifest(
    name="tool_policy",
    description=(
        "Apply one allow/deny policy to both the visible tool catalog and "
        "call-time enforcement, with optional shell mutation protection."
    ),
    registers=("event:before_send", "event:tool_call"),
    config_schema=ToolPolicyConfig,
    requires=(),
    priority=AtomInstallPriority.POLICY,
)

_BLOCKED_FILE_MUTATIONS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (
        re.compile(r"\bsed\b[^|;]*\s-i(?:\s|$|[.'\"])"),
        "Do not use `sed -i` to edit files. Use the edit tool instead.",
    ),
    (
        re.compile(r"\bsed\b[^|;]*--in-place"),
        "Do not use `sed --in-place` to edit files. Use the edit tool instead.",
    ),
    (
        re.compile(r"\bawk\b[^|;]*-i\s+inplace"),
        "Do not use `awk -i inplace` to edit files. Use the edit tool instead.",
    ),
)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


class _ToolPolicyRuntime:
    def __init__(self, config: ToolPolicyConfig) -> None:
        self._allow = tuple(config.allow)
        self._deny = tuple(config.deny)
        self._protect_file_mutations = config.protect_file_mutations

    def visible(self, name: str) -> bool:
        allow_hit = _matches(name, self._allow)
        deny_hit = _matches(name, self._deny)
        if self._allow and self._deny:
            return not deny_hit or allow_hit
        if self._deny and deny_hit:
            return False
        return not self._allow or allow_hit

    def before_send(self, event: BeforeSendEvent) -> dict[str, list[Tool]]:
        return {"tools": [tool for tool in event.tools if self.visible(tool.name)]}

    def tool_call(self, event: ToolCallEvent) -> dict[str, JsonValue] | None:
        if not self.visible(event.tool_name):
            return {
                "block": True,
                "reason": f"tool '{event.tool_name}' is not allowed",
            }
        if not self._protect_file_mutations or event.tool_name != "bash":
            return None
        command = event.args.get("cmd") or event.args.get("command", "")
        if not isinstance(command, str):
            return None
        for pattern, reason in _BLOCKED_FILE_MUTATIONS:
            if pattern.search(command):
                return {"block": True, "reason": reason}
        return None


def install(api: AtomAPI, config: ToolPolicyConfig) -> None:
    runtime = _ToolPolicyRuntime(config)
    api.on(BeforeSendEvent.CHANNEL, runtime.before_send)
    api.on(ToolCallEvent.CHANNEL, runtime.tool_call)


__all__ = ("MANIFEST", "ToolPolicyConfig", "install")
