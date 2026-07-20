"""Glob-based tool-call allow/deny gate.

Each list entry is an :mod:`fnmatch`-style pattern; literal tool names
work unchanged because they contain no wildcard characters.

Semantics:

- **Only ``allow`` set** -- positive list: a name passes iff it matches.
- **Only ``deny`` set** -- negative list: a name is blocked iff it matches.
- **Both set** -- ``deny`` is the broad ban, ``allow`` carves exceptions.
  Names matching neither pass freely.
"""

from __future__ import annotations

import fnmatch

from pydantic import BaseModel

from agentm.core.abi import AtomAPI, AtomInstallPriority, JsonValue, ToolCallEvent
from agentm.extensions import ExtensionManifest


class PermissionConfig(BaseModel):
    allow: list[str] = []
    deny: list[str] = []


MANIFEST = ExtensionManifest(
    name="permission",
    description="Block or allow tool calls via allow/deny lists (glob-aware).",
    registers=("event:tool_call",),
    config_schema=PermissionConfig,
    requires=(),
    priority=AtomInstallPriority.POLICY,
)


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


class _PermissionRuntime:
    def __init__(self, allow: tuple[str, ...], deny: tuple[str, ...]) -> None:
        self._allow = allow
        self._deny = deny
        self._both = bool(allow) and bool(deny)

    def on_tool_call(self, event: ToolCallEvent) -> dict[str, JsonValue] | None:
        name = event.tool_name
        allow_hit = bool(self._allow) and _matches_any(name, self._allow)
        deny_hit = bool(self._deny) and _matches_any(name, self._deny)

        if self._both:
            if deny_hit and not allow_hit:
                return {"block": True, "reason": f"tool '{name}' denied by denylist"}
            return None

        if deny_hit:
            return {"block": True, "reason": f"tool '{name}' denied by denylist"}
        if self._allow and not allow_hit:
            return {"block": True, "reason": f"tool '{name}' is not in allowlist"}
        return None


def install(api: AtomAPI, config: PermissionConfig) -> None:
    allow = tuple(config.allow)
    deny = tuple(config.deny)
    if not allow and not deny:
        return
    rt = _PermissionRuntime(allow, deny)
    api.on(ToolCallEvent.CHANNEL, rt.on_tool_call)
