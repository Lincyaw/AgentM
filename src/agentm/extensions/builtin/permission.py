"""Policy-gate atom for the §7 ``extensions.builtin.permission`` row.

Implements allow/deny filtering on the ``tool_call`` channel. Each list
entry is an :mod:`fnmatch`-style pattern (``*`` / ``?`` / ``[seq]``);
literal tool names work unchanged because they contain no wildcard
characters.

Semantics, by which lists are populated:

- **Only ``allow`` set** — classic positive list: a name passes iff it
  matches some ``allow`` pattern; everything else is blocked.
- **Only ``deny`` set** — negative list: a name is blocked iff it
  matches some ``deny`` pattern; everything else passes.
- **Both set** — ``deny`` is the broad ban, ``allow`` is the exception
  list. A name is blocked iff it matches ``deny`` *and* matches nothing
  in ``allow``. Names matching neither list pass freely. This is the
  shape used to deny-by-default an entire namespace while permitting a
  specific sub-namespace::

      deny:  ["mcp__*"]            # broad ban over the MCP namespace
      allow: ["mcp__fetch__*"]     # carve-out for one server

  Native AgentM tools (``read`` / ``edit`` / ``bash`` / ...) match
  neither pattern and are unaffected.
"""

from __future__ import annotations

import fnmatch
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import ToolCallEvent
from agentm.extensions import ExtensionManifest


class PermissionConfig(BaseModel):
    allow: list[str] = []
    deny: list[str] = []


MANIFEST = ExtensionManifest(
    name="permission",
    description="Block or allow tool calls via allow/deny lists (glob-aware).",
    registers=("event:tool_call",),
    config_schema=PermissionConfig,
    requires=(),  # Leaf policy atom: can guard absent, present, or future tools.
    tier=2,
)


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    """Return True if ``name`` matches any fnmatch pattern in ``patterns``.

    A pattern without wildcard metacharacters reduces to literal equality,
    so callers that pass exact tool names keep their previous semantics.
    """

    for pattern in patterns:
        if fnmatch.fnmatchcase(name, pattern):
            return True
    return False


class _PermissionRuntime:
    def __init__(self, session: Any, config: PermissionConfig) -> None:
        self._session = session
        self._allow = tuple(config.allow)
        self._deny = tuple(config.deny)
        self._both_set = bool(self._allow) and bool(self._deny)

    def active(self) -> bool:
        return bool(self._allow or self._deny)

    def install(self) -> None:
        self._session.bus.on(ToolCallEvent.CHANNEL, self.on_tool_call)

    def on_tool_call(self, event: ToolCallEvent) -> dict[str, Any] | None:
        name = event.tool_name
        allow_hit = bool(self._allow) and _matches_any(name, self._allow)
        deny_hit = bool(self._deny) and _matches_any(name, self._deny)

        if self._both_set:
            # ``allow`` carves exceptions out of ``deny``; non-matches pass.
            if deny_hit and not allow_hit:
                return {
                    "block": True,
                    "reason": f"tool '{name}' denied by denylist",
                }
            return None

        if deny_hit:
            return {
                "block": True,
                "reason": f"tool '{name}' denied by denylist",
            }
        if self._allow and not allow_hit:
            return {
                "block": True,
                "reason": f"tool '{name}' is not in allowlist",
            }
        return None


def install(session: Any, config: PermissionConfig) -> None:
    runtime = _PermissionRuntime(session, config)
    if not runtime.active():
        return
    runtime.install()
