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

from agentm.core.abi import ToolCallEvent
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="permission",
    description="Block or allow tool calls via allow/deny lists (glob-aware).",
    registers=("event:tool_call",),
    config_schema={
        "type": "object",
        "properties": {
            "allow": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "fnmatch-style patterns; absent or empty means 'allow "
                    "everything not denied'."
                ),
            },
            "deny": {
                "type": "array",
                "items": {"type": "string"},
                "description": "fnmatch-style patterns; takes precedence over allow.",
            },
        },
        "additionalProperties": True,
    },
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


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    allow = tuple(str(p) for p in config.get("allow", []))
    deny = tuple(str(p) for p in config.get("deny", []))
    if not allow and not deny:
        return

    both_set = bool(allow) and bool(deny)

    def _on_tool_call(event: ToolCallEvent) -> dict[str, Any] | None:
        name = event.tool_name
        allow_hit = bool(allow) and _matches_any(name, allow)
        deny_hit = bool(deny) and _matches_any(name, deny)

        if both_set:
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
        if allow and not allow_hit:
            return {
                "block": True,
                "reason": f"tool '{name}' is not in allowlist",
            }
        return None

    api.on(ToolCallEvent.CHANNEL, _on_tool_call)
