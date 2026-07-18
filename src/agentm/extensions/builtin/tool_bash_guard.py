"""Block destructive file-editing shell commands (sed -i, awk inplace).

When agents fail to use the edit tool, they fall back to shell commands
like ``sed -i`` that bypass all safety checks and frequently corrupt
files. This extension intercepts ``bash`` tool calls and blocks known
destructive patterns, directing the agent to use the edit tool instead.
"""

from __future__ import annotations

import re
from typing import Any, Final

from pydantic import BaseModel

from agentm.core.abi import ToolCallEvent
from agentm.extensions import ExtensionManifest


class ToolBashGuardConfig(BaseModel):
    model_config = {"extra": "allow"}


MANIFEST = ExtensionManifest(
    name="tool_bash_guard",
    description="Block destructive file-editing shell commands (sed -i, awk inplace).",
    registers=("event:tool_call",),
    config_schema=ToolBashGuardConfig,
    requires=(),
)

_BLOCKED: Final[list[tuple[re.Pattern[str], str]]] = [
    (
        re.compile(r"\bsed\b[^|;]*\s-i(?:\s|$|[.'\"])"),
        "Do not use `sed -i` to edit files. Use the edit tool instead — "
        "it supports both string replacement (old_string/new_string) and "
        "line-range replacement (start_line/end_line).",
    ),
    (
        re.compile(r"\bsed\b[^|;]*--in-place"),
        "Do not use `sed --in-place` to edit files. Use the edit tool instead.",
    ),
    (
        re.compile(r"\bawk\b[^|;]*-i\s+inplace"),
        "Do not use `awk -i inplace` to edit files. Use the edit tool instead.",
    ),
]


class _ToolBashGuardRuntime:
    def __init__(self, session: Any) -> None:
        self._session = session

    def install(self) -> None:
        self._session.bus.on(ToolCallEvent.CHANNEL, self.on_tool_call)

    def on_tool_call(self, event: ToolCallEvent) -> dict[str, Any] | None:
        if event.tool_name != "bash":
            return None
        cmd = event.args.get("command", "")
        if not isinstance(cmd, str):
            return None
        for pattern, reason in _BLOCKED:
            if pattern.search(cmd):
                return {"block": True, "reason": reason}
        return None


def install(session: Any, config: ToolBashGuardConfig) -> None:
    del config
    _ToolBashGuardRuntime(session).install()
