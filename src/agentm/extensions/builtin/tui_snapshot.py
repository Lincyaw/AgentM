"""Tool atom: read the latest TUI screen dump for self-debugging.

A chat-client TUI (e.g. ``terminal-go``) writes its rendered frame to a
dump file on ``/dump``. This atom registers a ``tui_snapshot`` tool that
lets the agent read that file back — so the agent can "see what the user
sees" when diagnosing a rendering or interaction bug.

Pure file read with two conveniences over the bare ``read`` tool:
ANSI escape stripping (the model gets clean layout text) and a staleness
header (the model knows how old the frame is, since the dump is
user-triggered). The dump path is ``$AGENTM_TUI_DUMP`` or, failing that,
``/tmp/agentm-tui-dump.txt`` — the same default ``terminal-go`` writes to.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest


class TuiSnapshotConfig(BaseModel):
    model_config = {"extra": "allow"}

    dump_path: str | None = None


MANIFEST = ExtensionManifest(
    name="tui_snapshot",
    description=(
        "Register the tui_snapshot tool: read the latest chat-client TUI "
        "screen dump (written on /dump) so the agent can see the UI."
    ),
    registers=("tool:tui_snapshot",),
    config_schema=TuiSnapshotConfig,
)

_DEFAULT_DUMP_PATH: Final[str] = "/tmp/agentm-tui-dump.txt"

# CSI sequences (colours, cursor moves) + OSC 8 hyperlinks the TUI emits.
_CSI_RE: Final = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_OSC_RE: Final = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": (
                "Dump file to read. Defaults to $AGENTM_TUI_DUMP or "
                f"{_DEFAULT_DUMP_PATH}."
            ),
        },
        "raw": {
            "type": "boolean",
            "default": False,
            "description": (
                "Keep ANSI escape codes (colours/layout). Default strips "
                "them for clean plain text."
            ),
        },
        "tail": {
            "type": "integer",
            "description": (
                "Return only the last N lines. Omit for the whole frame."
            ),
        },
    },
    "required": [],
    "additionalProperties": False,
}


def _strip_ansi(text: str) -> str:
    return _CSI_RE.sub("", _OSC_RE.sub("", text))


def _resolve_path(configured: str | None, arg: Any) -> str:
    if isinstance(arg, str) and arg:
        return arg
    if configured:
        return configured
    return os.environ.get("AGENTM_TUI_DUMP") or _DEFAULT_DUMP_PATH


def install(api: ExtensionAPI, config: TuiSnapshotConfig) -> None:
    configured_path = config.dump_path

    async def _execute(args: dict[str, Any]) -> ToolResult:
        path = Path(_resolve_path(configured_path, args.get("path")))
        if not path.is_file():
            return _error(
                f"No TUI dump at {path}. Ask the user to press /dump (or run "
                "the /dump command) in the terminal client first, then retry."
            )
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            age = max(0.0, time.time() - path.stat().st_mtime)
        except OSError as exc:
            return _error(f"Cannot read TUI dump {path}: {exc}")

        if not bool(args.get("raw", False)):
            text = _strip_ansi(text)
        tail = args.get("tail")
        if isinstance(tail, int) and tail > 0:
            text = "\n".join(text.splitlines()[-tail:])

        header = f"[tui_snapshot {path} — {age:.0f}s old]\n"
        if age > 120:
            header += (
                "[warning: frame is stale; ask the user to /dump again for "
                "the current screen]\n"
            )
        return _ok(header + text)

    api.register_tool(
        FunctionTool(
            name="tui_snapshot",
            description=(
                "Read the latest terminal-client screen dump so you can see "
                "the TUI the user sees. The dump is written when the user "
                "runs /dump in the client. Use this to diagnose rendering, "
                "layout, or interaction bugs you cannot infer from the "
                "conversation alone."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
            metadata={"file_op": "read"},
        )
    )


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
