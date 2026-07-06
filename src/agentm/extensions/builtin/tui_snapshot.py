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
User paths such as ``~/agentm-tui-dump.txt`` and ``$TMPDIR/dump.txt`` are
expanded consistently for the config value, environment override, and per-call
path argument.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.core.lib import expand_path
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
    requires=(),
    api_version=1,
    tier=1,
)

_DEFAULT_DUMP_PATH: Final[str] = "/tmp/agentm-tui-dump.txt"

# CSI sequences (colours, cursor moves) + OSC 8 hyperlinks the TUI emits.
_CSI_RE: Final = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_OSC_RE: Final = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


class _TuiSnapshotArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str | None = Field(
        default=None,
        description=(
            f"Dump file to read. Defaults to $AGENTM_TUI_DUMP or {_DEFAULT_DUMP_PATH}."
        ),
    )
    raw: bool = Field(
        default=False,
        description=(
            "Keep ANSI escape codes (colours/layout). Default strips "
            "them for clean plain text."
        ),
    )
    tail: int | None = Field(
        default=None,
        description="Return only the last N lines. Omit for the whole frame.",
    )


def _strip_ansi(text: str) -> str:
    return _CSI_RE.sub("", _OSC_RE.sub("", text))


def _resolve_path(configured: str | None, arg: Any) -> Path:
    if isinstance(arg, str) and arg:
        return expand_path(arg)
    if configured:
        return expand_path(configured)
    env_path = os.environ.get("AGENTM_TUI_DUMP")
    if env_path:
        return expand_path(env_path)
    return Path(_DEFAULT_DUMP_PATH)


class _TuiSnapshotRuntime:
    def __init__(self, api: ExtensionAPI, config: TuiSnapshotConfig) -> None:
        self._api = api
        self._configured_path = config.dump_path

    def install(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="tui_snapshot",
                description=(
                    "Read the latest terminal-client screen dump so you can see "
                    "the TUI the user sees. The dump is written when the user "
                    "runs /dump in the client. Use this to diagnose rendering, "
                    "layout, or interaction bugs you cannot infer from the "
                    "conversation alone. The result starts with a one-line "
                    "freshness header (dump path + age in seconds, plus a "
                    "stale warning past ~2 min) — it is not part of the "
                    "screen content."
                ),
                parameters=_TuiSnapshotArgs,
                fn=self.execute,
                metadata={"file_op": "read"},
            )
        )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        path = _resolve_path(self._configured_path, args.get("path"))
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


def install(api: ExtensionAPI, config: TuiSnapshotConfig) -> None:
    _TuiSnapshotRuntime(api, config).install()


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)


__all__ = (
    "MANIFEST",
    "TuiSnapshotConfig",
    "install",
)
