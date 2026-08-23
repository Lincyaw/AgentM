"""Importable entrypoints used by process-executor behavior tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agentm.core.abi.messages import ImageContent, TextContent
from agentm.core.abi.tool import ToolResult, ToolTerminate


def echo(args: dict[str, Any]) -> ToolResult:
    text = args.get("text")
    count = args.get("count")
    if not isinstance(text, str) or not isinstance(count, int):
        raise TypeError("echo requires text and count")
    return ToolResult(
        content=[
            TextContent(type="text", text=text),
            ImageContent(type="image", data=b"\x00\x01", mime_type="image/test"),
        ],
        extras={"nested": [count, True, None]},
    )


async def terminate(args: dict[str, Any]) -> ToolTerminate:
    text = args.get("text")
    if not isinstance(text, str):
        raise TypeError("terminate requires text")
    await asyncio.sleep(0)
    return ToolTerminate(
        result=ToolResult(
            content=[TextContent(type="text", text=text)],
        ),
        reason="test:complete",
    )


async def wait_forever(args: dict[str, Any]) -> ToolResult:
    started_path = args.get("started_path")
    if not isinstance(started_path, str):
        raise TypeError("wait_forever requires started_path")
    Path(started_path).write_text("started", encoding="utf-8")
    await asyncio.sleep(3600)
    raise AssertionError("unreachable")


__all__ = ["echo", "terminate", "wait_forever"]
