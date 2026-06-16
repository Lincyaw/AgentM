"""``MCPTool`` — adapter from the kernel ``Tool`` Protocol to MCP ``tools/call``.

Each instance closes over a live :class:`ClientSession` and the *remote*
tool name (the un-namespaced one the MCP server knows). ``Tool.name`` is
the namespaced form the LLM sees.

See ``.claude/designs/mcp-integration.md`` §4.3.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Final

from loguru import logger

from agentm.core.abi import ImageContent, TextContent, ToolResult

from .client import _SessionLike


@dataclass(slots=True)
class MCPTool:
    """Bridges one remote MCP tool into the AgentM tool catalog."""

    name: str
    description: str
    parameters: dict[str, Any]
    _server: str
    _remote_name: str
    _session: _SessionLike
    _timeout: float | None = None
    # ``extras`` kept on a per-instance basis so ``execute`` doesn't allocate
    # a fresh dict each call but the snapshot of identifying metadata
    # remains tied to the registered tool, not the per-call args.
    _extras: dict[str, Any] = field(default_factory=dict)

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
    ) -> ToolResult:
        """Race the MCP ``tools/call`` RPC against cancellation + timeout.

        Cancellation is best-effort: we cancel the in-flight ``call_tool``
        task and surface an error ``ToolResult``. The MCP SDK itself owns
        sending ``notifications/cancelled`` upstream when its anyio scope
        unwinds. ``_timeout`` (configured via ``timeout_seconds``) bounds
        the wait so a hung server cannot stall the agent loop indefinitely.
        """

        rpc_task = asyncio.create_task(
            self._session.call_tool(self._remote_name, args)
        )
        waiters: set[asyncio.Task[Any]] = {rpc_task}
        cancel_task: asyncio.Task[Any] | None = None
        if signal is not None:
            cancel_task = asyncio.create_task(signal.wait())
            waiters.add(cancel_task)

        try:
            done, _pending = await asyncio.wait(
                waiters,
                timeout=self._timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if rpc_task in done:
                if cancel_task is not None:
                    cancel_task.cancel()
                raw = rpc_task.result()
            else:
                # Either the signal fired or the timeout elapsed; cancel
                # the in-flight RPC either way.
                timed_out = not done  # asyncio.wait returns empty done on timeout
                rpc_task.cancel()
                try:
                    await rpc_task
                except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
                    # Draining the cancelled in-flight RPC after timeout/signal.
                    logger.debug("mcp_bridge: cancelled RPC drain raised: {}", exc)
                if cancel_task is not None and not cancel_task.done():
                    cancel_task.cancel()
                reason = (
                    f"timed out after {self._timeout}s"
                    if timed_out
                    else "cancelled by session signal"
                )
                return ToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"mcp_bridge: tool {self.name!r} {reason}.",
                        )
                    ],
                    is_error=True,
                    extras=dict(self._extras),
                )
        except Exception as exc:
            # Transport / protocol failure. Surface as is_error=True so
            # the loop continues; the loud raise-on-install contract only
            # applies to the snapshot phase.
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"mcp_bridge: {self.name!r} failed: {exc!r}",
                    )
                ],
                is_error=True,
                extras=dict(self._extras),
            )

        return ToolResult(
            content=_translate_content_blocks(getattr(raw, "content", []) or []),
            is_error=bool(getattr(raw, "isError", False)),
            extras=dict(self._extras),
        )


def _translate_content_blocks(
    blocks: list[Any],
) -> list[TextContent | ImageContent]:
    """Map MCP content blocks onto kernel ``TextContent`` / ``ImageContent``.

    Unknown block types are flattened to a textual ``repr`` so the LLM at
    least sees *something*; this preserves the principle that a tool result
    never silently disappears. MCP's ``EmbeddedResource`` and ``ResourceLink``
    blocks (out of scope for v0) fall through this path.
    """

    out: list[TextContent | ImageContent] = []
    for block in blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "")
            out.append(TextContent(type="text", text=str(text)))
            continue
        if btype == "image":
            data = getattr(block, "data", b"") or b""
            mime = getattr(block, "mimeType", None) or "application/octet-stream"
            # MCP ships image data as base64 string; decode if needed.
            if isinstance(data, str):
                import base64

                try:
                    decoded = base64.b64decode(data)
                except Exception:
                    decoded = data.encode("utf-8", errors="replace")
            else:
                decoded = bytes(data)
            out.append(ImageContent(type="image", data=decoded, mime_type=str(mime)))
            continue
        out.append(TextContent(type="text", text=repr(block)))
    return out


__all__: Final = ["MCPTool", "_translate_content_blocks"]
