"""Wire-frame router: ``metadata.kind`` -> one widget mutation.

The single place wire shapes meet widgets. The gateway streams the full event
taxonomy (see ``wire_driver``); this maps each ``metadata.kind`` to a transcript
mutation, a StatusBar update, or a toast. Unknown kinds are ignored (forward-
compatible). See ``.claude/designs/textual-tui.md`` §5.4.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .widgets import (
    ApprovalBlock,
    AssistantTurn,
    SubagentBlock,
    SystemTurn,
    ToolBlock,
)

if TYPE_CHECKING:
    from .app import AgentMTui

Handler = Callable[[dict[str, Any], dict[str, Any]], Awaitable[None]]

# Control/observability kinds shown as a toast, with the formatter + variant.
# Self-modification (agent-triggered reload) is rendered louder on purpose.
_SELF_MODIFY_TRIGGERS = {"agent", "propose_change_approved"}


class Router:
    def __init__(self, app: AgentMTui) -> None:
        self._app = app
        self._active: AssistantTurn | None = None
        self._tools: dict[str, ToolBlock] = {}
        self._children: dict[str, SubagentBlock] = {}
        self._dispatch: dict[str, Handler] = {
            "turn_start": self._turn_start,
            "stream_text": self._stream_text,
            "stream_thinking": self._stream_thinking,
            "tool_call": self._tool_call,
            "tool_result": self._tool_result,
            "usage": self._usage,
            "child_start": self._child_start,
            "child_end": self._child_end,
            "assistant_text": self._assistant_text,
            "approval_request": self._approval_request,
            "approval_resolved": self._approval_resolved,
            "diagnostic_warning": self._diagnostic,
            "diagnostic_error": self._diagnostic,
            "agent_end": self._agent_end,
            "api_send_user_message": self._injected,
            "session_ready": self._session_ready,
            "api_register": self._api_register,
            "cost_budget_exceeded": self._budget,
            "extension_install": self._control,
            "extension_reload": self._control,
            "extension_unload": self._control,
            "resource_write": self._control,
            "plan_submitted": self._control,
            "after_compact": self._control,
            "command_dispatched": self._control,
        }

    @property
    def active(self) -> AssistantTurn | None:
        return self._active

    async def dispatch(self, body: dict[str, Any]) -> None:
        meta = body.get("metadata")
        meta = meta if isinstance(meta, dict) else {}
        kind = str(meta.get("kind") or "assistant_text")
        fn = self._dispatch.get(kind)
        if fn is not None:
            await fn(body, meta)

    # --- conversation ---------------------------------------------------

    async def _ensure_turn(self) -> AssistantTurn:
        if self._active is None:
            turn = AssistantTurn()
            await self._app.mount_widget(turn)
            self._active = turn
        return self._active

    async def _turn_start(self, body: dict, meta: dict) -> None:
        turn = AssistantTurn()
        await self._app.mount_widget(turn)
        self._active = turn
        self._app.set_phase("thinking")

    async def _stream_text(self, body: dict, meta: dict) -> None:
        (await self._ensure_turn()).append_text(str(body.get("content") or ""))
        self._app.set_phase("streaming")

    async def _stream_thinking(self, body: dict, meta: dict) -> None:
        (await self._ensure_turn()).append_thinking(str(body.get("content") or ""))
        self._app.set_phase("thinking")

    async def _tool_call(self, body: dict, meta: dict) -> None:
        turn = await self._ensure_turn()
        args = meta.get("args")
        block = ToolBlock(
            name=str(meta.get("name") or "tool"),
            args=args if isinstance(args, dict) else {},
        )
        self._tools[str(meta.get("tool_call_id") or "")] = block
        await turn.add_block(block)
        self._app.set_phase("tool")
        self._app.scroll_end()

    async def _tool_result(self, body: dict, meta: dict) -> None:
        block = self._tools.get(str(meta.get("tool_call_id") or ""))
        if block is not None:
            block.set_result(
                ok=bool(meta.get("ok", True)), content=str(body.get("content") or "")
            )
            self._app.scroll_end()

    async def _usage(self, body: dict, meta: dict) -> None:
        self._app.status.tokens_in += int(meta.get("input_tokens") or 0)
        self._app.status.tokens_out += int(meta.get("output_tokens") or 0)
        self._app.show_status()

    async def _child_start(self, body: dict, meta: dict) -> None:
        turn = await self._ensure_turn()
        block = SubagentBlock(purpose=str(meta.get("purpose") or "subagent"))
        self._children[str(meta.get("child_id") or "")] = block
        await turn.add_block(block)
        self._app.set_phase("subagent")
        self._app.scroll_end()

    async def _child_end(self, body: dict, meta: dict) -> None:
        block = self._children.get(str(meta.get("child_id") or ""))
        if block is not None:
            block.finish(error=meta.get("error"))

    async def _assistant_text(self, body: dict, meta: dict) -> None:
        turn = await self._ensure_turn()
        content = str(body.get("content") or "")
        await turn.set_final_text(content)
        self._app.note_assistant_text(content)  # for /copy-last
        self._active = None  # the turn is complete; the next turn_start opens one
        self._app.set_phase("idle")
        self._app.scroll_end()

    async def _agent_end(self, body: dict, meta: dict) -> None:
        self._app.mark_idle()

    # --- approvals ------------------------------------------------------

    async def _approval_request(self, body: dict, meta: dict) -> None:
        buttons = [b for b in (body.get("buttons") or []) if isinstance(b, dict)]
        block = ApprovalBlock(
            content=str(body.get("content") or ""),
            buttons=buttons,
            on_select=self._app.send_button,
        )
        await self._app.mount_widget(block)
        block.focus()
        self._app.scroll_end()

    async def _approval_resolved(self, body: dict, meta: dict) -> None:
        self._app.toast(f"approval {meta.get('decision', 'resolved')}")

    # --- control / observability ---------------------------------------

    async def _diagnostic(self, body: dict, meta: dict) -> None:
        self._app.toast(str(body.get("content") or ""), variant="warn")

    async def _injected(self, body: dict, meta: dict) -> None:
        await self._app.mount_widget(
            SystemTurn(
                str(body.get("content") or ""),
                source=str(meta.get("extension") or "extension"),
            )
        )
        self._app.scroll_end()

    async def _session_ready(self, body: dict, meta: dict) -> None:
        self._app.status.model = str(meta.get("model") or self._app.status.model)
        tools = meta.get("tool_names")
        if isinstance(tools, list):
            for t in tools:
                self._app.catalog.add_tool(str(t))
            self._app.status.tool_count = len(self._app.catalog.tools)
        commands = meta.get("command_names")
        if isinstance(commands, list):
            for c in commands:
                self._app.catalog.add_command(_as_slash(str(c)))
        self._app.show_status()

    async def _api_register(self, body: dict, meta: dict) -> None:
        name = str(meta.get("name") or "")
        if meta.get("reg_kind") == "tool":
            self._app.catalog.add_tool(name)
            self._app.status.tool_count = len(self._app.catalog.tools)
            self._app.show_status()
        elif meta.get("reg_kind") == "command":
            self._app.catalog.add_command(_as_slash(name))

    async def _budget(self, body: dict, meta: dict) -> None:
        self._app.status.budget_exceeded = True
        self._app.catalog.budget = (
            f"exceeded: {meta.get('used')}/{meta.get('limit')} "
            f"{meta.get('currency', '')}".strip()
        )
        self._app.show_status()
        self._app.toast(
            f"budget exceeded: {meta.get('used')}/{meta.get('limit')} "
            f"{meta.get('currency', '')}".strip(),
            variant="warn",
        )

    async def _control(self, body: dict, meta: dict) -> None:
        kind = str(meta.get("kind") or "")
        # Record extension lifecycle into the catalog (backs /extensions),
        # for every phase — not just the ones we toast.
        if kind == "extension_install":
            mod = str(meta.get("module_path") or "")
            err = meta.get("error")
            self._app.catalog.extensions[mod] = (
                f"{meta.get('phase')}: {err}" if err else str(meta.get("phase") or "")
            )
        elif kind in ("extension_reload", "extension_unload"):
            self._app.catalog.extensions[str(meta.get("name") or "")] = kind.split(
                "_", 1
            )[1]
        # Skip the noisy bootstrap install start/end; surface failures + reloads.
        if kind == "extension_install" and meta.get("phase") != "error":
            return
        text, variant = _control_summary(kind, meta)
        if text:
            self._app.toast(text, variant=variant)


def _as_slash(name: str) -> str:
    """Normalise a command name to its ``/name`` invocation form."""
    return name if name.startswith("/") else f"/{name}"


def _control_summary(kind: str, meta: dict[str, Any]) -> tuple[str, str]:
    if kind == "extension_reload":
        self_modify = bool(meta.get("is_self_modify")) or (
            meta.get("trigger") in _SELF_MODIFY_TRIGGERS
        )
        prefix = "★ self-modify" if self_modify else "reload"
        err = f"  ({meta.get('error')})" if meta.get("error") else ""
        return f"{prefix}: {meta.get('name')}{err}", (
            "selfmod" if self_modify else "info"
        )
    if kind == "extension_unload":
        return f"unload: {meta.get('name')}", "info"
    if kind == "extension_install":  # only reaches here on error
        return f"install failed: {meta.get('module_path')}", "warn"
    if kind == "resource_write":
        return f"wrote {meta.get('path')}  ({meta.get('author')})", "info"
    if kind == "plan_submitted":
        return "plan submitted", "info"
    if kind == "after_compact":
        return (
            f"compacted: kept {meta.get('kept')}, dropped {meta.get('discarded')}",
            "info",
        )
    if kind == "command_dispatched":
        return f"/{meta.get('name')}", "info"
    return "", "info"


__all__ = ["Router"]
