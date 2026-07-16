"""Non-conversational controls handled by the gateway runtime."""

from __future__ import annotations

from typing import Any

from loguru import logger

from agentm.core.abi import ToolContinue, ToolResult, ToolTerminate
from agentm.core.abi.tool_executor import execute_tool_call
from agentm.gateway.wire import InboundBody


def push_user_input(inbox: Any, body: InboundBody, content: str) -> None:
    """Deliver a user input while preserving the wire request dedup key."""
    from agentm.core.runtime.session_inbox import InboxItem

    request_id = (
        body.request_id
        if body.request_id is not None
        else (
            body.raw.get("request_id")
            if isinstance(body.raw, dict) and body.raw.get("request_id") is not None
            else None
        )
    )
    if request_id is None:
        inbox.push(InboxItem(source="user", payload=content))
        return
    inbox.push(
        InboxItem(
            source="user",
            payload=content,
            dedup_key=str(request_id),
        )
    )


def apply_thinking_level(session: Any, body: InboundBody) -> None:
    """Apply an optional per-input thinking level to a live session."""
    thinking = body.thinking
    if thinking is None:
        return
    setter = getattr(session, "set_thinking_level", None)
    if callable(setter):
        setter(thinking)


class GatewayControlMixin:
    """Control-plane behavior shared by the concrete gateway runtime."""

    _sessions: Any
    _state: Any
    _overrides: Any
    _approval: Any
    _child_registry: Any
    _send_error: Any
    _emit_outbound: Any
    _get_or_create_session_for_input: Any

    def _interrupt_session(self, session_key: str) -> None:
        sess = self._sessions.get(session_key)
        if sess is None:
            return
        status_fn = getattr(sess, "status", None)
        if callable(status_fn):
            try:
                status = status_fn()
            except Exception as exc:
                logger.debug("interrupt: session status probe failed: {}", exc)
                status = {}
            if isinstance(status, dict) and status.get("phase") != "running":
                return
        sess.interrupt()
        snapshot = self._state.snapshot_for(session_key)
        if snapshot.phase != "interrupting":
            snapshot.phase = "interrupting"
        snapshot.active_turn_id = None
        if not snapshot.pending_interactions:
            self._state.schedule_snapshot(session_key)

    def _steer_session(self, session_key: str, body: InboundBody) -> None:
        """Inject a user message mid-turn without waiting for turn end."""
        sess = self._sessions.get(session_key)
        if sess is None:
            return
        content = body.content or ""
        if not content.strip():
            return
        apply_thinking_level(sess, body)
        sess.send_user_message(content)

    async def _cancel_background_task(
        self, session_key: str, body: InboundBody
    ) -> None:
        """Cancel a background task without creating a conversational turn."""
        task_id = (body.task_id or "").strip()
        if not task_id:
            await self._send_error(
                session_key,
                body,
                ValueError("cancel_background requires task_id"),
            )
            return
        sess = self._sessions.get(session_key)
        if sess is None:
            await self._send_error(
                session_key,
                body,
                RuntimeError("no live session for background task"),
            )
            return
        cancel_tool = next(
            (tool for tool in sess.tools if tool.name == "cancel_background"),
            None,
        )
        if cancel_tool is None:
            await self._send_error(
                session_key,
                body,
                RuntimeError("cancel_background tool is unavailable"),
            )
            return
        try:
            result = await execute_tool_call(
                cancel_tool,
                {"task_id": task_id},
                signal=None,
            )
        except Exception as exc:
            logger.exception("cancel_background failed for {}", task_id)
            await self._send_error(session_key, body, exc)
            return
        if isinstance(result, ToolContinue | ToolTerminate):
            tool_result = result.result
        elif isinstance(result, ToolResult):
            tool_result = result
        else:
            tool_result = None
        if tool_result is not None and tool_result.is_error:
            await self._send_error(
                session_key,
                body,
                RuntimeError(f"cancel_background rejected task {task_id}"),
            )
            return
        await self._state.emit_snapshot(session_key)

    async def _switch_model_control(
        self, session_key: str, body: InboundBody
    ) -> None:
        """Switch models without rendering a slash-command turn."""
        name = (body.content or "").strip()
        if not name:
            await self._send_error(
                session_key,
                body,
                ValueError("switch_model requires content"),
            )
            return
        ok, message = await self._overrides.switch_model(
            session_key,
            name,
            sessions=self._sessions,
            state=self._state,
        )
        if not ok:
            await self._send_error(
                session_key,
                body,
                RuntimeError(f"could not switch model to '{name}': {message}"),
            )
            return

        try:
            from agentm.core.lib.user_config import load_user_config

            models = list(load_user_config().models.keys())
        except Exception:
            logger.exception("switch_model: failed to read model profiles")
            models = []
        await self._emit_outbound(
            {
                "channel": body.channel,
                "chat_id": body.chat_id,
                "content": "",
                "thread_id": body.thread_id,
                "metadata": {
                    "kind": "session_ready",
                    "model": message,
                    "models": models,
                },
                "_session_key": session_key,
            }
        )

    async def _set_config_control(
        self,
        session_key: str,
        scenario: str | None,
        body: InboundBody,
    ) -> None:
        """Apply a non-conversational session config change."""
        raw = body.raw or {}
        key = str(raw.get("key") or "").strip()
        if key == "permission_rule":
            await self._set_permission_rule_control(session_key, body, raw)
            return
        if key != "auto_compact":
            await self._send_error(
                session_key,
                body,
                ValueError(f"unsupported config key {key!r}"),
            )
            return
        enabled = raw.get("enabled")
        if not isinstance(enabled, bool):
            await self._send_error(
                session_key,
                body,
                ValueError("auto_compact requires boolean enabled"),
            )
            return

        sess = await self._get_or_create_session_for_input(
            session_key,
            scenario,
            body,
        )
        control = sess.get_service("llm_compaction.control")
        setter = getattr(control, "set_enabled", None)
        if control is None or not callable(setter):
            await self._send_error(
                session_key,
                body,
                RuntimeError("auto_compact is not available in this session"),
            )
            return

        actual = bool(setter(enabled))
        await self._emit_outbound(
            {
                "channel": body.channel,
                "chat_id": body.chat_id,
                "content": "",
                "thread_id": body.thread_id,
                "metadata": {
                    "kind": "config_update",
                    "key": key,
                    "enabled": actual,
                },
                "_session_key": session_key,
            }
        )

    async def _set_permission_rule_control(
        self,
        session_key: str,
        body: InboundBody,
        raw: dict[str, Any],
    ) -> None:
        """Apply a permission rule to the live approval policy."""
        kind = str(raw.get("kind") or "").strip().lower()
        pattern = str(raw.get("pattern") or "").strip()
        if kind not in {"allow", "ask", "deny"}:
            await self._send_error(
                session_key,
                body,
                ValueError("permission_rule kind must be allow, ask, or deny"),
            )
            return
        if not pattern:
            await self._send_error(
                session_key,
                body,
                ValueError("permission_rule requires pattern"),
            )
            return
        if not self._approval.add_session_rule(session_key, kind, pattern):
            await self._send_error(
                session_key,
                body,
                RuntimeError("failed to apply permission rule"),
            )
            return

        await self._emit_outbound(
            {
                "channel": body.channel,
                "chat_id": body.chat_id,
                "content": "",
                "thread_id": body.thread_id,
                "metadata": {
                    "kind": "config_update",
                    "key": "permission_rule",
                    "rule_kind": kind,
                    "pattern": pattern,
                },
                "_session_key": session_key,
            }
        )

    def _interrupt_child(self, child_id: str) -> None:
        child = self._child_registry.get(child_id)
        if child is not None:
            child.interrupt()

    def _deliver_child_input(self, child_id: str, body: InboundBody) -> None:
        """Push a human turn into a live child's inbox."""
        child = self._child_registry.get(child_id)
        if child is None:
            return
        content = body.content or ""
        if not content.strip():
            return
        apply_thinking_level(child, body)
        push_user_input(child.inbox, body, content)

    def _resolve_interaction(self, session_key: str, body: InboundBody) -> None:
        if body.button_value:
            self._approval.resolve(body.button_value, body.sender_id)
            self._state.schedule_snapshot(session_key)


__all__ = [
    "GatewayControlMixin",
    "apply_thinking_level",
    "push_user_input",
]
