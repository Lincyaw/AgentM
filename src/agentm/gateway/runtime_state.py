"""Gateway session-state read model and snapshot delivery."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

from loguru import logger

from agentm.gateway.approval import ApprovalManager
from agentm.gateway.chat_session_map import ChatSessionMap
from agentm.gateway.child_registry import ChildSessionRegistry
from agentm.gateway.session_manager import SessionManager

type SessionPhase = Literal[
    "idle",
    "running",
    "waiting_interaction",
    "interrupting",
    "errored",
    "unknown",
]


@dataclass(slots=True)
class _SessionSnapshot:
    phase: SessionPhase = "idle"
    active_turn_id: str | None = None
    tool_names: list[str] = field(default_factory=list)
    command_names: list[str] = field(default_factory=list)
    pending_interactions: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    last_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "active_turn_id": self.active_turn_id,
            "tool_names": list(self.tool_names),
            "command_names": list(self.command_names),
            "pending_interactions": list(self.pending_interactions),
            "children": list(self.children),
            "last_error": self.last_error,
        }


class GatewaySessionState:
    """Gateway-owned read model for routes, snapshots, commands, and counts."""

    def __init__(
        self,
        *,
        chat_map: ChatSessionMap,
        sessions: SessionManager,
        approval: ApprovalManager,
        child_registry: ChildSessionRegistry,
        active_model_name: Callable[[str], str],
        active_scenario_name: Callable[[str], str],
        outbound_sink: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        self._chat_map = chat_map
        self._sessions = sessions
        self._approval = approval
        self._child_registry = child_registry
        self._active_model_name = active_model_name
        self._active_scenario_name = active_scenario_name
        self._outbound_sink = outbound_sink
        self._session_commands: dict[str, set[str]] = {}
        self._session_routes: dict[str, tuple[str, str, str | None]] = {}
        self._snapshots: dict[str, _SessionSnapshot] = {}
        self._turn_counts: dict[str, int] = {}
        self._snapshot_tasks: dict[str, asyncio.Task[None]] = {}

    def record_route(self, session_key: str, body: dict[str, Any]) -> None:
        channel = str(body.get("channel") or "")
        chat_id = str(body.get("chat_id") or "")
        if not channel or not chat_id:
            return
        thread_id = body.get("thread_id")
        if thread_id is not None and not isinstance(thread_id, str):
            thread_id = str(thread_id)
        self._session_routes[session_key] = (channel, chat_id, thread_id)

    def snapshot_for(self, session_key: str) -> _SessionSnapshot:
        return self._snapshots.setdefault(session_key, _SessionSnapshot())

    def _set_children_snapshot(self, snapshot: _SessionSnapshot) -> None:
        snapshot.children = sorted(self._child_registry.ids())

    def _set_pending_interactions_snapshot(
        self, session_key: str, snapshot: _SessionSnapshot
    ) -> None:
        snapshot.pending_interactions = sorted(
            self._approval.pending_for_session(session_key)
        )

    def _derive_phase(self, snapshot: _SessionSnapshot) -> SessionPhase:
        if snapshot.last_error:
            return "errored"
        if snapshot.pending_interactions:
            return "waiting_interaction"
        if snapshot.active_turn_id:
            return "running"
        if snapshot.phase in {"interrupting", "unknown"}:
            return snapshot.phase
        return "idle"

    def update_snapshot(
        self, session_key: str, kind: str, metadata: dict[str, Any]
    ) -> bool:
        snapshot = self.snapshot_for(session_key)
        before = snapshot.as_dict()

        if kind == "session_ready":
            tools = metadata.get("tool_names")
            if isinstance(tools, list):
                snapshot.tool_names = sorted(
                    {str(tool) for tool in tools if isinstance(tool, str)}
                )
            commands = metadata.get("command_names")
            if isinstance(commands, list):
                snapshot.command_names = sorted(
                    {str(command) for command in commands if isinstance(command, str)}
                )
        elif kind == "turn_start":
            turn_id = metadata.get("turn_id")
            if isinstance(turn_id, str):
                snapshot.active_turn_id = turn_id
            snapshot.phase = "running"
            snapshot.last_error = None
        elif kind == "turn_end":
            snapshot.active_turn_id = None
        elif kind == "agent_end":
            cause = metadata.get("cause")
            if isinstance(cause, str) and cause:
                lowered = cause.lower()
                if lowered in {
                    "none",
                    "normal",
                    "end_turn",
                    "modelendturn",
                    "toolterminated",
                    "cancel",
                    "cancelled",
                    "keyboardinterrupt",
                }:
                    snapshot.last_error = None
                else:
                    snapshot.last_error = cause
            snapshot.active_turn_id = None
        elif kind in {"child_start", "child_end"}:
            self._set_children_snapshot(snapshot)
        elif kind == "approval_request":
            approval_id = metadata.get("approval_id")
            if isinstance(approval_id, str) and approval_id:
                pending = self._approval.pending_for_session(session_key)
                if approval_id not in pending:
                    pending.append(approval_id)
                snapshot.pending_interactions = sorted(set(pending))
            snapshot.phase = "waiting_interaction"
        elif kind == "approval_resolved":
            approval_id = metadata.get("approval_id")
            if isinstance(approval_id, str) and approval_id:
                snapshot.pending_interactions = sorted(
                    interaction
                    for interaction in snapshot.pending_interactions
                    if interaction != approval_id
                )
            else:
                self._set_pending_interactions_snapshot(session_key, snapshot)
        else:
            return False

        if kind in {
            "child_start",
            "child_end",
            "approval_request",
            "approval_resolved",
        }:
            self._set_children_snapshot(snapshot)
            self._set_pending_interactions_snapshot(session_key, snapshot)

        snapshot.phase = self._derive_phase(snapshot)

        if kind == "approval_request" and not snapshot.pending_interactions:
            self._set_pending_interactions_snapshot(session_key, snapshot)

        return snapshot.as_dict() != before

    async def emit_snapshot(self, session_key: str) -> None:
        route = self._session_routes.get(session_key)
        if route is None:
            return
        snapshot = self.snapshot_for(session_key)
        self._set_children_snapshot(snapshot)
        self._set_pending_interactions_snapshot(session_key, snapshot)
        snapshot.phase = self._derive_phase(snapshot)

        channel, chat_id, thread_id = route
        await self._outbound_sink(
            {
                "channel": channel,
                "chat_id": chat_id,
                "content": "",
                **({"thread_id": thread_id} if thread_id is not None else {}),
                "metadata": {
                    "kind": "session_snapshot",
                    "session_id": self._sessions.session_id(session_key),
                    **snapshot.as_dict(),
                },
                "_session_key": session_key,
            }
        )

    def schedule_snapshot(self, session_key: str) -> None:
        """Coalesce snapshot work and retain tasks until completion."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        current = self._snapshot_tasks.get(session_key)
        if current is not None and not current.done():
            return
        task = loop.create_task(
            self.emit_snapshot(session_key),
            name=f"agentm-gateway-snapshot:{session_key}",
        )
        self._snapshot_tasks[session_key] = task
        task.add_done_callback(partial(self._finish_snapshot_task, session_key))

    def _finish_snapshot_task(
        self,
        session_key: str,
        task: asyncio.Task[None],
    ) -> None:
        if self._snapshot_tasks.get(session_key) is task:
            self._snapshot_tasks.pop(session_key, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("gateway snapshot delivery failed for {}", session_key)

    def remember_commands(self, session_key: str, meta: dict[str, Any]) -> None:
        names = meta.get("command_names")
        if not isinstance(names, list):
            return
        self._session_commands[session_key] = {
            name for name in names if isinstance(name, str)
        }

    def remember_live_commands(self, session_key: str, sess: Any) -> set[str] | None:
        names = getattr(sess, "command_names", None)
        if not isinstance(names, list):
            return None
        known = {name for name in names if isinstance(name, str)}
        self._session_commands[session_key] = known
        return known

    def command_names(self, session_key: str) -> set[str] | None:
        return self._session_commands.get(session_key)

    def list_session_commands(self, session_key: str) -> list[str]:
        return sorted(self._session_commands.get(session_key, set()))

    def increment_turn_count(self, session_key: str) -> None:
        self._turn_counts[session_key] = self._turn_counts.get(session_key, 0) + 1

    def reset_turn_count(self, session_key: str) -> None:
        self._turn_counts.pop(session_key, None)

    def clear_runtime_session(self, session_key: str) -> None:
        self._turn_counts.pop(session_key, None)
        self._snapshots.pop(session_key, None)
        self._session_routes.pop(session_key, None)
        task = self._snapshot_tasks.pop(session_key, None)
        if task is not None:
            task.cancel()

    def forget_session(self, session_key: str) -> None:
        self._session_commands.pop(session_key, None)
        self.clear_runtime_session(session_key)

    def route_stats(self, session_key: str) -> dict[str, Any]:
        return {
            "session_id": self._sessions.session_id(session_key),
            "turn_count": self._turn_counts.get(session_key, 0),
            "pending_approvals": self._approval.pending_count,
        }

    def debug_state(
        self,
        session_key: str,
        *,
        inflight_count: int,
        outbox_ready: bool,
    ) -> dict[str, Any]:
        current_session_route = self._session_routes.get(session_key)
        child_ids = self._child_registry.ids()
        chat_routes = self._chat_map.snapshot()
        chat_metadata = self._chat_map.snapshot_metadata()

        sessions_state: dict[str, Any] = {}
        tracked_keys = (
            set(self._session_routes)
            | set(self._snapshots)
            | set(chat_routes)
        )
        for key in sorted(tracked_keys):
            route = self._session_routes.get(key)
            sessions_state[key] = {
                "session_id": self._sessions.session_id(key),
                "route": (
                    {
                        "channel": route[0],
                        "chat_id": route[1],
                        "thread_id": route[2],
                    }
                    if route is not None
                    else None
                ),
                "snapshot": self.snapshot_for(key).as_dict(),
                "model": self._active_model_name(key),
                "scenario": self._active_scenario_name(key),
                "chat_session_map": chat_routes.get(key),
                "metadata": chat_metadata.get(key, {}),
            }

        return {
            "session_key": session_key,
            "session": {
                "session_id": self._sessions.session_id(session_key),
                "route": (
                    {
                        "channel": current_session_route[0],
                        "chat_id": current_session_route[1],
                        "thread_id": current_session_route[2],
                    }
                    if current_session_route is not None
                    else None
                ),
                "snapshot": self.snapshot_for(session_key).as_dict(),
                "model": self._active_model_name(session_key),
                "scenario": self._active_scenario_name(session_key),
                "command_names": self.list_session_commands(session_key),
                "turn_count": self._turn_counts.get(session_key, 0),
                "pending_approvals": self._approval.pending_for_session(session_key),
                "child_sessions": child_ids,
            },
            "sessions": sessions_state,
            "global": {
                "inflight_tasks": inflight_count,
                "tracked_sessions": len(self._session_routes),
                "outbox_ready": outbox_ready,
                "total_pending_approvals": self._approval.pending_count,
            },
        }


__all__ = ["GatewaySessionState", "SessionPhase"]
