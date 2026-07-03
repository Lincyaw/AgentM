"""Gateway runtime — glues WireServer, Router, SessionManager, Approval, Commands.

One :class:`GatewayRuntime` per ``agentm gateway`` process.  Holds the
outbound sink that every component writes into; the sink routes an
``outbound`` envelope to the durable outbox of the peer(s) serving its
channel.

Extracted from ``gateway.cli`` so the runtime logic is testable and
importable without pulling in the Typer CLI layer.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

from agentm.core.abi import GATEWAY_SCHEDULER_SERVICE
from agentm.gateway.approval import ApprovalManager
from agentm.gateway.chat_session_map import ChatSessionMap
from agentm.gateway.child_registry import ChildSessionRegistry
from agentm.gateway.commands import (
    UNKNOWN_REPLY,
    CommandContext,
    CommandInbound,
    CommandRouter,
    parse_invocation,
)
from agentm.gateway.outbox import SqliteOutbox
from agentm.gateway.peer import PeerSession
from agentm.gateway.router import RouterAction, dispatch
from agentm.gateway.scheduler import (
    GatewayScheduler,
    GatewayScheduleStore,
    ScheduledJob,
)
from agentm.gateway.server import WireServer
from agentm.gateway.session_manager import SessionManager
from agentm.gateway.wire import (
    DURABLE_OUTBOUND_KINDS,
    EPHEMERAL_OUTBOUND_KINDS,
    KIND_OUTBOUND,
    WIRE_VERSION,
    Envelope,
    InboundBody,
)


SessionPhase = Literal[
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
            "tool_names": self.tool_names,
            "command_names": self.command_names,
            "pending_interactions": self.pending_interactions,
            "children": self.children,
            "last_error": self.last_error,
        }


def _scenario_payload(info: Any) -> dict[str, str]:
    return {
        "name": str(getattr(info, "name", "")),
        "source": str(getattr(info, "source", "")),
        "manifest_path": str(getattr(info, "manifest_path", "")),
        "description": str(getattr(info, "description", "")),
    }


def route_targets(
    peers: list[PeerSession],
    peer_channels: dict[str, str],
    target_channel: str,
    *,
    session_key: str | None = None,
    peer_session_keys: dict[str, set[str]] | None = None,
) -> list[PeerSession]:
    """Pick the peers an outbound should reach (§3.2).

    Prefer peers that have sent an inbound for this exact ``session_key``.
    This lets multiple terminal clients (each a separate peer) connect to one
    gateway without cross-delivering ``terminal`` traffic for different chats.
    If no session-specific peer is known yet, fall back to the historical
    channel routing so single-client and multi-surface deployments still work.
    """
    if session_key and peer_session_keys:
        session_matching = [
            p for p in peers if session_key in peer_session_keys.get(p.peer_id, set())
        ]
        if session_matching:
            return session_matching
    matching = [p for p in peers if peer_channels.get(p.peer_id) == target_channel]
    return matching if matching else peers


@dataclass(slots=True)
class _BoundGatewaySchedulerService:
    """Session-scoped schedule service exposed to atoms.

    The monitor atom sees only this narrow service, not gateway internals. Route
    metadata comes from the mutable turn context that SessionManager updates on
    each inbound, so schedules created by the model target the same chat/thread
    that is currently driving the session.
    """

    scheduler: GatewayScheduler
    session_key: str
    turn_context: dict[str, Any]
    active_scenario: Callable[[], str]

    def create(
        self,
        *,
        cron: str,
        prompt: str,
        recurring: bool = True,
    ) -> dict[str, Any]:
        channel = str(self.turn_context.get("channel") or "")
        chat_id = str(self.turn_context.get("chat_id") or "")
        if not channel or not chat_id:
            return {"error": "gateway scheduler route context is unavailable"}
        thread_id_raw = self.turn_context.get("thread_id")
        sender_id = str(self.turn_context.get("sender_id") or "")
        scenario = self.active_scenario() or None
        try:
            job = self.scheduler.create(
                session_key=self.session_key,
                channel=channel,
                chat_id=chat_id,
                thread_id=str(thread_id_raw) if thread_id_raw is not None else None,
                sender_id=sender_id,
                scenario=scenario,
                cron=cron,
                prompt=prompt,
                recurring=recurring,
            )
        except ValueError as exc:
            return {"error": str(exc)}
        return job.to_dict()

    def list(self) -> list[dict[str, Any]]:
        return [
            job.to_dict()
            for job in self.scheduler.list(session_key=self.session_key)
        ]

    def delete(self, job_id: str) -> bool:
        return self.scheduler.delete(job_id, session_key=self.session_key)


class GatewayRuntime:
    """Glues WireServer <-> Router/SessionManager/Approval/Commands.

    One instance per ``agentm gateway`` process. Holds the outbound sink
    that every component writes into; the sink routes an ``outbound``
    envelope to the durable outbox of the peer(s) serving its channel.
    """

    def __init__(
        self,
        *,
        cwd: str,
        scenario: str | None,
        outbox: SqliteOutbox,
        chat_map: ChatSessionMap,
        session_factory: Callable[
            [str, str, str | None, str | None, dict[str, Any]], Awaitable[Any]
        ],
        command_router: CommandRouter,
        approval_policy: tuple[frozenset[str], frozenset[str], float],
        model_name: str = "",
        make_factory: Callable[[str], Any] | None = None,
        schedule_store: GatewayScheduleStore | None = None,
    ) -> None:
        self._cwd = cwd
        self._scenario = scenario
        self._outbox = outbox
        self._chat_map = chat_map
        self._command_router = command_router
        self._model_name = model_name
        self._default_session_factory = session_factory
        self._make_factory = make_factory
        self._session_model_names: dict[str, str] = {}
        self._session_model_factories: dict[
            str,
            Callable[
                [str, str, str | None, str | None, dict[str, Any]], Awaitable[Any]
            ],
        ] = {}
        self._session_scenarios: dict[str, str] = {}
        require, block, timeout = approval_policy
        self._approval = ApprovalManager(
            self._emit_outbound,
            require_approval=require,
            always_block=block,
            timeout_seconds=timeout,
        )
        # Live sub-agent sessions, addressable by id. Shared with the
        # SessionManager (which seeds it as a per-session service the sub_agent
        # atom registers children into) and read here to route a child-addressed
        # inbound to that child's inbox (see interactive-subagent design).
        self._child_registry = ChildSessionRegistry()
        self._sessions = SessionManager(
            cwd=cwd,
            chat_map=chat_map,
            session_factory=self._session_factory_for_key,
            outbound_sink=self._emit_outbound,
            approval_manager=self._approval,
            child_registry=self._child_registry,
        )
        self._server: WireServer | None = None
        self._inflight: set[asyncio.Task[Any]] = set()
        self._peer_channels: dict[str, str] = {}
        self._peer_session_keys: dict[str, set[str]] = {}
        self._peer_cwds: dict[str, str] = {}
        self._session_commands: dict[str, set[str]] = {}
        self._session_routes: dict[str, tuple[str, str, str | None]] = {}
        self._snapshots: dict[str, _SessionSnapshot] = {}
        self._request_id_cache: OrderedDict[str, None] = OrderedDict()
        self._max_request_cache_size = 2048
        # Per-chat prompt count, surfaced by /status. Incremented on each
        # conversational turn (a prompt, not a control command).
        self._turn_counts: dict[str, int] = {}
        self._scheduler: GatewayScheduler | None = None
        if schedule_store is not None:
            self._scheduler = GatewayScheduler(
                store=schedule_store,
                fire=self._fire_scheduled_job,
            )


    def attach_server(self, server: WireServer) -> None:
        self._server = server

    def start_scheduler(self) -> None:
        if self._scheduler is not None:
            self._scheduler.start()

    async def _session_factory_for_key(
        self,
        cwd: str,
        session_key: str,
        scenario: str | None,
        resume: str | None,
        wire_services: dict[str, Any],
    ) -> Any:
        if self._scheduler is not None:
            turn_context = wire_services.get("turn_context")
            if isinstance(turn_context, dict):
                def active_scenario_for_session() -> str:
                    return self._active_scenario_name(session_key)

                wire_services[GATEWAY_SCHEDULER_SERVICE] = (
                    _BoundGatewaySchedulerService(
                        scheduler=self._scheduler,
                        session_key=session_key,
                        turn_context=turn_context,
                        active_scenario=active_scenario_for_session,
                    )
                )
        factory = self._session_model_factories.get(session_key)
        if factory is None:
            model_name = self._session_model_names.get(
                session_key
            ) or self._chat_map.metadata(session_key).get("model")
            if (
                model_name
                and model_name != self._model_name
                and self._make_factory is not None
            ):
                try:
                    factory = self._make_factory(model_name)
                    self._session_model_factories[session_key] = factory
                    self._session_model_names[session_key] = model_name
                except Exception:
                    logger.exception(
                        "failed to restore model override {} for {}",
                        model_name,
                        session_key,
                    )
        if factory is None:
            factory = self._default_session_factory
        return await factory(cwd, session_key, scenario, resume, wire_services)

    def _active_model_name(self, session_key: str) -> str:
        return (
            self._session_model_names.get(session_key)
            or self._chat_map.metadata(session_key).get("model")
            or self._model_name
        )

    def _active_scenario_name(self, session_key: str) -> str:
        return (
            self._session_scenarios.get(session_key)
            or self._chat_map.metadata(session_key).get("scenario")
            or self._scenario
            or ""
        )

    def describe_capabilities(self) -> dict[str, Any]:
        """Static, session-independent capabilities for the welcome handshake.

        A chat client connects (and gets the welcome) *before* it sends a first
        message, but a session — and thus the ``session_ready`` frame that
        advertises the scenario's tools and in-session commands — is created
        only on that first prompt. So the welcome carries what the gateway knows
        without a live session: the configured model profiles, the active model,
        the scenario name, and the gateway's own command catalog (name + kind +
        summary). The terminal seeds its model picker and command palette from
        this immediately; ``session_ready`` later augments it with
        session-specific tools/commands."""
        from agentm.core.lib.user_config import load_user_config

        try:
            models = list(load_user_config().models.keys())
        except Exception:
            logger.exception("describe_capabilities: failed to read model profiles")
            models = []
        all_handlers = list(self._command_router.registry.all())
        commands = [
            {"name": h.name, "kind": h.kind, "summary": h.summary}
            for h in all_handlers
            if h.namespace is None
        ]
        # Skills are gateway-registered commands under the "skill" namespace
        # (invoked as /skill:<name>). They are discovered at startup and static,
        # so the welcome frame can advertise them for the terminal's /skills view.
        skills = [
            {"name": h.name, "summary": h.summary}
            for h in all_handlers
            if h.namespace == "skill"
        ]
        try:
            from agentm.extensions.loader import list_scenarios

            scenarios = [_scenario_payload(info) for info in list_scenarios()]
        except Exception:
            logger.exception("describe_capabilities: failed to list scenarios")
            scenarios = []
        return {
            "models": models,
            "model": self._model_name,
            "scenario": self._scenario or "",
            "scenarios": scenarios,
            "commands": commands,
            "skills": skills,
        }

    # -- outbound sink ------------------------------------------------

    async def _emit_outbound(self, body: dict[str, Any]) -> None:
        """Enqueue one ``outbound`` envelope to the peer(s) serving its channel.

        Routing (§3.2): a chat client "serves" the channel it stamps on its
        inbound (§3.4); we learn that mapping in :meth:`handle_inbound`. An
        outbound goes only to peers known to serve ``body["channel"]`` — so
        with a Feishu peer and a terminal peer connected at once, a Feishu
        reply does not get mis-delivered to the terminal (and vice versa).
        If no connected peer is known to serve the channel (a single-client
        deployment before its first inbound, or an empty channel) we fall
        back to every peer. The durable per-peer outbox means a
        momentarily-disconnected client still gets its message on reconnect.
        """
        if self._server is None:
            return
        raw_key = body.pop("_session_key", None)
        session_key = str(raw_key) if raw_key else None
        target_channel = str(body.get("channel") or "")
        meta = body.get("metadata")
        kind = str((meta or {}).get("kind") or "assistant_text")
        if kind == "session_ready" and isinstance(meta, dict):
            if session_key is not None:
                self._remember_session_commands(session_key, meta)
            self._merge_gateway_commands(meta)
        if session_key is not None:
            self._record_session_route(session_key, body)
            # Outbound helpers in tests may use a hand-built runtime object.
            if isinstance(meta, dict) and kind != "session_snapshot":
                changed = self._update_session_snapshot(session_key, kind, meta)
                if changed:
                    await self._emit_session_snapshot(session_key)
        env = Envelope(
            v=WIRE_VERSION,
            id=f"out-{uuid.uuid4().hex[:12]}",
            kind=KIND_OUTBOUND,
            ts=time.time(),
            session_key=session_key,
            body=body,
        )
        targets = route_targets(
            list(self._server.registry),
            self._peer_channels,
            target_channel,
            session_key=session_key,
            peer_session_keys=self._peer_session_keys,
        )
        durable = kind in DURABLE_OUTBOUND_KINDS
        if not durable and kind not in EPHEMERAL_OUTBOUND_KINDS:
            logger.warning(f"outbound kind {kind!r} is not in the known wire vocabulary; delivering as ephemeral (best-effort, droppable)")
        for peer in targets:
            outbox_id: int | None = None
            if durable:
                outbox_id = await asyncio.to_thread(
                    self._outbox.enqueue, peer.peer_id, env
                )
            peer.send_q.put(env, outbox_id)

    def _record_session_route(self, session_key: str, body: dict[str, Any]) -> None:
        routes = getattr(self, "_session_routes", None)
        if routes is None:
            routes = {}
            self._session_routes = routes
        channel = str(body.get("channel") or "")
        chat_id = str(body.get("chat_id") or "")
        if not channel or not chat_id:
            return
        thread_id = body.get("thread_id")
        if thread_id is not None and not isinstance(thread_id, str):
            thread_id = str(thread_id)
        routes[session_key] = (channel, chat_id, thread_id)

    def _snapshot_for(self, session_key: str) -> _SessionSnapshot:
        snapshots = getattr(self, "_snapshots", None)
        if snapshots is None:
            snapshots = {}
            self._snapshots = snapshots
        return snapshots.setdefault(session_key, _SessionSnapshot())

    def _set_children_snapshot(self, snapshot: _SessionSnapshot) -> None:
        snapshot.children = sorted(self._child_registry.ids())

    def _set_pending_interactions_snapshot(self, session_key: str, snapshot: _SessionSnapshot) -> None:
        snapshot.pending_interactions = sorted(self._approval.pending_for_session(session_key))

    def _derive_phase_from_snapshot(self, snapshot: _SessionSnapshot) -> SessionPhase:
        if snapshot.last_error:
            return "errored"
        if snapshot.pending_interactions:
            return "waiting_interaction"
        if snapshot.active_turn_id:
            return "running"
        if snapshot.phase in {"interrupting", "unknown"}:
            return snapshot.phase
        return "idle"

    def _get_gateway_debug_state(self, session_key: str) -> dict[str, Any]:
        session_routes = getattr(self, "_session_routes", {})
        snapshots = getattr(self, "_snapshots", {})
        current_session_route = session_routes.get(session_key)
        child_registry = getattr(self, "_child_registry", None)
        child_ids = child_registry.ids() if child_registry is not None else []
        approval = getattr(self, "_approval", None)
        chat_routes = self._chat_map.snapshot()
        chat_metadata = self._chat_map.snapshot_metadata()

        sessions_state: dict[str, Any] = {}
        for key in sorted(
            set(session_routes.keys()) | set(snapshots.keys()) | set(chat_routes)
        ):
            route = session_routes.get(key)
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
                "snapshot": self._snapshot_for(key).as_dict(),
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
                "snapshot": self._snapshot_for(session_key).as_dict(),
                "model": self._active_model_name(session_key),
                "scenario": self._active_scenario_name(session_key),
                "command_names": sorted(self._session_commands.get(session_key, set())),
                "turn_count": self._turn_counts.get(session_key, 0),
                "pending_approvals": (
                    approval.pending_for_session(session_key)
                    if approval is not None
                    else []
                ),
                "child_sessions": child_ids,
            },
            "sessions": sessions_state,
            "global": {
                "inflight_tasks": len(self._inflight),
                "tracked_sessions": len(session_routes),
                "outbox_ready": getattr(self, "_outbox", None) is not None,
                "total_pending_approvals": (
                    approval.pending_count if approval is not None else 0
                ),
            },
        }

    def _update_session_snapshot(
        self, session_key: str, kind: str, metadata: dict[str, Any]
    ) -> bool:
        snapshot = self._snapshot_for(session_key)
        before = snapshot.as_dict()

        if kind == "session_ready":
            tools = metadata.get("tool_names")
            if isinstance(tools, list):
                snapshot.tool_names = sorted({str(t) for t in tools if isinstance(t, str)})
            commands = metadata.get("command_names")
            if isinstance(commands, list):
                snapshot.command_names = sorted(
                    {str(c) for c in commands if isinstance(c, str)}
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
        elif kind == "child_start":
            self._set_children_snapshot(snapshot)
        elif kind == "child_end":
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
                    [i for i in snapshot.pending_interactions if i != approval_id]
                )
            else:
                # Fallback to authoritative source when id is missing.
                self._set_pending_interactions_snapshot(session_key, snapshot)
        else:
            return False

        # Child/approval snapshots can change quickly and are derived from other
        # services. Keep them always current after any mutation.
        if kind in {"child_start", "child_end", "approval_request", "approval_resolved"}:
            self._set_children_snapshot(snapshot)
            self._set_pending_interactions_snapshot(session_key, snapshot)

        snapshot.phase = self._derive_phase_from_snapshot(snapshot)

        if kind == "approval_request":
            if not snapshot.pending_interactions:
                self._set_pending_interactions_snapshot(session_key, snapshot)

        return snapshot.as_dict() != before

    async def _emit_session_snapshot(self, session_key: str) -> None:
        routes = getattr(self, "_session_routes", None)
        route = routes.get(session_key) if routes is not None else None
        if route is None:
            return
        snapshot = self._snapshot_for(session_key)
        self._set_children_snapshot(snapshot)
        self._set_pending_interactions_snapshot(session_key, snapshot)
        snapshot.phase = self._derive_phase_from_snapshot(snapshot)

        channel, chat_id, thread_id = route
        await self._emit_outbound(
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

    def _remember_session_commands(
        self, session_key: str, meta: dict[str, Any]
    ) -> None:
        """Cache command names a session_ready frame carries."""
        names = meta.get("command_names")
        if not isinstance(names, list):
            return
        self._session_commands[session_key] = {
            n for n in names if isinstance(n, str)
        }

    def _remember_live_session_commands(
        self, session_key: str, sess: Any
    ) -> set[str] | None:
        names = getattr(sess, "command_names", None)
        if not isinstance(names, list):
            return None
        known = {name for name in names if isinstance(name, str)}
        self._session_commands[session_key] = known
        return known

    def _merge_gateway_commands(self, meta: dict[str, Any]) -> None:
        """Fold the gateway's own builtin commands into a session_ready frame."""
        existing = meta.get("command_names")
        names: list[str] = list(existing) if isinstance(existing, list) else []
        seen = set(names)
        for handler in self._command_router.registry.all():
            if handler.namespace is not None:
                continue
            if handler.name not in seen:
                names.append(handler.name)
                seen.add(handler.name)
        meta["command_names"] = names

    def _is_mutating_action(self, action: RouterAction) -> bool:
        return action in {
            RouterAction.SUBMIT,
            RouterAction.RESOLVE_APPROVAL,
            RouterAction.RUN_COMMAND,
            RouterAction.INTERACTION_RESPONSE,
            RouterAction.INTERRUPT,
            RouterAction.PROMPT_SESSION,
        }

    def _request_cache_key(
        self, peer_id: str, session_key: str, action: RouterAction, request_id: str
    ) -> str:
        return f"{peer_id}|{session_key}|{action.value}|{request_id}"

    def _is_duplicate_request(
        self, peer_id: str, session_key: str, action: RouterAction, request_id: str
    ) -> bool:
        key = self._request_cache_key(peer_id, session_key, action, request_id)
        if key in self._request_id_cache:
            self._request_id_cache.move_to_end(key)
            return True
        self._request_id_cache[key] = None
        while len(self._request_id_cache) > self._max_request_cache_size:
            self._request_id_cache.popitem(last=False)
        return False

    async def _emit_request_ack(
        self,
        session_key: str,
        body: InboundBody,
        action: RouterAction,
        *,
        duplicate: bool,
    ) -> None:
        if body.request_id is None:
            return
        status = "duplicate" if duplicate else "accepted"
        metadata = {
            "kind": "request_ack",
            "request_id": body.request_id,
            "action": action.value,
            "status": status,
            "policy": body.policy or "cooperative",
        }
        if body.interaction_id is not None:
            metadata["interaction_id"] = body.interaction_id
        await self._emit_outbound(
            {
                "channel": body.channel,
                "chat_id": body.chat_id,
                "thread_id": body.thread_id,
                "_session_key": session_key,
                "content": "",
                "metadata": metadata,
            }
        )

    # -- inbound handler ----------------------------------------------

    async def handle_inbound(self, peer: PeerSession, env: Envelope) -> None:
        session_key = env.session_key
        if not session_key:
            logger.warning(f"dropping inbound id={env.id} with no session_key")
            return
        try:
            decision = dispatch(env)
        except Exception:
            logger.exception(f"router failed for inbound id={env.id}")
            return
        body = decision.body
        if body.channel:
            self._peer_channels[peer.peer_id] = body.channel
        self._peer_session_keys.setdefault(peer.peer_id, set()).add(session_key)
        if peer.cwd is not None and session_key not in self._peer_cwds:
            self._peer_cwds[session_key] = peer.cwd

        if (
            body.request_id is not None
            and self._is_mutating_action(decision.action)
        ):
            duplicate = self._is_duplicate_request(
                peer.peer_id, session_key, decision.action, body.request_id
            )
            await self._emit_request_ack(
                session_key, body, decision.action, duplicate=duplicate
            )
            if duplicate:
                return

        if decision.action in {
            RouterAction.RESOLVE_APPROVAL,
            RouterAction.INTERACTION_RESPONSE,
        }:
            self._resolve_interaction(session_key, body)
            return

        # A session_key that names a registered child is interactive sub-agent
        # input: deliver it to that child's own inbox (the child's driver runs
        # the turn and its trajectory forwards back over the child_id-stamped
        # wire) rather than treating the id as a chat key for get_or_create.
        # The child is caller-agnostic — the human's message rides the same
        # inbox the parent's inject_instruction would (interactive-subagent §A).
        if session_key in self._child_registry:
            if decision.action is RouterAction.INTERRUPT:
                self._interrupt_child(session_key)
            else:
                self._deliver_child_input(session_key, body)
            return
        if decision.action is RouterAction.INTERRUPT:
            self._interrupt_session(session_key)
            return
        if decision.action is RouterAction.STEER:
            self._steer_session(session_key, body)
            return
        task = asyncio.create_task(
            self._dispatch_command_or_prompt(session_key, env.scenario, decision.action, body),
            name=f"gw-inbound-{session_key}",
        )
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _dispatch_command_or_prompt(
        self,
        session_key: str,
        scenario: str | None,
        action: RouterAction,
        body: InboundBody,
    ) -> None:
        try:
            if action is RouterAction.RUN_COMMAND:
                await self._run_command(session_key, scenario, body)
            elif action in {RouterAction.SUBMIT, RouterAction.PROMPT_SESSION}:
                await self._submit_session_input(session_key, scenario, body)
            else:
                logger.error(
                    f"unsupported action in _dispatch_command_or_prompt: {action}"
                )
        except Exception as exc:
            logger.exception(f"error handling inbound from {session_key}")
            await self._send_error(session_key, body, exc)

    def _interrupt_session(self, session_key: str) -> None:
        sess = self._sessions.get(session_key)
        if sess is not None:
            sess.interrupt()
            snapshot = self._snapshot_for(session_key)
            if snapshot.phase != "interrupting":
                snapshot.phase = "interrupting"
            snapshot.active_turn_id = None
            if not snapshot.pending_interactions:
                self._schedule_session_snapshot(session_key)

    def _steer_session(self, session_key: str, body: InboundBody) -> None:
        """Inject a user message mid-turn without waiting for turn end."""
        sess = self._sessions.get(session_key)
        if sess is None:
            return
        content = body.content or ""
        if not content.strip():
            return
        sess.send_user_message(content)

    def _interrupt_child(self, child_id: str) -> None:
        child = self._child_registry.get(child_id)
        if child is not None:
            child.interrupt()

    def _deliver_child_input(self, child_id: str, body: InboundBody) -> None:
        """Push a human turn into a live child's inbox as ``source="user"``.

        This is the inbound twin of ``child_wire_forwarder``: the child runs on
        its own driver, so we never call ``prompt()`` (that would contend with
        whatever already drives the child's loop — the parent's monitor while
        the task is live, or the child's own idle driver afterward). A plain
        inbox push is drained at the child's next turn boundary by whoever owns
        the loop, which is exactly the caller-agnostic semantics the design
        wants (the parent perceives it too, since the message lands in the
        child's shared, parent-forwarded trajectory)."""

        child = self._child_registry.get(child_id)
        if child is None:
            return
        content = body.content or ""
        if not content.strip():
            return
        self._push_user_input(child.inbox, body, content)

    def _resolve_interaction(self, session_key: str, body: InboundBody) -> None:
        if body.button_value:
            self._approval.resolve(body.button_value, body.sender_id)
            # Waits for approval_request/approval_resolved outbound to drive the
            # definitive projection, but keep an immediate state write for
            # clients that are sensitive to control latency.
            self._schedule_session_snapshot(session_key)

    def _schedule_session_snapshot(self, session_key: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._emit_session_snapshot(session_key))

    async def _run_command(
        self, session_key: str, scenario: str | None, body: InboundBody
    ) -> None:
        ctx = self._command_context(session_key, body)
        msg = CommandInbound(
            session_key=session_key,
            channel=body.channel,
            chat_id=body.chat_id,
            sender_id=body.sender_id,
            content=body.content,
            thread_id=body.thread_id,
        )
        result = await self._command_router.try_dispatch(msg, ctx)
        if result is None:
            inv = parse_invocation(msg)
            if inv is not None and inv.name:
                command_key = (
                    inv.name
                    if inv.namespace is None
                    else f"{inv.namespace}:{inv.name}"
                )
                sess = await self._get_or_create_session_for_input(
                    session_key, scenario, body
                )
                known = self._remember_live_session_commands(session_key, sess)
                if known is None:
                    known = self._session_commands.get(session_key)
                if known is not None and command_key in known:
                    await sess.prompt(body.content)
                    await self._emit_session_snapshot(session_key)
                    return
                if known is not None:
                    await self._emit_unknown_command(session_key, body, inv.raw)
                    return
            await self._submit_session_input(session_key, scenario, body)
            return
        for out in result.outbound:
            out_body = out.to_dict()
            out_body["_session_key"] = session_key
            await self._emit_outbound(out_body)
        if result.side_effect is not None:
            try:
                await result.side_effect(self)
            except Exception:
                logger.exception("command side_effect raised")
        if result.expanded_prompt is not None:
            from dataclasses import replace

            await self._prompt_session(
                session_key, scenario, replace(body, content=result.expanded_prompt)
            )

    def _push_user_input(
        self, inbox: Any, body: InboundBody, content: str
    ) -> None:
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
        # Preserve existing behavior for old clients that send no request_id:
        # no dedup hint is used.
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

    def _resolve_peer_cwd(self, session_key: str) -> str | None:
        """Resolve the cwd from the peer's hello for this session."""
        return self._peer_cwds.get(session_key)

    async def _get_or_create_session_for_input(
        self, session_key: str, scenario: str | None, body: InboundBody
    ) -> Any:
        peer_cwd = self._resolve_peer_cwd(session_key)
        if scenario:
            from agentm.extensions.loader import validate_scenario

            validate_scenario(scenario)
            self._session_scenarios[session_key] = scenario
            self._chat_map.set_metadata(session_key, scenario=scenario)
        sess = await self._sessions.get_or_create(
            session_key, self._active_scenario_name(session_key), body, cwd=peer_cwd
        )
        self._sessions.set_turn_context(session_key, body)
        # Route context may be required before we can send a session snapshot.
        self._record_session_route(
            session_key,
            {
                "channel": body.channel,
                "chat_id": body.chat_id,
                "thread_id": body.thread_id,
            },
        )
        self._snapshot_for(session_key)
        return sess

    async def _submit_session_input(
        self, session_key: str, scenario: str | None, body: InboundBody
    ) -> None:
        if body.policy is not None and body.policy.lower() == "interrupt_first":
            self._interrupt_session(session_key)
        sess = await self._get_or_create_session_for_input(
            session_key, scenario, body
        )
        self._turn_counts[session_key] = self._turn_counts.get(session_key, 0) + 1
        if body.content == "":
            return
        self._push_user_input(sess.inbox, body, body.content)
        await self._emit_session_snapshot(session_key)

    async def _prompt_session(
        self, session_key: str, scenario: str | None, body: InboundBody
    ) -> None:
        await self._submit_session_input(session_key, scenario, body)

    async def _fire_scheduled_job(self, job: ScheduledJob) -> None:
        request_id = f"gateway-schedule:{job.id}:{uuid.uuid4().hex[:12]}"
        content = (
            f"<system-reminder source=\"gateway_schedule\" job_id=\"{job.id}\">\n"
            f"{job.prompt}\n"
            "</system-reminder>"
        )
        body = InboundBody(
            channel=job.channel,
            chat_id=job.chat_id,
            content=content,
            thread_id=job.thread_id,
            sender_id=job.sender_id,
            request_id=request_id,
        )
        await self._submit_session_input(job.session_key, job.scenario, body)

    # -- command-context plumbing -------------------------------------

    def _command_context(
        self, session_key: str, body: InboundBody
    ) -> CommandContext:
        async def end_session() -> None:
            await self._sessions.shutdown_session(session_key)
            self._turn_counts.pop(session_key, None)
            self._snapshots.pop(session_key, None)
            self._session_routes.pop(session_key, None)

        async def forget_chat_mapping() -> None:
            self._sessions.forget(session_key)
            self._session_commands.pop(session_key, None)
            self._turn_counts.pop(session_key, None)
            self._snapshots.pop(session_key, None)
            self._session_routes.pop(session_key, None)

        def get_route_stats() -> dict[str, Any]:
            return {
                "session_id": self._sessions.session_id(session_key),
                "turn_count": self._turn_counts.get(session_key, 0),
                "pending_approvals": self._approval.pending_count,
            }

        def get_gateway_debug_state() -> dict[str, Any]:
            return self._get_gateway_debug_state(session_key)

        def get_extension_api() -> Any | None:
            sess = self._sessions.get(session_key)
            return sess.extension_api if sess is not None else None

        async def load_session_history(target_sid: str) -> dict[str, Any] | None:
            return await self._load_session_history(target_sid)

        def list_models() -> tuple[str, list[str]]:
            from agentm.core.lib.user_config import load_user_config

            return (
                self._active_model_name(session_key),
                list(load_user_config().models.keys()),
            )

        async def switch_model(name: str) -> tuple[bool, str]:
            return await self._switch_model(session_key, name)

        def list_scenarios() -> tuple[str, list[dict[str, str]]]:
            from agentm.extensions.loader import list_scenarios as _list_scenarios

            return (
                self._active_scenario_name(session_key),
                [_scenario_payload(info) for info in _list_scenarios()],
            )

        async def switch_scenario(name: str) -> tuple[bool, str]:
            return await self._switch_scenario(session_key, name)

        def create_schedule(
            cron: str,
            prompt: str,
            *,
            recurring: bool = True,
        ) -> dict[str, Any]:
            if self._scheduler is None:
                return {"error": "gateway scheduler is not configured"}
            active_scenario = self._active_scenario_name(session_key) or None
            try:
                job = self._scheduler.create(
                    session_key=session_key,
                    channel=body.channel,
                    chat_id=body.chat_id,
                    thread_id=body.thread_id,
                    sender_id=body.sender_id,
                    scenario=active_scenario,
                    cron=cron,
                    prompt=prompt,
                    recurring=recurring,
                )
            except ValueError as exc:
                return {"error": str(exc)}
            return job.to_dict()

        def list_schedules() -> list[dict[str, Any]]:
            if self._scheduler is None:
                return []
            return [
                job.to_dict()
                for job in self._scheduler.list(session_key=session_key)
            ]

        def delete_schedule(job_id: str) -> bool:
            if self._scheduler is None:
                return False
            return self._scheduler.delete(job_id, session_key=session_key)

        async def run_schedule(job_id: str) -> tuple[bool, str]:
            if self._scheduler is None:
                return (False, "gateway scheduler is not configured")
            return await self._scheduler.fire_now(job_id, session_key=session_key)

        async def resume_session(target_sid: str) -> None:
            await self._sessions.shutdown_session(session_key)
            self._sessions.set_chat_mapping(session_key, target_sid)

        async def fork_session(up_to: int | None) -> str | None:
            source_sid = self._sessions.session_id(session_key)
            if not source_sid:
                return None
            await self._sessions.shutdown_session(session_key)
            self._sessions.set_pending_fork(session_key, source_sid, up_to)
            self._turn_counts.pop(session_key, None)
            return source_sid

        def list_session_commands() -> list[str]:
            return sorted(self._session_commands.get(session_key, set()))

        return CommandContext(
            session_key=session_key,
            channel=body.channel,
            chat_id=body.chat_id,
            sender_id=body.sender_id,
            thread_id=body.thread_id,
            end_session=end_session,
            forget_chat_mapping=forget_chat_mapping,
            get_route_stats=get_route_stats,
            get_gateway_debug_state=get_gateway_debug_state,
            list_commands=self._command_router.registry.all,
            get_extension_api=get_extension_api,
            list_models=list_models,
            switch_model=switch_model,
            list_scenarios=list_scenarios,
            switch_scenario=switch_scenario,
            create_schedule=create_schedule,
            list_schedules=list_schedules,
            delete_schedule=delete_schedule,
            run_schedule=run_schedule,
            cwd=self._cwd,
            resume_session=resume_session,
            load_session_history=load_session_history,
            fork_session=fork_session,
            list_session_commands=list_session_commands,
        )


    async def _load_session_history(self, session_id: str) -> dict[str, Any] | None:
        """Load a JSON-safe transcript for a previously persisted session."""
        if not session_id:
            return None

        def _load() -> dict[str, Any] | None:
            from agentm.core.lib.message_codec import serialize_payload
            from agentm.core.runtime.session_bootstrap import make_default_session_store

            try:
                state = make_default_session_store(self._cwd).open(session_id)
            except FileNotFoundError:
                return None
            header = state.get_header()
            context = state.build_session_context()
            return {
                "session_id": state.get_session_id(),
                "cwd": header.cwd if header is not None else self._cwd,
                "messages": [serialize_payload(message) for message in context.messages],
            }

        return await asyncio.to_thread(_load)

    async def _switch_model(self, session_key: str, name: str) -> tuple[bool, str]:
        """Swap the session factory to ``name``'s model profile and start a
        fresh session (same as ``/new``)."""
        from agentm.core.lib.user_config import load_user_config

        if self._make_factory is None:
            return (False, "model switching is not configured")
        key = name.lower()
        if key not in load_user_config().models:
            return (False, f"unknown model '{name}'")
        self._session_model_factories[session_key] = self._make_factory(key)
        self._session_model_names[session_key] = key
        self._chat_map.set_metadata(session_key, model=key)
        await self._sessions.shutdown_session(session_key)
        self._sessions.forget(session_key)
        self._session_commands.pop(session_key, None)
        self._turn_counts.pop(session_key, None)
        self._snapshots.pop(session_key, None)
        self._session_routes.pop(session_key, None)
        return (True, key)

    async def _switch_scenario(self, session_key: str, name: str) -> tuple[bool, str]:
        """Switch the active scenario and start a fresh session for this chat."""
        from agentm.extensions.loader import ScenarioLoadError, validate_scenario

        key = name.strip()
        if not key:
            return (False, "scenario name is required")
        try:
            validate_scenario(key)
        except ScenarioLoadError as exc:
            return (False, str(exc))
        self._session_scenarios[session_key] = key
        self._chat_map.set_metadata(session_key, scenario=key)
        await self._sessions.shutdown_session(session_key)
        self._sessions.forget(session_key)
        self._session_commands.pop(session_key, None)
        self._turn_counts.pop(session_key, None)
        self._snapshots.pop(session_key, None)
        self._session_routes.pop(session_key, None)
        return (True, key)

    async def _emit_unknown_command(
        self, session_key: str, body: InboundBody, raw: str
    ) -> None:
        await self._emit_outbound(
            {
                "channel": body.channel,
                "chat_id": body.chat_id,
                "content": UNKNOWN_REPLY.format(raw=raw),
                "thread_id": body.thread_id,
                "metadata": {"kind": "diagnostic_error"},
                "_session_key": session_key,
            }
        )

    async def _send_error(
        self, session_key: str, body: InboundBody, exc: Exception
    ) -> None:
        err_type = type(exc).__name__
        err_msg = (str(exc).strip() or "(no message)")[:800]
        await self._emit_outbound(
            {
                "channel": body.channel,
                "chat_id": body.chat_id,
                "content": f"Gateway error — {err_type}: {err_msg}",
                "metadata": {"kind": "diagnostic_error"},
                "_session_key": session_key,
            }
        )

    async def shutdown(self) -> None:
        if self._scheduler is not None:
            await self._scheduler.stop()
        inflight = list(self._inflight)
        for task in inflight:
            task.cancel()
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        await self._sessions.shutdown_all()
