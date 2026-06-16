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
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

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


def route_targets(
    peers: list[PeerSession],
    peer_channels: dict[str, str],
    target_channel: str,
) -> list[PeerSession]:
    """Pick the peers an outbound for ``target_channel`` should reach (§3.2).

    A peer "serves" the channel it stamped on its inbound (§3.4), recorded
    in ``peer_channels``. Route only to peers known to serve
    ``target_channel`` — so a Feishu reply is not mis-delivered to a
    simultaneously-connected terminal client. If no connected peer is known
    to serve the channel (single-client deployment before its first inbound,
    or an empty channel), fall back to every peer so the degenerate case
    still delivers.
    """
    matching = [p for p in peers if peer_channels.get(p.peer_id) == target_channel]
    return matching if matching else peers


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
    ) -> None:
        self._cwd = cwd
        self._scenario = scenario
        self._outbox = outbox
        self._command_router = command_router
        self._model_name = model_name
        self._make_factory = make_factory
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
            session_factory=session_factory,
            outbound_sink=self._emit_outbound,
            approval_manager=self._approval,
            child_registry=self._child_registry,
        )
        self._server: WireServer | None = None
        self._inflight: set[asyncio.Task[Any]] = set()
        self._peer_channels: dict[str, str] = {}
        self._session_commands: dict[str, set[str]] = {}
        # Per-chat prompt count, surfaced by /status. Incremented on each
        # conversational turn (a prompt, not a control command).
        self._turn_counts: dict[str, int] = {}

    def attach_server(self, server: WireServer) -> None:
        self._server = server

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
        return {
            "models": models,
            "model": self._model_name,
            "scenario": self._scenario or "",
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
        session_key = str(body.pop("_session_key", "") or "") or None
        target_channel = str(body.get("channel") or "")
        meta = body.get("metadata")
        kind = str((meta or {}).get("kind") or "assistant_text")
        if kind == "session_ready" and isinstance(meta, dict):
            if session_key is not None:
                self._remember_session_commands(session_key, meta)
            self._merge_gateway_commands(meta)
        env = Envelope(
            v=WIRE_VERSION,
            id=f"out-{uuid.uuid4().hex[:12]}",
            kind=KIND_OUTBOUND,
            ts=time.time(),
            session_key=session_key,
            body=body,
        )
        targets = route_targets(
            list(self._server.registry), self._peer_channels, target_channel
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

    def _remember_session_commands(
        self, session_key: str, meta: dict[str, Any]
    ) -> None:
        """Cache the bare command names a session_ready frame carries."""
        names = meta.get("command_names")
        if not isinstance(names, list):
            return
        self._session_commands[session_key] = {
            n for n in names if isinstance(n, str) and ":" not in n
        }

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
        if decision.action is RouterAction.RESOLVE_APPROVAL:
            self._resolve_approval(body)
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
                await self._run_command(session_key, body)
            else:
                await self._prompt_session(session_key, scenario, body)
        except Exception as exc:
            logger.exception(f"error handling inbound from {session_key}")
            await self._send_error(session_key, body, exc)

    def _interrupt_session(self, session_key: str) -> None:
        sess = self._sessions.get(session_key)
        if sess is not None:
            sess.interrupt()

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
        from agentm.core.runtime.session_inbox import InboxItem

        child = self._child_registry.get(child_id)
        if child is None:
            return
        content = body.content or ""
        if not content.strip():
            return
        child.inbox.push(InboxItem(source="user", payload=content))

    def _resolve_approval(self, body: InboundBody) -> None:
        if body.button_value:
            self._approval.resolve(body.button_value, body.sender_id)

    async def _run_command(self, session_key: str, body: InboundBody) -> None:
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
            if inv is not None and inv.name and inv.namespace is None:
                known = self._session_commands.get(session_key)
                if known is not None and inv.name not in known:
                    await self._emit_unknown_command(session_key, body, inv.raw)
                    return
            await self._prompt_session(session_key, None, body)
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
                session_key, None, replace(body, content=result.expanded_prompt)
            )

    async def _prompt_session(
        self, session_key: str, scenario: str | None, body: InboundBody
    ) -> None:
        sess = await self._sessions.get_or_create(
            session_key, scenario or self._scenario, body
        )
        self._sessions.set_turn_context(session_key, body)
        self._turn_counts[session_key] = self._turn_counts.get(session_key, 0) + 1
        await sess.prompt(body.content)

    # -- command-context plumbing -------------------------------------

    def _command_context(
        self, session_key: str, body: InboundBody
    ) -> CommandContext:
        async def end_session() -> None:
            await self._sessions.shutdown_session(session_key)
            self._turn_counts.pop(session_key, None)

        async def forget_chat_mapping() -> None:
            self._sessions.forget(session_key)
            self._session_commands.pop(session_key, None)
            self._turn_counts.pop(session_key, None)

        def get_route_stats() -> dict[str, Any]:
            return {
                "session_id": self._sessions.session_id(session_key),
                "turn_count": self._turn_counts.get(session_key, 0),
                "pending_approvals": self._approval.pending_count,
            }

        def get_extension_api() -> Any | None:
            sess = self._sessions.get(session_key)
            return sess.extension_api if sess is not None else None

        def list_models() -> tuple[str, list[str]]:
            from agentm.core.lib.user_config import load_user_config

            return (self._model_name, list(load_user_config().models.keys()))

        async def switch_model(name: str) -> tuple[bool, str]:
            return await self._switch_model(session_key, name)

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
            list_commands=self._command_router.registry.all,
            get_extension_api=get_extension_api,
            list_models=list_models,
            switch_model=switch_model,
            cwd=self._cwd,
            resume_session=resume_session,
            fork_session=fork_session,
            list_session_commands=list_session_commands,
        )

    async def _switch_model(self, session_key: str, name: str) -> tuple[bool, str]:
        """Swap the session factory to ``name``'s model profile and start a
        fresh session (same as ``/new``)."""
        from agentm.core.lib.user_config import load_user_config

        if self._make_factory is None:
            return (False, "model switching is not configured")
        key = name.lower()
        if key not in load_user_config().models:
            return (False, f"unknown model '{name}'")
        self._sessions.set_factory(self._make_factory(key))
        self._model_name = key
        await self._sessions.shutdown_session(session_key)
        self._sessions.forget(session_key)
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
        inflight = list(self._inflight)
        for task in inflight:
            task.cancel()
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        await self._sessions.shutdown_all()
