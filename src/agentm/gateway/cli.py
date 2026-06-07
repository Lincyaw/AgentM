"""``agentm gateway`` subcommand — the single-process gateway runtime.

Boots one process that holds every chat session in memory and serves all
chat-client peers over the v2 wire protocol (§1). Wires together:

    WireServer  <- inbound -  Router  -> SessionManager (sessions dict)
        ^                        |              |
        |                        |              v
     Outbox (per peer)      CommandRouter   wire_driver atom
        ^                   ApprovalManager      |
        +------------------- outbound sink <------+

The outbound sink the SessionManager / ApprovalManager / CommandRouter
all write into enqueues an ``outbound`` envelope onto the connected chat
client's durable outbox; the WireServer's delivery worker ships it.

Run as a long-lived daemon and connect chat clients separately::

    agentm gateway --bind unix:///tmp/gw.sock
    agentm-terminal --connect unix:///tmp/gw.sock
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

import typer

from agentm.gateway import DEFAULT_SOCKET_URL, autoload_dotenv
from agentm.gateway.approval import ApprovalManager
from agentm.gateway.auth import (
    Authenticator,
    TokenAuthenticator,
    UnixPeerCredAuthenticator,
)
from agentm.gateway.chat_session_map import ChatSessionMap
from agentm.gateway.commands import (
    UNKNOWN_REPLY,
    CommandContext,
    CommandInbound,
    CommandRouter,
    discover_commands,
    parse_invocation,
)
from agentm.gateway.outbox import SqliteInbox, SqliteOutbox
from agentm.gateway.peer import PeerSession
from agentm.gateway.router import RouterAction, dispatch
from agentm.gateway.server import WireServer
from agentm.gateway.session_manager import SessionManager
from agentm.gateway.transport import (
    ServerTransport,
    UnixServerTransport,
    WebSocketServerTransport,
)
from agentm.gateway.wire import (
    DURABLE_OUTBOUND_KINDS,
    EPHEMERAL_OUTBOUND_KINDS,
    KIND_OUTBOUND,
    WIRE_VERSION,
    Envelope,
    InboundBody,
)

autoload_dotenv()

PROG = "agentm gateway"

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_SIGINT = 130

logger = logging.getLogger("agentm.gateway.cli")


# -------- bind resolution ---------------------------------------------


@dataclass(frozen=True)
class BindSpec:
    """Resolved ``--bind`` configuration."""

    scheme: str
    socket_path: str = ""
    allow_uids: frozenset[int] | None = frozenset()
    host: str = ""
    port: int = 0
    url: str = ""
    tokens: frozenset[str] = frozenset()
    allow_anonymous: bool = False
    tls_cert: str | None = None
    tls_key: str | None = None


def _load_tokens_file(path: str) -> set[str]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(
            f"--bind-token-file {path!r}: cannot read: {exc.strerror or exc}"
        ) from exc
    tokens: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens.add(line)
    return tokens


def _resolve_bind(
    *,
    bind: str | None,
    bind_allow_uid: list[int] | None,
    bind_allow_any_uid: bool,
    bind_token_file: str | None,
    bind_allow_anonymous: bool,
    tls_cert: str | None,
    tls_key: str | None,
) -> BindSpec:
    """Merge CLI flags > shared default into a :class:`BindSpec`.

    Supports ``unix://``, ``ws://``, ``wss://``. Raises ``SystemExit``
    (exit 2) on invalid input.
    """
    url = bind or DEFAULT_SOCKET_URL
    parsed = urlparse(str(url))
    scheme = parsed.scheme

    if scheme == "unix":
        if tls_cert or tls_key:
            raise SystemExit(
                "--tls-cert/--tls-key are only valid with ws://wss:// binds, "
                f"not {url!r}."
            )
        socket_path = parsed.path or parsed.netloc
        if not socket_path:
            raise SystemExit(
                f"--bind {url!r} has no socket path; use unix:///abs/path/to/sock"
            )
        cli_uids = list(bind_allow_uid or ())
        cli_any = bool(bind_allow_any_uid)
        if cli_uids and cli_any:
            raise SystemExit(
                "--bind-allow-uid and --bind-allow-any-uid are mutually exclusive."
            )
        allow_uids: frozenset[int] | None
        if cli_any:
            allow_uids = None
        elif cli_uids:
            allow_uids = frozenset(int(x) for x in cli_uids)
        else:
            allow_uids = frozenset({os.geteuid()})
        return BindSpec(scheme="unix", socket_path=socket_path, allow_uids=allow_uids)

    if scheme not in ("ws", "wss"):
        raise SystemExit(
            f"--bind scheme {scheme!r} not supported; use unix://, ws://, or wss://."
        )
    if scheme == "wss" and (not tls_cert or not tls_key):
        raise SystemExit("wss:// bind requires --tls-cert and --tls-key.")
    if scheme == "ws" and (tls_cert or tls_key):
        raise SystemExit("ws:// bind cannot use TLS; switch to wss:// or drop --tls-*.")
    if bind_allow_uid or bind_allow_any_uid:
        raise SystemExit(
            "--bind-allow-uid / --bind-allow-any-uid are unix-only; "
            "use --bind-token-file with ws://wss://."
        )
    tokens: set[str] = set()
    if bind_token_file:
        tokens.update(_load_tokens_file(bind_token_file))
    if not tokens and not bind_allow_anonymous:
        raise SystemExit(
            f"{scheme}:// bind requires --bind-token-file. Pass "
            "--bind-allow-anonymous to opt out (NOT recommended)."
        )
    host = parsed.hostname or "0.0.0.0"
    port = parsed.port or (443 if scheme == "wss" else 80)
    return BindSpec(
        scheme=scheme,
        host=host,
        port=port,
        url=f"{scheme}://{host}:{port}{parsed.path or '/'}",
        tokens=frozenset(tokens),
        allow_anonymous=bool(bind_allow_anonymous) and not tokens,
        tls_cert=tls_cert,
        tls_key=tls_key,
    )


def _build_server_transport(spec: BindSpec) -> ServerTransport:
    if spec.scheme == "unix":
        return UnixServerTransport(spec.socket_path)
    ssl_context = None
    if spec.scheme == "wss":
        import ssl

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        assert spec.tls_cert is not None and spec.tls_key is not None
        ssl_context.load_cert_chain(certfile=spec.tls_cert, keyfile=spec.tls_key)
    return WebSocketServerTransport(
        host=spec.host, port=spec.port, ssl_context=ssl_context
    )


def _build_authenticator(spec: BindSpec) -> Authenticator:
    if spec.scheme == "unix":
        return UnixPeerCredAuthenticator(
            allowed_uids=set(spec.allow_uids) if spec.allow_uids is not None else None
        )
    if spec.allow_anonymous:
        from agentm.gateway.auth import AllowAllAuthenticator

        return AllowAllAuthenticator()
    return TokenAuthenticator(allowed_tokens=set(spec.tokens))


# -------- session factory ---------------------------------------------


def _build_session_factory(
    *, provider: str, model: str, profile: Any | None, reasoning_effort: str | None = None
) -> Callable[[str, str, str | None, str | None, dict[str, Any]], Awaitable[Any]]:
    from typing import cast as _cast

    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.lib.user_config import apply_reasoning_effort
    from agentm.core.abi import EventBus
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession
    from agentm.core.runtime.session_bootstrap import (
        make_default_session_store,
        resolve_session_state,
    )

    async def factory(
        cwd: str,
        session_key: str,
        scenario: str | None,
        resume: str | None,
        wire_services: dict[str, Any],
    ) -> Any:
        del session_key  # routing identity; not needed for construction
        store = make_default_session_store(cwd)
        state = resolve_session_state(
            cwd=cwd, resume=resume, continue_recent=False, session_store=store
        )
        if profile is not None:
            build_config = profile.to_build_config()
        else:
            build_config = {"model": model}
        apply_reasoning_effort(build_config, reasoning_effort)
        provider_spec = DEFAULT_PROVIDER_REGISTRY.build(provider, build_config)
        config = AgentSessionConfig(
            cwd=cwd,
            provider=provider_spec,
            scenario=scenario,
            session_manager=_cast(Any, state),
            bus=EventBus(),
            # Seed the wire_driver's services before atoms install and mount the
            # atom during create() so it forwards the creation-time
            # SessionReadyEvent (the chat client's slash-command catalog).
            initial_services=dict(wire_services),
            extra_extensions=[("agentm.extensions.builtin.wire_driver", {})],
        )
        return await AgentSession.create(config)

    return factory


# -------- gateway runtime ---------------------------------------------


def _route_targets(
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


class _GatewayRuntime:
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
        # Active model label + a builder that produces a new session factory for
        # a given model name — together these back the runtime ``/model`` switch.
        self._model_name = model_name
        self._make_factory = make_factory
        require, block, timeout = approval_policy
        self._approval = ApprovalManager(
            self._emit_outbound,
            require_approval=require,
            always_block=block,
            timeout_seconds=timeout,
        )
        self._sessions = SessionManager(
            cwd=cwd,
            chat_map=chat_map,
            session_factory=session_factory,
            outbound_sink=self._emit_outbound,
            approval_manager=self._approval,
        )
        self._server: WireServer | None = None
        # Detached inbound-dispatch tasks (§3.2). Tracked so they are not
        # GC'd mid-flight and are cancelled/awaited on shutdown rather than
        # orphaned.
        self._inflight: set[asyncio.Task[Any]] = set()
        # peer_id -> the channel that peer serves, learned from its inbound
        # (§3.4). Drives outbound routing so a reply reaches only the chat
        # client that owns the conversation.
        self._peer_channels: dict[str, str] = {}
        # session_key -> the bare command names the session registered
        # (``compact`` etc), learned from each session_ready frame as it
        # passes through the outbound sink. Lets _run_command tell a known
        # session command (forward to the session) from one unknown to both
        # layers (surface an "unknown command" diagnostic), and lets /help
        # list session commands the gateway registry doesn't know about.
        self._session_commands: dict[str, set[str]] = {}

    def attach_server(self, server: WireServer) -> None:
        self._server = server

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
            # Record the session-registered command names BEFORE merging the
            # gateway builtins in — these are the names dispatched inside the
            # session by the slash_commands floor atom (compact, ...).
            if session_key is not None:
                self._remember_session_commands(session_key, meta)
            # Fold the gateway's own builtin commands (/new, /status,
            # /model, ...) into the session_ready command list so chat clients
            # surface them in autocomplete. They are routed by the gateway, not
            # the session, so they never appear in the session-emitted list.
            self._merge_gateway_commands(meta)
        env = Envelope(
            v=WIRE_VERSION,
            id=f"out-{uuid.uuid4().hex[:12]}",
            kind=KIND_OUTBOUND,
            ts=time.time(),
            session_key=session_key,
            body=body,
        )
        targets = _route_targets(
            list(self._server.registry), self._peer_channels, target_channel
        )
        # Unified ordered delivery (§2.6): every outbound — durable and
        # ephemeral — is enqueued onto the peer's single send queue, drained
        # in order by the per-peer sender. Enqueue order == event order ==
        # delivery order, so no frame overtakes another. The two classes
        # differ only in persistence: a durable frame is first written to the
        # outbox (so it replays on reconnect) and its row id rides along to be
        # acked after a successful write; an ephemeral frame carries no row id
        # (best-effort, dropped under backpressure / on a dead peer).
        durable = kind in DURABLE_OUTBOUND_KINDS
        if not durable and kind not in EPHEMERAL_OUTBOUND_KINDS:
            # An unknown kind falls through as ephemeral — surface it, since a
            # typo'd DURABLE kind would otherwise silently downgrade.
            logger.warning(
                "outbound kind %r is not in the known wire vocabulary; "
                "delivering as ephemeral (best-effort, droppable)",
                kind,
            )
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
        """Cache the bare command names a session_ready frame carries, keyed
        by session. These are the session-registered commands (dispatched by
        the in-session slash_commands atom); the gateway uses them to forward
        a ``/compact``-style inbound to the session instead of rejecting it,
        and to list them under ``/help``. Namespaced names (``ns:name``) are
        skipped — they are not bare ``/name`` invocations."""
        names = meta.get("command_names")
        if not isinstance(names, list):
            return
        self._session_commands[session_key] = {
            n for n in names if isinstance(n, str) and ":" not in n
        }

    def _merge_gateway_commands(self, meta: dict[str, Any]) -> None:
        """Fold the gateway's top-level command names (builtins like /model,
        /new, /status, plus markdown and skill commands) into a
        session_ready frame's ``command_names`` (deduped, order preserved).
        These are routed by the gateway, not registered inside the session, so
        without this the chat client never learns about them and they don't
        appear in slash autocomplete. Namespaced commands (``/atom:*``) are
        skipped — they are not bare ``/name`` invocations."""
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
            logger.warning("dropping inbound id=%s with no session_key", env.id)
            return
        try:
            decision = dispatch(env)
        except Exception:
            logger.exception("router failed for inbound id=%s", env.id)
            return
        body = decision.body
        # Learn which channel this peer serves (§3.4) so its replies route
        # back to it instead of broadcasting to every connected client.
        if body.channel:
            self._peer_channels[peer.peer_id] = body.channel
        if decision.action is RouterAction.INTERRUPT:
            # Preempt the in-flight prompt inline (NOT as a detached task):
            # it must not queue behind the very prompt it is cancelling.
            # AgentSession.interrupt() sets the kernel abort signal -> the
            # loop ends with SignalAborted (context preserved); a no-op when
            # nothing is running.
            self._interrupt_session(session_key)
            return
        if decision.action is RouterAction.RESOLVE_APPROVAL:
            # Resolve inline: it must not sit behind a session.prompt that
            # is itself awaiting THIS approval's future (that would
            # deadlock the read loop). resolve() is synchronous and fast.
            self._resolve_approval(body)
            return
        # Commands and prompts run as detached tasks so a prompt blocked on
        # an approval future never stalls the WireServer read loop — the
        # button click that resolves the future must still be routed
        # (mirrors the per-inbound task-spawn the v1 gateway used).
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
            logger.exception("error handling inbound from %s", session_key)
            await self._send_error(session_key, body, exc)

    def _interrupt_session(self, session_key: str) -> None:
        sess = self._sessions.get(session_key)
        if sess is not None:
            sess.interrupt()

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
            # try_dispatch returns None for two cases: (a) not a slash command
            # at all, or (b) a slash command the GATEWAY registry doesn't own.
            # Case (b) is the common one for session-registered commands like
            # /compact — those are dispatched INSIDE the session by the
            # slash_commands floor atom, which hooks the session's "input"
            # event. So forward the raw /... text to the session prompt path:
            # the session seam runs it. Only when the name is unknown to BOTH
            # the gateway AND the session (per-session set learned from the
            # session_ready frame) do we surface the "unknown command"
            # diagnostic — otherwise a genuine session command would be
            # wrongly rejected. If we have no session-command knowledge yet
            # (e.g. the session_ready frame hasn't arrived), we forward
            # optimistically rather than reject; a truly-unknown name then
            # reaches the model as text (acceptable v1 tradeoff).
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
        await sess.prompt(body.content)

    # -- command-context plumbing -------------------------------------

    def _command_context(
        self, session_key: str, body: InboundBody
    ) -> CommandContext:
        async def end_session() -> None:
            await self._sessions.shutdown_session(session_key)

        async def forget_chat_mapping() -> None:
            self._sessions.forget(session_key)

        def get_route_stats() -> dict[str, Any]:
            return {
                "session_id": self._sessions.session_id(session_key),
                "turn_count": 0,
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
            list_session_commands=list_session_commands,
        )

    async def _switch_model(self, session_key: str, name: str) -> tuple[bool, str]:
        """Swap the session factory to ``name``'s model profile and start a
        fresh session (same as ``/new``).

        Note: the factory is process-wide, so a switch affects the model used
        for every chat's *next* session, not just this one. Adequate for the
        single-user terminal; revisit if multi-tenant per-chat models are
        needed."""
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
        """Surface the "no such command" diagnostic for a slash command that
        is unknown to both the gateway registry and the session's registered
        set. Mirrors the reply the router used to emit before unknown-command
        routing moved into the gateway."""
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
        # Cancel + drain any in-flight inbound dispatches before tearing
        # down sessions, so no prompt task is left orphaned.
        inflight = list(self._inflight)
        for task in inflight:
            task.cancel()
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        await self._sessions.shutdown_all()


# -------- systemd integration ------------------------------------------
#
# A single mechanism installs BOTH the gateway and the Feishu client as
# systemd units (system-wide when run as root, per-user otherwise). The
# render step is a pure function (resolved inputs -> unit text) so tests can
# assert on the rendered strings without touching real systemd or system
# dirs; the install step writes the files and drives ``systemctl``.

_GATEWAY_UNIT = "agentm-gateway"
_FEISHU_UNIT = "agentm-feishu"


@dataclass(frozen=True)
class _SystemdPlan:
    """Everything the render/install steps need, fully resolved up front.

    ``system`` selects system units (``/etc/systemd/system``, ``User=``,
    ``/run/agentm`` socket) vs user units (``~/.config/systemd/user``,
    ``%t/agentm`` socket, no ``User=``). ``feishu_bin`` is ``None`` when the
    ``agentm-feishu`` entry point cannot be resolved — the feishu unit is
    then skipped (the gateway unit is still rendered/installed).
    """

    system: bool
    unit_dir: Path
    socket_url: str
    gateway_exec_start: str
    env_file: Path
    working_dir: Path
    path_env: str
    run_as: str | None
    feishu_bin: str | None


def _render_gateway_unit(plan: _SystemdPlan) -> str:
    import textwrap

    user_lines = ""
    if plan.system and plan.run_as:
        user_lines = f"User={plan.run_as}\nGroup={plan.run_as}\n"
    # The socket lives under <runtime>/agentm/ — systemd must create that
    # subdir before ExecStart, or the gateway's bind() fails with FileNotFound
    # (exit 1). RuntimeDirectory=agentm creates %t/agentm for user units (=
    # $XDG_RUNTIME_DIR/agentm) and /run/agentm for system units, matching the
    # pinned socket in both modes.
    runtime_lines = "RuntimeDirectory=agentm\nRuntimeDirectoryMode=0750\n"
    wanted_by = "multi-user.target" if plan.system else "default.target"
    return textwrap.dedent(f"""\
        [Unit]
        Description=AgentM gateway (single-process chat session host)
        After=network-online.target
        Wants=network-online.target

        [Service]
        Type=simple
        {user_lines}WorkingDirectory={plan.working_dir}
        EnvironmentFile=-{plan.env_file}
        {runtime_lines}Environment=AGENTM_SOCKET={plan.socket_url}
        Environment=PATH={plan.path_env}
        ExecStart={plan.gateway_exec_start}
        Restart=always
        RestartSec=2
        TimeoutStopSec=10

        [Install]
        WantedBy={wanted_by}
    """)


def _render_feishu_unit(plan: _SystemdPlan) -> str:
    import shlex
    import textwrap

    assert plan.feishu_bin is not None  # caller guards; only render when present
    user_lines = ""
    if plan.system and plan.run_as:
        user_lines = f"User={plan.run_as}\nGroup={plan.run_as}\n"
    wanted_by = "multi-user.target" if plan.system else "default.target"
    # --scenario is intentionally omitted; the feishu client defaults to the
    # chatbot scenario in code. Wants/After (not Requires) keep the coupling
    # loose: a gateway restart does NOT cascade-restart feishu, which relies on
    # its auto-reconnect + outbox replay instead.
    exec_start = " ".join(
        shlex.quote(tok)
        for tok in [plan.feishu_bin, "--connect", plan.socket_url]
    )
    return textwrap.dedent(f"""\
        [Unit]
        Description=AgentM Feishu/Lark chat client
        After={_GATEWAY_UNIT}.service
        Wants={_GATEWAY_UNIT}.service

        [Service]
        Type=simple
        {user_lines}WorkingDirectory={plan.working_dir}
        EnvironmentFile=-{plan.env_file}
        Environment=PATH={plan.path_env}
        ExecStart={exec_start}
        Restart=always
        RestartSec=3
        TimeoutStopSec=10

        [Install]
        WantedBy={wanted_by}
    """)


def _resolve_feishu_bin(agentm_bin: str) -> str | None:
    """Find ``agentm-feishu``: $PATH first, else next to the ``agentm`` bin."""
    import shutil

    found = shutil.which("agentm-feishu")
    if found:
        return found
    sibling = Path(agentm_bin).resolve().parent / "agentm-feishu"
    return str(sibling) if sibling.exists() else None


def _baked_gateway_argv() -> tuple[list[str], str]:
    """Return (gateway-flags, workspace) from the current invocation.

    Strips the systemd flags. ``sys.argv[0]`` is ``"agentm gateway"`` (merged
    by the main CLI dispatcher); the rest are the gateway flags. The workspace
    is the ``--cwd`` value if present, else the current dir.
    """
    cleaned: list[str] = []
    workspace = str(Path.cwd())
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--install-systemd", "--uninstall-systemd"):
            i += 1
            continue
        if arg == "--cwd":
            if i + 1 < len(argv):
                workspace = argv[i + 1]
                i += 2
            else:
                i += 1
            continue
        if arg.startswith("--cwd="):
            workspace = arg.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned, workspace


def _build_systemd_plan() -> _SystemdPlan:
    """Resolve a full :class:`_SystemdPlan` from the live invocation/env."""
    import shlex
    import shutil

    agentm_bin = shutil.which("agentm")
    if agentm_bin is None:
        raise SystemExit(
            "Cannot find `agentm` on $PATH. Install it first or run from "
            "the venv (`uv run agentm gateway --install-systemd`)."
        )

    # User units only — `~/.config/systemd/user` + `systemctl --user`. We do
    # not install system units even when run as root (one less moving part; a
    # root user manager works the same via `systemctl --user`). %t expands to
    # the per-user runtime dir at start.
    system = False
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    socket_url = "unix://%t/agentm/gw.sock"
    run_as = None

    cleaned, workspace = _baked_gateway_argv()
    # Absolutize: systemd rejects a relative WorkingDirectory / EnvironmentFile
    # ("bad unit file setting"), and a relative --cwd baked into ExecStart would
    # resolve against the unit's WorkingDirectory at runtime. resolve() also
    # canonicalizes ~/.. so all three references agree on one absolute path.
    workspace_path = Path(workspace).expanduser().resolve()
    env_file = workspace_path / ".env"

    # Both units MUST agree on the socket: force --bind to the pinned value
    # (drop any --bind the operator passed so it cannot diverge from feishu).
    flags: list[str] = []
    skip_next = False
    for arg in cleaned:
        if skip_next:
            skip_next = False
            continue
        if arg == "--bind":
            skip_next = True
            continue
        if arg.startswith("--bind="):
            continue
        flags.append(arg)
    # Re-add --cwd: _baked_gateway_argv() strips it out (to derive the
    # workspace), but the gateway resolves its session cwd + state-dir from
    # --cwd, NOT from the unit's WorkingDirectory. Without this the installed
    # unit would run sessions under the install-time cwd while its
    # EnvironmentFile pointed at <workspace>/.env — defeating onboard. Keep
    # ExecStart (--cwd), WorkingDirectory, and EnvironmentFile all on the same
    # workspace.
    flags = ["--bind", socket_url, "--cwd", str(workspace_path), *flags]
    # shlex.quote each token: a workspace path with a space or a multi-word
    # --scenario value must not be re-split by systemd's whitespace tokenizer.
    gateway_exec_start = " ".join(
        shlex.quote(tok) for tok in [agentm_bin, "gateway", *flags]
    )

    bin_dir = str(Path(agentm_bin).resolve().parent)
    path_env = f"{bin_dir}:/usr/local/bin:/usr/bin:/bin"

    return _SystemdPlan(
        system=system,
        unit_dir=unit_dir,
        socket_url=socket_url,
        gateway_exec_start=gateway_exec_start,
        env_file=env_file,
        working_dir=workspace_path,
        path_env=path_env,
        run_as=run_as,
        feishu_bin=_resolve_feishu_bin(agentm_bin),
    )


def _systemctl(plan: _SystemdPlan) -> list[str]:
    return ["systemctl"] if plan.system else ["systemctl", "--user"]


def _systemd_action(*, install: bool) -> None:
    """Install or uninstall the gateway + feishu systemd USER units.

    Always user units (``~/.config/systemd/user`` + ``systemctl --user``, socket
    ``%t/agentm/gw.sock``) — we do not install system units even as root. Both
    units share the same socket (gateway ``--bind`` == feishu ``--connect``).
    The feishu unit is skipped (with a note) when ``agentm-feishu`` is not on
    PATH (run ``uv sync --all-packages``).
    """
    import subprocess

    plan = _build_systemd_plan()
    sctl = _systemctl(plan)
    units = [f"{_GATEWAY_UNIT}.service", f"{_FEISHU_UNIT}.service"]

    if not install:
        for unit in units:
            subprocess.run([*sctl, "stop", unit], check=False)
            subprocess.run([*sctl, "disable", unit], check=False)
        removed = False
        for unit in units:
            path = plan.unit_dir / unit
            if path.exists():
                path.unlink()
                sys.stdout.write(f"Removed {path}\n")
                removed = True
        if removed:
            subprocess.run([*sctl, "daemon-reload"], check=False)
        else:
            sys.stdout.write(
                f"No agentm units in {plan.unit_dir}, nothing to remove.\n"
            )
        return

    plan.unit_dir.mkdir(parents=True, exist_ok=True)
    gateway_path = plan.unit_dir / f"{_GATEWAY_UNIT}.service"
    gateway_path.write_text(_render_gateway_unit(plan), encoding="utf-8")
    sys.stdout.write(f"Wrote {gateway_path}\n")

    enable_units = [f"{_GATEWAY_UNIT}.service"]
    if plan.feishu_bin is not None:
        feishu_path = plan.unit_dir / f"{_FEISHU_UNIT}.service"
        feishu_path.write_text(_render_feishu_unit(plan), encoding="utf-8")
        sys.stdout.write(f"Wrote {feishu_path}\n")
        enable_units.append(f"{_FEISHU_UNIT}.service")
    else:
        sys.stdout.write(
            "NOTE: agentm-feishu not found; skipping the feishu unit. Run "
            "`uv sync --all-packages` then re-run --install-systemd to add it.\n"
        )

    if not plan.env_file.exists():
        sys.stdout.write(
            f"NOTE: {plan.env_file} is missing — the feishu client needs "
            "LARK_APP_ID / LARK_APP_SECRET there (and model creds for the "
            "gateway). Run `agentm onboard` or create it, then restart.\n"
        )

    subprocess.run([*sctl, "daemon-reload"], check=True)
    subprocess.run([*sctl, "enable", "--now", *enable_units], check=True)

    j = "journalctl --user" if not plan.system else "journalctl"
    sys.stdout.write(
        f"\nInstalled and started: {', '.join(enable_units)}\n"
        f"  {j} -u {_GATEWAY_UNIT} -f    # follow gateway logs\n"
        f"  {' '.join(sctl)} status {_GATEWAY_UNIT}\n"
        f"  contrib/gateway-peers/deploy/update.sh   # pull latest + restart\n"
        f"  agentm gateway --uninstall-systemd       # remove\n"
    )
    if not plan.system:
        sys.stdout.write(
            "NOTE: user units stop at logout unless lingering is enabled:\n"
            f"  sudo loginctl enable-linger {os.environ.get('USER', '$USER')}\n"
        )


# -------- typer app ----------------------------------------------------


app = typer.Typer(
    name=PROG,
    add_completion=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def cli(
    cwd: Annotated[
        str,
        typer.Option("--cwd", envvar="AGENTM_CWD", help="Working directory for sessions."),
    ] = "",
    scenario: Annotated[
        str,
        typer.Option("--scenario", envvar="AGENTM_SCENARIO", help="Default scenario."),
    ] = "local",
    state_dir: Annotated[
        Path | None,
        typer.Option(
            "--state-dir",
            envvar="AGENTM_STATE_DIR",
            help="Persistent state (outbox/inbox/session_map). Default: <cwd>/.agentm/gateway.",
        ),
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", envvar="AGENTM_PROVIDER")
    ] = None,
    model: Annotated[str | None, typer.Option("--model", envvar="AGENTM_MODEL")] = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option("--reasoning-effort", envvar="AGENTM_REASONING_EFFORT"),
    ] = None,
    log_level: Annotated[
        str, typer.Option("--log-level", envvar="AGENTM_LOG_LEVEL")
    ] = "INFO",
    bind: Annotated[
        str | None,
        typer.Option(
            "--bind",
            envvar="AGENTM_SOCKET",
            help=(
                "Wire-server URL. unix:///abs/path (peer-cred, current uid), "
                "ws://host:port/path (token), wss://host:port/path (token+TLS)."
            ),
        ),
    ] = None,
    bind_token_file: Annotated[Path | None, typer.Option("--bind-token-file")] = None,
    bind_allow_anonymous: Annotated[
        bool, typer.Option("--bind-allow-anonymous")
    ] = False,
    tls_cert: Annotated[Path | None, typer.Option("--tls-cert")] = None,
    tls_key: Annotated[Path | None, typer.Option("--tls-key")] = None,
    bind_allow_uid: Annotated[
        list[int] | None, typer.Option("--bind-allow-uid", metavar="UID")
    ] = None,
    bind_allow_any_uid: Annotated[
        bool, typer.Option("--bind-allow-any-uid")
    ] = False,
    require_approval: Annotated[
        list[str] | None,
        typer.Option(
            "--require-approval",
            metavar="TOOL",
            help="Tool that needs a button press before it runs (repeatable; '*' = all).",
        ),
    ] = None,
    atoms_allow: Annotated[
        list[str] | None,
        typer.Option(
            "--atom-allow",
            metavar="ATOM",
            help="Surface /atom:install for this atom (repeatable; '*' = all mountable).",
        ),
    ] = None,
    check: Annotated[
        bool,
        typer.Option("--check", help="Validate config and exit (no server, no LLM)."),
    ] = False,
    install_systemd: Annotated[
        bool,
        typer.Option(
            "--install-systemd",
            help=(
                "Install systemd USER units for the gateway AND feishu client "
                "(~/.config/systemd/user), enable + start them, then exit."
            ),
        ),
    ] = False,
    uninstall_systemd: Annotated[
        bool,
        typer.Option(
            "--uninstall-systemd",
            help="Stop, disable and remove the agentm systemd units, then exit.",
        ),
    ] = False,
) -> None:
    """Single-process gateway: hold all chat sessions and serve chat clients."""
    if install_systemd or uninstall_systemd:
        _systemd_action(install=install_systemd)
        return
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    resolved_cwd = cwd or str(Path.cwd())
    try:
        rc = asyncio.run(
            _arun(
                cwd=resolved_cwd,
                scenario=scenario,
                state_dir=state_dir,
                provider_flag=provider,
                model_flag=model,
                reasoning_effort=reasoning_effort,
                bind=bind,
                bind_token_file=str(bind_token_file) if bind_token_file else None,
                bind_allow_anonymous=bind_allow_anonymous,
                tls_cert=str(tls_cert) if tls_cert else None,
                tls_key=str(tls_key) if tls_key else None,
                bind_allow_uid=bind_allow_uid,
                bind_allow_any_uid=bind_allow_any_uid,
                require_approval=require_approval,
                atoms_allow=atoms_allow,
                check=check,
            )
        )
    except KeyboardInterrupt:
        rc = EXIT_SIGINT
    except SystemExit as exc:
        if isinstance(exc.code, int):
            rc = exc.code
        else:
            sys.stderr.write(f"{exc.code}\n")
            rc = EXIT_CONFIG_ERROR
    raise typer.Exit(code=rc)


async def _arun(
    *,
    cwd: str,
    scenario: str,
    state_dir: Path | None,
    provider_flag: str | None,
    model_flag: str | None,
    reasoning_effort: str | None,
    bind: str | None,
    bind_token_file: str | None,
    bind_allow_anonymous: bool,
    tls_cert: str | None,
    tls_key: str | None,
    bind_allow_uid: list[int] | None,
    bind_allow_any_uid: bool,
    require_approval: list[str] | None,
    atoms_allow: list[str] | None,
    check: bool,
) -> int:
    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.lib.user_config import resolve_model_profile

    bind_spec = _resolve_bind(
        bind=bind,
        bind_allow_uid=bind_allow_uid,
        bind_allow_any_uid=bind_allow_any_uid,
        bind_token_file=bind_token_file,
        bind_allow_anonymous=bind_allow_anonymous,
        tls_cert=tls_cert,
        tls_key=tls_key,
    )

    raw_model = model_flag
    profile = resolve_model_profile(raw_model if isinstance(raw_model, str) else None)
    if profile is not None:
        resolved_provider = provider_flag or profile.provider
        resolved_model = profile.model
    else:
        resolved_provider = (
            provider_flag or DEFAULT_PROVIDER_REGISTRY.default_provider().id
        )
        resolved_model = raw_model or DEFAULT_PROVIDER_REGISTRY.default_model(
            resolved_provider
        )

    resolved_state_dir = state_dir or (Path(cwd) / ".agentm" / "gateway")

    if check:
        payload = {
            "kind": "check",
            "scenario": scenario,
            "state_dir": str(resolved_state_dir),
            "bind": {"scheme": bind_spec.scheme, "url": bind_spec.url or f"unix://{bind_spec.socket_path}"},
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
        return EXIT_OK

    resolved_state_dir.mkdir(parents=True, exist_ok=True)
    outbox = SqliteOutbox(str(resolved_state_dir / "wire-outbox.sqlite"))
    inbox = SqliteInbox(str(resolved_state_dir / "wire-inbox.sqlite"))
    chat_map = ChatSessionMap(resolved_state_dir / "session_map.json")
    command_registry = discover_commands(
        cwd,
        atom_commands_enabled=bool(atoms_allow),
        atom_allow=atoms_allow or [],
    )
    approval_policy: tuple[frozenset[str], frozenset[str], float] = (
        frozenset(require_approval or ()),
        frozenset(),
        300.0,
    )
    # Builder the runtime calls on ``/model <name>`` to produce a fresh factory
    # for a named profile (the command validates the name exists first).
    def make_factory(model_name: str) -> Any:
        prof = resolve_model_profile(model_name)
        if prof is None:  # pragma: no cover — caller validates
            raise ValueError(f"no model profile {model_name!r}")
        return _build_session_factory(
            provider=provider_flag or prof.provider,
            model=prof.model,
            profile=prof,
            reasoning_effort=reasoning_effort,
        )

    from agentm.core.lib.user_config import load_user_config

    initial_model_label = (
        raw_model or load_user_config().default_model or resolved_model
    )
    runtime = _GatewayRuntime(
        cwd=cwd,
        scenario=scenario,
        outbox=outbox,
        chat_map=chat_map,
        session_factory=_build_session_factory(
            provider=resolved_provider,
            model=resolved_model,
            profile=profile,
            reasoning_effort=reasoning_effort,
        ),
        command_router=CommandRouter(registry=command_registry),
        approval_policy=approval_policy,
        model_name=str(initial_model_label or ""),
        make_factory=make_factory,
    )
    server = WireServer(
        transport=_build_server_transport(bind_spec),
        outbox=outbox,
        inbox=inbox,
        on_inbound=runtime.handle_inbound,
        authenticator=_build_authenticator(bind_spec),
    )
    runtime.attach_server(server)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover — Windows
            pass

    await server.start()
    if bind_spec.scheme == "unix":
        logger.info("gateway bound at unix://%s", bind_spec.socket_path)
    else:
        logger.info("gateway bound at %s", bind_spec.url)

    try:
        await stop_event.wait()
    finally:
        logger.info("gateway shutting down")
        await server.stop()
        await runtime.shutdown()
        outbox.close()
        inbox.close()
    return EXIT_OK


def main() -> None:
    """Entry point for the ``agentm gateway`` subcommand."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
