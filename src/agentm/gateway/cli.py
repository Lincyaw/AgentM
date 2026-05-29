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
    CommandContext,
    CommandInbound,
    CommandRouter,
    discover_commands,
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
    *, provider: str, model: str, profile: Any | None
) -> Callable[[str, str, str | None, str | None, dict[str, Any]], Awaitable[Any]]:
    from typing import cast as _cast

    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
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
            provider_spec = DEFAULT_PROVIDER_REGISTRY.build(
                provider, profile.to_build_config()
            )
        else:
            provider_spec = DEFAULT_PROVIDER_REGISTRY.build(provider, {"model": model})
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
            # Parsed as not-a-command after all; treat as a normal prompt.
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

_UNIT_NAME = "agentm-gateway"


def _systemd_action(*, install: bool) -> None:
    """Install or uninstall a systemd user unit for the gateway.

    Install reconstructs the current ``agentm gateway …`` invocation
    (minus ``--install-systemd`` itself) from ``sys.argv`` and bakes it
    into ``ExecStart``. The unit inherits the current environment via an
    ``EnvironmentFile`` pointing at ``~/.config/agentm/gateway.env``.
    """
    import shutil
    import subprocess
    import textwrap

    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_path = unit_dir / f"{_UNIT_NAME}.service"

    if not install:
        subprocess.run(
            ["systemctl", "--user", "stop", _UNIT_NAME],
            check=False,
        )
        subprocess.run(
            ["systemctl", "--user", "disable", _UNIT_NAME],
            check=False,
        )
        if unit_path.exists():
            unit_path.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            sys.stdout.write(f"Removed {unit_path}\n")
        else:
            sys.stdout.write(f"{unit_path} does not exist, nothing to remove.\n")
        return

    agentm_bin = shutil.which("agentm")
    if agentm_bin is None:
        raise SystemExit(
            "Cannot find `agentm` on $PATH. Install it first or run from "
            "the venv (`uv run agentm gateway --install-systemd`)."
        )

    argv = sys.argv[:]
    cleaned: list[str] = []
    for arg in argv:
        if arg in ("--install-systemd", "--uninstall-systemd"):
            continue
        cleaned.append(arg)
    # sys.argv[0] is "agentm gateway" (merged by the main CLI dispatcher);
    # the remaining elements are the gateway flags.  Reconstruct a clean
    # command line using the resolved binary path.
    exec_start = f"{agentm_bin} gateway {' '.join(cleaned[1:])}"

    env_dir = Path.home() / ".config" / "agentm"
    env_file = env_dir / "gateway.env"

    unit = textwrap.dedent(f"""\
        [Unit]
        Description=AgentM Gateway
        After=network.target

        [Service]
        Type=simple
        ExecStart={exec_start}
        Restart=on-failure
        RestartSec=3
        EnvironmentFile=-{env_file}
        WorkingDirectory={Path.cwd()}

        [Install]
        WantedBy=default.target
    """)

    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(unit, encoding="utf-8")
    sys.stdout.write(f"Wrote {unit_path}\n")

    if not env_file.exists():
        env_dir.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            "# Environment variables for agentm-gateway.\n"
            "# e.g. AGENTM_MODEL=doubao\n",
            encoding="utf-8",
        )
        sys.stdout.write(f"Created {env_file} (edit to add API keys / env vars)\n")

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", _UNIT_NAME], check=True)
    sys.stdout.write(
        f"\n{_UNIT_NAME} is running. Useful commands:\n"
        f"  journalctl --user -u {_UNIT_NAME} -f    # follow logs\n"
        f"  systemctl --user status {_UNIT_NAME}     # check status\n"
        f"  systemctl --user restart {_UNIT_NAME}    # restart\n"
        f"  agentm gateway --uninstall-systemd       # remove\n"
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
    ] = "general_purpose",
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
            help="Generate a systemd user unit, enable and start it, then exit.",
        ),
    ] = False,
    uninstall_systemd: Annotated[
        bool,
        typer.Option(
            "--uninstall-systemd",
            help="Stop, disable and remove the systemd user unit, then exit.",
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
            provider=provider_flag or prof.provider, model=prof.model, profile=prof
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
            provider=resolved_provider, model=resolved_model, profile=profile
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
