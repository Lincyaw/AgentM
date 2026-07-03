"""``agentm gateway`` subcommand — the single-process gateway CLI.

Parses CLI flags, wires together the runtime components, and runs the
gateway server.  Runtime orchestration lives in :mod:`gateway.runtime`;
systemd integration lives in :mod:`gateway.systemd`.

Run as a long-lived daemon and connect chat clients separately::

    agentm gateway --bind unix:///tmp/gw.sock
    agentm-terminal --connect unix:///tmp/gw.sock
"""

from __future__ import annotations

import asyncio
import json
import logging as _stdlib_logging
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

import typer
from loguru import logger

from agentm.gateway import autoload_dotenv, default_socket_url, load_token_file
from agentm.gateway.auth import (
    Authenticator,
    TokenAuthenticator,
    UnixPeerCredAuthenticator,
)
from agentm.gateway.chat_session_map import ChatSessionMap
from agentm.gateway.commands import (
    CommandRouter,
    discover_commands,
)
from agentm.gateway.outbox import SqliteInbox, SqliteOutbox
from agentm.gateway.runtime import GatewayRuntime
from agentm.gateway.scheduler import GatewayScheduleStore
from agentm.gateway.server import WireServer
from agentm.gateway.systemd import systemd_action
from agentm.gateway.transport import (
    ServerTransport,
    UnixServerTransport,
    WebSocketServerTransport,
)
from agentm.gateway.workspace import WorkspaceResolver, load_gateway_config

autoload_dotenv()

PROG = "agentm gateway"

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_SIGINT = 130


def _restore_terminal() -> None:
    """Best-effort ``stty sane`` so Ctrl-C works even if a prior program
    (e.g. a crashed TUI) left the PTY in raw mode."""
    if not sys.stdin.isatty():
        return
    try:
        import subprocess
        subprocess.run(["stty", "sane"], stdin=sys.stdin, check=False)  # noqa: S603, S607
    except Exception as exc:  # noqa: BLE001
        logger.debug("stty sane failed (harmless): {}", exc)


class _InterceptHandler(_stdlib_logging.Handler):
    def emit(self, record: _stdlib_logging.LogRecord) -> None:
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame = _stdlib_logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == _stdlib_logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# -------- bind resolution ---------------------------------------------


@dataclass(frozen=True, slots=True)
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
        return set(load_token_file(path, option_name="--bind-token-file"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


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
    """Merge CLI flags > shared default into a :class:`BindSpec`."""
    url = bind or default_socket_url(create_runtime_dir=True)
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
    *,
    provider: str,
    model: str,
    profile: Any | None,
    reasoning_effort: str | None = None,
    workspace: WorkspaceResolver | None = None,
) -> Callable[[str, str, str | None, str | None, dict[str, Any]], Awaitable[Any]]:
    from typing import cast as _cast

    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.lib.user_config import ModelBuildConfig, apply_reasoning_effort
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
        del session_key
        if workspace is not None and workspace.active:
            channel = (wire_services.get("turn_context") or {}).get("channel", "")
            cwd = workspace.resolve(channel)
        store = make_default_session_store(cwd)
        # A /fork seeds fork_source / fork_up_to into wire_services; when present
        # resolve_session_state branches to SessionStore.fork (a new session
        # seeded from the source transcript) instead of resume/create.
        fork_source = wire_services.get("fork_source")
        fork_up_to = wire_services.get("fork_up_to")
        state = resolve_session_state(
            cwd=cwd,
            resume=resume,
            continue_recent=False,
            session_store=store,
            fork=fork_source,
            fork_up_to=fork_up_to,
        )
        header = getattr(state, "get_header", lambda: None)()
        stored_config = getattr(header, "config", None) if header is not None else None
        stored_lineage = (
            stored_config.get("lineage") if isinstance(stored_config, dict) else None
        )
        stored_experiment = (
            stored_config.get("experiment") if isinstance(stored_config, dict) else None
        )
        if fork_source:
            fork_point: dict[str, Any] = {
                "up_to": fork_up_to if fork_up_to is not None else "end"
            }
            lineage: dict[str, Any] = {
                "kind": "fork",
                "entrypoint": "agentm.gateway",
                "source_session_id": fork_source,
                "fork_point": fork_point,
            }
            if isinstance(stored_lineage, dict):
                lineage["source_lineage"] = dict(stored_lineage)
        elif isinstance(stored_lineage, dict):
            lineage = dict(stored_lineage)
        else:
            lineage = {"kind": "root", "entrypoint": "agentm.gateway"}
        experiment = (
            dict(stored_experiment) if isinstance(stored_experiment, dict) else None
        )
        build_config: ModelBuildConfig
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
            initial_services=dict(wire_services),
            extra_extensions=[("agentm.extensions.builtin.wire_driver", {})],
            parent_session_id=fork_source if fork_source else None,
            lineage=lineage,
            experiment=experiment,
        )
        return await AgentSession.create(config)

    return factory


def _validate_scenario(scenario: str) -> None:
    from agentm.extensions.loader import ScenarioLoadError, validate_scenario

    try:
        validate_scenario(scenario)
    except ScenarioLoadError as exc:
        raise SystemExit(f"--scenario {scenario!r}: {exc}") from exc


# -------- typer app ----------------------------------------------------


app = typer.Typer(
    name=PROG,
    add_completion=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback(invoke_without_command=True)
def cli(
    cwd: Annotated[
        str,
        typer.Option("--cwd", envvar="AGENTM_CWD", help="Working directory for sessions."),
    ] = "",
    scenario: Annotated[
        str,
        typer.Option("--scenario", envvar="AGENTM_SCENARIO", help="Default scenario."),
    ] = "chatbot",
    state_dir: Annotated[
        Path | None,
        typer.Option(
            "--state-dir",
            envvar="AGENTM_STATE_DIR",
            help="Persistent state (outbox/inbox/session_map). Default: $AGENTM_HOME/gateway.",
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
        systemd_action(install=install_systemd)
        return
    log_level = str(log_level).upper()
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level: <7}</level> <cyan>{file}:{line}</cyan> <level>{message}</level>")
    # Also ship operational logs into the OTel logs pipeline (→ collector →
    # ClickHouse) when an OTLP endpoint is configured. No-op otherwise.
    from agentm.core.observability.otel_export import attach_loguru_otel_sink
    attach_loguru_otel_sink(level=log_level)
    _stdlib_logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    # Ensure the terminal is in cooked mode so Ctrl-C generates SIGINT.
    # A prior process (e.g. a TUI that crashed) may have left the PTY in
    # raw mode where intr is disabled; without this, Ctrl-C is swallowed.
    _restore_terminal()
    resolved_cwd = str(Path(cwd or str(Path.cwd())).expanduser())
    autoload_dotenv(Path(resolved_cwd))
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
    from agentm.core.lib.user_config import (
        agentm_home_dir,
        resolve_model_profile,
        resolve_provider_model,
    )

    bind_spec = _resolve_bind(
        bind=bind,
        bind_allow_uid=bind_allow_uid,
        bind_allow_any_uid=bind_allow_any_uid,
        bind_token_file=bind_token_file,
        bind_allow_anonymous=bind_allow_anonymous,
        tls_cert=tls_cert,
        tls_key=tls_key,
    )

    resolved_provider, resolved_model, profile = resolve_provider_model(
        provider_flag=provider_flag,
        model_flag=model_flag,
    )
    raw_model = model_flag

    resolved_state_dir = state_dir or (agentm_home_dir() / "gateway")
    _validate_scenario(scenario)

    if check:
        payload = {
            "kind": "check",
            "scenario": scenario,
            "state_dir": str(resolved_state_dir),
            "bind": {"scheme": bind_spec.scheme, "url": bind_spec.url or f"unix://{bind_spec.socket_path}"},
            "schedule_store": str(resolved_state_dir / "schedules.json"),
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
        return EXIT_OK

    resolved_state_dir.mkdir(parents=True, exist_ok=True)
    outbox = SqliteOutbox(str(resolved_state_dir / "wire-outbox.sqlite"))
    inbox = SqliteInbox(str(resolved_state_dir / "wire-inbox.sqlite"))
    chat_map = ChatSessionMap(resolved_state_dir / "session_map.json")
    schedule_store = GatewayScheduleStore(resolved_state_dir / "schedules.json")
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
    workspace = load_gateway_config(cwd)

    def make_factory(model_name: str) -> Any:
        prof = resolve_model_profile(model_name)
        if prof is None:  # pragma: no cover — caller validates
            raise ValueError(f"no model profile {model_name!r}")
        return _build_session_factory(
            provider=provider_flag or prof.provider,
            model=prof.model,
            profile=prof,
            reasoning_effort=reasoning_effort,
            workspace=workspace,
        )

    from agentm.core.lib.user_config import load_user_config

    initial_model_label = (
        raw_model or load_user_config().default_model or resolved_model
    )
    runtime = GatewayRuntime(
        cwd=cwd,
        scenario=scenario,
        outbox=outbox,
        chat_map=chat_map,
        session_factory=_build_session_factory(
            provider=resolved_provider,
            model=resolved_model,
            profile=profile,
            reasoning_effort=reasoning_effort,
            workspace=workspace,
        ),
        command_router=CommandRouter(registry=command_registry),
        approval_policy=approval_policy,
        model_name=str(initial_model_label or ""),
        make_factory=make_factory,
        schedule_store=schedule_store,
    )
    server = WireServer(
        transport=_build_server_transport(bind_spec),
        outbox=outbox,
        inbox=inbox,
        on_inbound=runtime.handle_inbound,
        authenticator=_build_authenticator(bind_spec),
        capabilities_provider=runtime.describe_capabilities,
    )
    runtime.attach_server(server)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    signal_count = 0

    def _on_signal(_signum: int, _frame: Any) -> None:
        # First signal starts a graceful drain; a second one means the user is
        # impatient (or the drain is wedged on a stuck in-flight turn), so hard
        # exit instead of swallowing it into another no-op ``stop_event.set()``.
        #
        # This is a low-level ``signal.signal`` handler, NOT
        # ``loop.add_signal_handler``, and that distinction is the whole point:
        # an add_signal_handler callback only runs when the event loop reaches
        # its next scheduling point. If an in-flight turn wedges the loop
        # thread on a synchronous blocking call, that callback never fires —
        # and neither do the ``wait_for`` drain timeouts below — so Ctrl-C gets
        # swallowed no matter how many times it is pressed. A signal.signal
        # handler runs synchronously in the main thread the instant the signal
        # lands, so the second press can force an exit even past a wedged loop.
        nonlocal signal_count
        signal_count += 1
        if signal_count == 1:
            # stop_event.set() is not safe to call directly from a handler that
            # may have interrupted the loop mid-step; hop back onto the loop.
            # os.write to fd 2 is async-signal-safe (loguru's lock is not).
            loop.call_soon_threadsafe(stop_event.set)
            os.write(2, b"signal received - draining; press Ctrl-C again to force exit\n")
        else:
            os.write(2, b"second signal received - forcing exit\n")
            os._exit(EXIT_SIGINT)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError) as exc:  # pragma: no cover — non-main thread / Windows
            # Signal handlers can't be set off the main thread / on Windows;
            # the gateway still runs, just without graceful-drain on this signal.
            logger.debug("gateway: could not install handler for {}: {}", sig, exc)

    await server.start()
    runtime.start_scheduler()
    if bind_spec.scheme == "unix":
        logger.info(f"gateway bound at unix://{bind_spec.socket_path}")
    else:
        logger.info(f"gateway bound at {bind_spec.url}")

    try:
        await stop_event.wait()
    finally:
        logger.info("gateway shutting down")
        # Bound the drain: an in-flight turn wedged in a blocking LLM request or
        # subprocess won't honour cancellation, and an unbounded gather would
        # hang the process — and then re-hang asyncio.run's own task cleanup.
        timed_out = False
        try:
            await asyncio.wait_for(server.stop(), timeout=5.0)
            await asyncio.wait_for(runtime.shutdown(), timeout=10.0)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("graceful shutdown timed out — forcing exit")
            timed_out = True
        # Close the SQLite stores cleanly before any hard exit — ``os._exit``
        # below bypasses every ``finally``, so this must run first.
        outbox.close()
        inbox.close()
        if timed_out:
            os._exit(EXIT_SIGINT)
    return EXIT_OK


def main() -> None:
    """Entry point for the ``agentm gateway`` subcommand."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
