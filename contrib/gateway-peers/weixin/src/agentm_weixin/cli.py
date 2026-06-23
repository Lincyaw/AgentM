"""``agentm-weixin`` — WeChat client process for the AgentM gateway.

Connects to an ``agentm-gateway --bind unix://…`` over the v2 wire
protocol, brings up the WeChat adapter (iLink Bot API long-poll), and
proxies inbound messages into the gateway and outbound messages back to
WeChat as plain text.

CLI design follows ``autoharness:cli-design``:

* All logs go to stderr. Stdout is unused (daemon).
* Exit codes are stable and standardized.
* Secrets never appear in argv. The bot token comes from persistent
  state (QR login) or the ``WEIXIN_TOKEN`` env variable.

Examples::

    # First-time login (interactive QR scan):
    agentm-weixin login

    # Run with default gateway socket:
    agentm-weixin run

    # Run with explicit gateway URL:
    agentm-weixin run --connect unix:///tmp/gw.sock
"""

from __future__ import annotations

import asyncio
import contextlib
import logging as _stdlib_logging
import signal
import sys
import uuid
from types import FrameType
from typing import Annotated

import typer

from agentm.gateway import DEFAULT_SOCKET_URL, autoload_dotenv
from agentm.gateway.client import AuthError, WireClient
from agentm.gateway.client_cli import ConnectError, ConnectOptions, resolve_connect
from agentm.gateway.wire import (
    KIND_ERROR,
    KIND_OUTBOUND,
    KIND_PING,
    KIND_PONG,
    Envelope,
)

from . import __version__
from loguru import logger

autoload_dotenv()

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_USAGE = 2
EXIT_AUTH = 4
EXIT_SIGINT = 6
EXIT_CONNECT = 7

PROG = "agentm-weixin"


def _err(kind: str, root: str, fix: str) -> None:
    sys.stderr.write(f"{PROG}: error: {kind}: {root}. {fix}.\n")
    sys.stderr.flush()


def _setup_logging(verbose: bool = False) -> None:
    """Configure loguru + stdlib intercept (shared by run and serve)."""

    class _InterceptHandler(_stdlib_logging.Handler):
        def emit(self, record: _stdlib_logging.LogRecord) -> None:
            level: str | int
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame: FrameType | None
            frame, depth = _stdlib_logging.currentframe(), 2
            while frame and frame.f_code.co_filename == _stdlib_logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO" if verbose else "WARNING",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level: <7}</level> "
            "<cyan>{file}:{line}</cyan> <level>{message}</level>"
        ),
    )
    _stdlib_logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{PROG} {__version__}")
        raise typer.Exit(code=EXIT_OK)


def _resolve_connect(url: str, tls_ca: str | None):
    try:
        return resolve_connect(url, tls_ca=tls_ca)
    except ConnectError as exc:
        _err("bad-argument", str(exc), "see --help for the supported schemes")
        raise typer.Exit(code=EXIT_USAGE) from exc


app = typer.Typer(
    name=PROG,
    add_completion=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)


# -- login subcommand --------------------------------------------------


@app.command()
def login(
    account_id: Annotated[
        str | None,
        typer.Option("--account-id", help="Account ID for re-login (optional)."),
    ] = None,
    _version: Annotated[
        bool,
        typer.Option("--version", is_eager=True, callback=_version_callback),
    ] = False,
) -> None:
    """Scan a WeChat QR code to connect this AgentM to WeChat."""
    import aiohttp  # noqa: PLC0415

    from .auth import login_qr  # noqa: PLC0415

    async def _do_login() -> int:
        async with aiohttp.ClientSession() as session:
            result = await login_qr(session, account_id=account_id)
            if result.connected:
                sys.stderr.write(f"\n{result.message}\n")
                sys.stderr.write(f"Account ID: {result.account_id}\n")
                sys.stderr.write("\n运行 `agentm-weixin run` 启动消息桥接\n")
                return EXIT_OK
            if result.already_connected:
                sys.stderr.write(f"\n{result.message}\n")
                return EXIT_OK
            sys.stderr.write(f"\n{result.message}\n")
            return EXIT_GENERIC

    try:
        rc = asyncio.run(_do_login())
    except KeyboardInterrupt:
        rc = EXIT_SIGINT
    raise typer.Exit(code=rc)


# -- run subcommand ----------------------------------------------------


@app.command()
def run(
    connect: Annotated[
        str,
        typer.Option(
            "--connect",
            envvar="AGENTM_SOCKET",
            metavar="URL",
            help=(
                "Gateway URL. Supported: unix:///path, ws://host:port, "
                "wss://host:port. Env: AGENTM_SOCKET."
            ),
        ),
    ] = DEFAULT_SOCKET_URL,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            envvar="AGENTM_TOKEN",
            help="Bearer token for ws/wss gateway auth. Env: AGENTM_TOKEN.",
        ),
    ] = None,
    token_file: Annotated[
        str | None,
        typer.Option(
            "--token-file",
            metavar="PATH",
            help="Read bearer token from PATH. Preferred for production.",
        ),
    ] = None,
    tls_ca: Annotated[
        str | None,
        typer.Option("--tls-ca", metavar="PATH", help="CA bundle for wss."),
    ] = None,
    account_id: Annotated[
        str | None,
        typer.Option(
            "--account-id",
            envvar="WEIXIN_ACCOUNT_ID",
            help=(
                "WeChat account ID (from login). If omitted, uses the first "
                "registered account."
            ),
        ),
    ] = None,
    channel_name: Annotated[
        str,
        typer.Option("--channel-name", help="Channel name on inbound envelopes."),
    ] = "weixin",
    scenario: Annotated[
        str | None,
        typer.Option(
            "--scenario",
            envvar="AGENTM_SCENARIO",
            help="Scenario for new sessions. Default: chatbot.",
        ),
    ] = "chatbot",
    session_scope: Annotated[
        str,
        typer.Option(
            "--session-scope",
            help="Session key scope: 'user' (per sender, default) or 'chat'.",
        ),
    ] = "user",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Raise log level to INFO."),
    ] = False,
    _version: Annotated[
        bool,
        typer.Option("--version", is_eager=True, callback=_version_callback),
    ] = False,
) -> None:
    """Run the WeChat ↔ AgentM gateway bridge.

    Requires a prior `agentm-weixin login` to obtain credentials.
    """
    from agentm.gateway import resolve_token  # noqa: PLC0415

    try:
        effective_token = resolve_token(token, token_file)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _setup_logging(verbose)

    # Resolve account
    from .state import list_account_ids, load_account  # noqa: PLC0415

    resolved_account_id = account_id
    if not resolved_account_id:
        ids = list_account_ids()
        if not ids:
            _err(
                "no-account",
                "no WeChat account registered",
                "run `agentm-weixin login` first",
            )
            raise typer.Exit(code=EXIT_USAGE)
        resolved_account_id = ids[0]
        if len(ids) > 1:
            logger.info(
                f"multiple accounts found, using first: {resolved_account_id}"
            )

    acct = load_account(resolved_account_id)
    if not acct or not acct.token:
        _err(
            "bad-account",
            f"account {resolved_account_id!r} has no token",
            "run `agentm-weixin login` to re-authenticate",
        )
        raise typer.Exit(code=EXIT_USAGE)

    try:
        rc = asyncio.run(
            _arun(
                connect_opts=ConnectOptions(
                    connect=connect, token=effective_token, tls_ca=tls_ca
                ),
                account_id=resolved_account_id,
                bot_token=acct.token,
                base_url=acct.base_url or "https://ilinkai.weixin.qq.com",
                cdn_base_url=acct.cdn_base_url or "https://novac2c.cdn.weixin.qq.com/c2c",
                channel_name=channel_name,
                scenario=scenario,
                session_scope=session_scope,
            )
        )
    except KeyboardInterrupt:
        rc = EXIT_SIGINT

    raise typer.Exit(code=rc)


async def _arun(
    *,
    connect_opts: ConnectOptions,
    account_id: str,
    bot_token: str,
    base_url: str,
    cdn_base_url: str,
    channel_name: str,
    scenario: str | None,
    session_scope: str,
) -> int:
    import aiohttp  # noqa: PLC0415

    from .adapter import WeixinAdapter, WeixinConfig  # noqa: PLC0415

    connect = connect_opts.connect
    gw_token = connect_opts.token
    _spec, transport = _resolve_connect(connect, connect_opts.tls_ca)

    peer_name = f"weixin-{uuid.uuid4().hex[:8]}"
    stop_event = asyncio.Event()
    exit_code = EXIT_OK
    adapter: WeixinAdapter | None = None

    async def on_outbound(env: Envelope) -> None:
        nonlocal exit_code
        if env.kind == KIND_OUTBOUND:
            if adapter is None:
                return
            try:
                await adapter.handle_outbound(env)
            except Exception:
                logger.exception(f"adapter.handle_outbound failed id={env.id}")
            return
        if env.kind in (KIND_PING, KIND_PONG):
            return
        if env.kind == KIND_ERROR:
            body = env.body if isinstance(env.body, dict) else {}
            code = body.get("code", "unknown")
            message = body.get("message", "")
            _err(
                "gateway-error",
                f"gateway emitted error code={code!r} message={message!r}",
                "check the gateway logs",
            )
            exit_code = EXIT_GENERIC
            stop_event.set()

    client = WireClient(
        transport=transport,
        peer_name=peer_name,
        token=gw_token,
        on_outbound=on_outbound,
    )

    try:
        await client.connect()
    except AuthError as exc:
        _err(
            "auth-failed",
            f"gateway rejected handshake (code={exc.code!r})",
            "verify gateway auth configuration",
        )
        return EXIT_AUTH
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        _err(
            "connect-failed",
            f"cannot connect to {connect!r} ({exc.__class__.__name__})",
            "is the gateway running",
        )
        return EXIT_CONNECT
    except OSError as exc:
        _err(
            "connect-failed",
            f"cannot connect to {connect!r}: {exc.strerror or exc}",
            "check the socket path and gateway state",
        )
        return EXIT_CONNECT

    cfg = WeixinConfig(
        account_id=account_id,
        token=bot_token,
        base_url=base_url,
        cdn_base_url=cdn_base_url,
        channel_name=channel_name,
        scenario=scenario,
        session_scope=session_scope,
    )

    async with aiohttp.ClientSession() as http_session:
        adapter = WeixinAdapter(client=client, config=cfg, http_session=http_session)

        sigint_seen = False

        def _on_sigint() -> None:
            nonlocal sigint_seen
            sigint_seen = True
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig, _on_sigint if sig == signal.SIGINT else stop_event.set
                )
            except (NotImplementedError, RuntimeError):
                pass

        adapter_task = asyncio.create_task(adapter.start(), name="weixin-adapter")
        stop_task = asyncio.create_task(stop_event.wait(), name="weixin-stop")
        reconnect_task = asyncio.create_task(
            client.run_reconnecting(connect_first=False), name="weixin-reconnect"
        )

        try:
            done, pending = await asyncio.wait(
                [adapter_task, stop_task, reconnect_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if adapter_task in done and adapter_task.exception() is not None:
                adapter_exc = adapter_task.exception()
                assert adapter_exc is not None
                _err(
                    "weixin-error",
                    f"adapter failed: {adapter_exc.__class__.__name__}: {adapter_exc}",
                    "check WeChat credentials and network",
                )
                return EXIT_AUTH
            if reconnect_task in done and reconnect_task.exception() is not None:
                reconnect_exc = reconnect_task.exception()
                assert reconnect_exc is not None
                if isinstance(reconnect_exc, AuthError):
                    _err(
                        "auth-failed",
                        f"gateway rejected reconnect (code={reconnect_exc.code!r})",
                        "verify gateway auth",
                    )
                    return EXIT_AUTH
                _err(
                    "gateway-error",
                    f"reconnect failed: {reconnect_exc.__class__.__name__}: {reconnect_exc}",
                    "check gateway state",
                )
                return EXIT_GENERIC
            for t in pending:
                t.cancel()
        finally:
            try:
                await adapter.stop()
            except Exception:
                logger.exception("adapter.stop raised")
            await client.close()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reconnect_task

    if sigint_seen:
        return EXIT_SIGINT
    return exit_code


# -- serve subcommand (supervisor) -------------------------------------


@app.command()
def serve(
    bind: Annotated[
        str,
        typer.Option(
            "--bind",
            metavar="URL",
            help=(
                "Gateway bind address. Default: unix socket under XDG_RUNTIME_DIR."
            ),
        ),
    ] = DEFAULT_SOCKET_URL,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            envvar="AGENTM_MODEL",
            help="Model profile for the gateway (from config.toml).",
        ),
    ] = None,
    account_id: Annotated[
        str | None,
        typer.Option(
            "--account-id",
            envvar="WEIXIN_ACCOUNT_ID",
            help="WeChat account ID. If omitted, uses the first registered.",
        ),
    ] = None,
    channel_name: Annotated[
        str,
        typer.Option("--channel-name", help="Channel name on inbound envelopes."),
    ] = "weixin",
    scenario: Annotated[
        str | None,
        typer.Option(
            "--scenario",
            envvar="AGENTM_SCENARIO",
            help="Scenario for new chat sessions. Default: chatbot.",
        ),
    ] = "chatbot",
    session_scope: Annotated[
        str,
        typer.Option("--session-scope", help="'user' (default) or 'chat'."),
    ] = "user",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Raise log level to INFO."),
    ] = False,
    _version: Annotated[
        bool,
        typer.Option("--version", is_eager=True, callback=_version_callback),
    ] = False,
) -> None:
    """Start gateway + WeChat adapter under supervisord.

    One command to run everything:

      agentm-weixin serve --model doubao

    Uses supervisord under the hood:
      - Auto-restarts either process on crash
      - Log rotation under ~/.agentm/weixin/logs/
      - Use `supervisorctl -c ~/.agentm/weixin/supervisord.conf` to manage
      - Ctrl-C stops both gracefully

    The gateway starts first (priority=10), the adapter connects with
    built-in reconnect logic so it tolerates a brief startup delay.
    """
    _setup_logging(verbose)

    from .state import list_account_ids, load_account  # noqa: PLC0415

    resolved_account_id = account_id
    if not resolved_account_id:
        ids = list_account_ids()
        if not ids:
            _err(
                "no-account",
                "no WeChat account registered",
                "run `agentm-weixin login` first",
            )
            raise typer.Exit(code=EXIT_USAGE)
        resolved_account_id = ids[0]

    acct = load_account(resolved_account_id)
    if not acct or not acct.token:
        _err(
            "bad-account",
            f"account {resolved_account_id!r} has no token",
            "run `agentm-weixin login` to re-authenticate",
        )
        raise typer.Exit(code=EXIT_USAGE)

    from .supervisor import generate_config, launch_supervisord, write_config  # noqa: PLC0415

    config_text = generate_config(
        bind_url=bind,
        model=model,
        gateway_scenario=scenario,
        account_id=resolved_account_id,
        channel_name=channel_name,
        adapter_scenario=scenario,
        session_scope=session_scope,
    )
    config_path = write_config(config_text)

    sys.stderr.write(f"\n{'='*60}\n")
    sys.stderr.write("  agentm-weixin serve\n")
    sys.stderr.write(f"  config:  {config_path}\n")
    sys.stderr.write(f"  manage:  supervisorctl -c {config_path} status\n")
    sys.stderr.write(f"{'='*60}\n\n")
    sys.stderr.flush()

    # execvp replaces this process with supervisord (never returns)
    launch_supervisord(config_path)


# -- list subcommand ---------------------------------------------------


@app.command(name="list")
def list_accounts(
    _version: Annotated[
        bool,
        typer.Option("--version", is_eager=True, callback=_version_callback),
    ] = False,
) -> None:
    """List registered WeChat accounts."""
    from .state import list_account_ids, load_account  # noqa: PLC0415

    ids = list_account_ids()
    if not ids:
        sys.stderr.write("No accounts registered. Run `agentm-weixin login` first.\n")
        raise typer.Exit(code=EXIT_OK)

    for aid in ids:
        acct = load_account(aid)
        status = "configured" if acct and acct.token else "no token"
        user = acct.user_id if acct else ""
        typer.echo(f"  {aid}  ({status}){f'  user={user}' if user else ''}")


# -- entrypoint --------------------------------------------------------


def main() -> None:
    """Entry point referenced by the ``agentm-weixin`` console script."""
    app()


if __name__ == "__main__":
    main()
