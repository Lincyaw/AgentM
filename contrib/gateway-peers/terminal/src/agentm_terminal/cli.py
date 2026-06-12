"""``agentm-terminal`` — interactive terminal chat-client peer for the gateway.

Connects to an ``agentm gateway`` over the v2 wire protocol (``unix://`` local,
``ws://``/``wss://`` remote) and renders the session's event stream. On a TTY it
opens the rich Textual TUI (``frontends.tui``); piped, it falls back to the
``json`` line protocol; ``--format text|json|tui`` overrides.

CLI design follows ``autoharness:cli-design``: logs to stderr, stdout carries
business data only, stable exit codes, structured error lines.
"""

from __future__ import annotations

import asyncio
import logging as _stdlib_logging
import os
import signal
import sys
import uuid
from typing import Annotated, Any

import typer

from agentm.gateway import DEFAULT_SOCKET_URL, autoload_dotenv
from agentm.gateway.client import AuthError
from agentm.gateway.client_cli import ConnectError, ConnectOptions, resolve_connect

from loguru import logger

from . import __version__
from .client import TerminalClient

autoload_dotenv()

# -- exit codes (cli-design rule group 3) ------------------------------
EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_USAGE = 2
EXIT_AUTH = 4
EXIT_SIGINT = 6
EXIT_CONNECT = 7

PROG = "agentm-terminal"
_FORMATS = ("text", "json", "tui")



def _err(kind: str, root: str, fix: str) -> None:
    sys.stderr.write(f"{PROG}: error: {kind}: {root}. {fix}.\n")
    sys.stderr.flush()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{PROG} {__version__}")
        raise typer.Exit(code=EXIT_OK)


app = typer.Typer(
    name=PROG,
    add_completion=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _resolve_format(arg: str | None) -> str:
    if arg is not None:
        return arg
    # Default: the rich TUI on a TTY (Claude-Code-like), json when piped.
    return "tui" if sys.stdout.isatty() else "json"


def _resolve_color(no_color_flag: bool, fmt: str) -> bool:
    if fmt == "json":
        return False
    if no_color_flag or os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _resolve_connect(url: str, tls_ca: str | None) -> Any:
    try:
        return resolve_connect(url, tls_ca=tls_ca)
    except ConnectError as exc:
        _err("bad-argument", str(exc), "see --help for the supported schemes")
        raise typer.Exit(code=EXIT_USAGE) from exc


@app.command()
def cli(
    connect: Annotated[
        str,
        typer.Option(
            "--connect",
            envvar="AGENTM_SOCKET",
            metavar="URL",
            help=(
                "Gateway URL: unix:///abs/path (local, peer-cred), "
                "ws://host:port/path or wss://host:port/path (remote, token). "
                "Default: the shared local socket. Env: AGENTM_SOCKET."
            ),
        ),
    ] = DEFAULT_SOCKET_URL,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            envvar="AGENTM_TOKEN",
            help=(
                "Bearer token for ws/wss gateways. Prefer --token-file or "
                "AGENTM_TOKEN (CLI args leak into /proc). Env: AGENTM_TOKEN."
            ),
        ),
    ] = None,
    token_file: Annotated[
        str | None,
        typer.Option("--token-file", metavar="PATH", help="Read the token from PATH."),
    ] = None,
    tls_ca: Annotated[
        str | None,
        typer.Option(
            "--tls-ca", metavar="PATH", help="CA bundle (PEM) to verify a wss:// cert."
        ),
    ] = None,
    output_format: Annotated[
        str | None,
        typer.Option(
            "--format",
            envvar="AGENTM_FORMAT",
            help=(
                "Frontend: tui (rich TUI), text, json. Default: tui on a TTY, "
                "json when piped. Use json for scripts / agent drivers."
            ),
        ),
    ] = None,
    theme: Annotated[
        str,
        typer.Option("--theme", help="TUI theme: dark or light (default: dark)."),
    ] = "dark",
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable ANSI colour in text mode."),
    ] = False,
    sender_id: Annotated[
        str,
        typer.Option("--sender-id", help="Inbound sender id (default: 'local')."),
    ] = "local",
    chat_id: Annotated[
        str,
        typer.Option("--chat-id", help="Chat / session id (default: 'terminal')."),
    ] = "terminal",
    scenario: Annotated[
        str | None,
        typer.Option(
            "--scenario",
            envvar="AGENTM_SCENARIO",
            help="Scenario the gateway builds this chat in (first message only).",
        ),
    ] = None,
    no_input: Annotated[
        bool,
        typer.Option("--no-input", help="Exit after the handshake (liveness probe)."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Raise stderr log level to INFO."),
    ] = False,
    _version: Annotated[
        bool,
        typer.Option(
            "--version", is_eager=True, callback=_version_callback, help="Print version."
        ),
    ] = False,
) -> None:
    """Interactive terminal chat-client peer for the AgentM gateway."""
    from agentm.gateway import resolve_token

    try:
        effective_token = resolve_token(token, token_file)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if output_format is not None and output_format not in _FORMATS:
        _err(
            "bad-argument",
            f"--format {output_format!r} is not one of {'/'.join(_FORMATS)}",
            f"choose one of {', '.join(_FORMATS)}",
        )
        raise typer.Exit(code=EXIT_USAGE)

    class _InterceptHandler(_stdlib_logging.Handler):
        def emit(self, record: _stdlib_logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame, depth = _stdlib_logging.currentframe(), 2
            while frame and frame.f_code.co_filename == _stdlib_logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO" if verbose else "WARNING",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level: <7}</level> <cyan>{file}:{line}</cyan> <level>{message}</level>",
    )
    _stdlib_logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    try:
        rc = asyncio.run(
            _arun(
                connect_opts=ConnectOptions(
                    connect=connect, token=effective_token, tls_ca=tls_ca
                ),
                output_format=output_format,
                theme=theme,
                no_color=no_color,
                sender_id=sender_id,
                chat_id=chat_id,
                scenario=scenario,
                no_input=no_input,
            )
        )
    except KeyboardInterrupt:
        rc = EXIT_SIGINT
    raise typer.Exit(code=rc)


# -- async run ---------------------------------------------------------


async def _arun(
    *,
    connect_opts: ConnectOptions,
    output_format: str | None,
    theme: str,
    no_color: bool,
    sender_id: str,
    chat_id: str,
    scenario: str | None,
    no_input: bool,
) -> int:
    fmt = _resolve_format(output_format)
    _spec, transport = _resolve_connect(connect_opts.connect, connect_opts.tls_ca)

    client = TerminalClient(
        transport=transport,
        peer_name=f"terminal-{uuid.uuid4().hex[:8]}",
        token=connect_opts.token,
        chat_id=chat_id,
        scenario=scenario,
    )

    try:
        await client.connect()
    except AuthError as exc:
        _err(
            "auth-failed",
            f"gateway rejected handshake (code={exc.code!r})",
            "verify the gateway's auth (--bind-allow-uid / --bind-token-file)",
        )
        return EXIT_AUTH
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        _err(
            "connect-failed",
            f"cannot connect to {connect_opts.connect!r} ({exc.__class__.__name__})",
            "is the gateway running with --bind on that address",
        )
        return EXIT_CONNECT
    except OSError as exc:
        _err(
            "connect-failed",
            f"cannot connect to {connect_opts.connect!r}: {exc.strerror or exc}",
            "check the socket path and gateway state",
        )
        return EXIT_CONNECT

    if no_input:
        await client.close()
        return EXIT_OK

    if fmt == "tui":
        from .frontends.tui import run_tui

        try:
            return await run_tui(
                client=client, sender_id=sender_id, chat_id=chat_id, theme=theme
            )
        finally:
            await client.close()

    return await _run_plain(
        client=client,
        fmt=fmt,
        color=_resolve_color(no_color, fmt),
        sender_id=sender_id,
        chat_id=chat_id,
    )


async def _run_plain(
    *,
    client: TerminalClient,
    fmt: str,
    color: bool,
    sender_id: str,
    chat_id: str,
) -> int:
    from .frontends.plain import PlainRenderer

    renderer = PlainRenderer(fmt=fmt, color=color)
    stop = asyncio.Event()
    sigint_seen = False

    def _on_sigint() -> None:
        nonlocal sigint_seen
        sigint_seen = True
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig, _on_sigint if sig == signal.SIGINT else stop.set
            )
        except (NotImplementedError, RuntimeError):  # pragma: no cover — Windows
            pass

    renderer.ready()

    async def _drain_outbound() -> None:
        async for body in client.outbound():
            try:
                renderer.render_outbound(body)
            except Exception:  # noqa: BLE001
                logger.exception("renderer failed")
        stop.set()

    reader = asyncio.create_task(
        _stdin_reader(client, sender_id, chat_id, stop), name="terminal-stdin"
    )
    drain = asyncio.create_task(_drain_outbound(), name="terminal-drain")
    stop_task = asyncio.create_task(stop.wait(), name="terminal-stop")
    try:
        _done, pending = await asyncio.wait(
            [reader, drain, stop_task], return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
    finally:
        renderer.stopped()
        await client.close()
    return EXIT_SIGINT if sigint_seen else EXIT_OK


async def _stdin_reader(
    client: TerminalClient, sender_id: str, chat_id: str, stop: asyncio.Event
) -> None:
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            stop.set()
            return
        content = line.rstrip("\n")
        if not content:
            continue
        body: dict[str, Any] = {
            "channel": "terminal",
            "sender_id": sender_id,
            "chat_id": chat_id,
            "content": content,
        }
        if content.startswith("="):
            value = content[1:].strip() or None
            body["content"] = f"[button click: {value or ''}]"
            if value is not None:
                body["button_value"] = value
        try:
            await client.send_inbound(body)
        except Exception:  # noqa: BLE001
            logger.exception("failed to send inbound; closing")
            stop.set()
            return


def main() -> None:
    """Entry point for the ``agentm-terminal`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
