"""``agentm-terminal`` — stdin/stdout client for the channels gateway.

Connects to an ``agentm-gateway --bind unix://…`` over the v1 wire
protocol, reads user input from stdin, and renders gateway outbound
messages on stdout.

CLI design follows ``autoharness:cli-design``:

* All logs go to stderr. Stdout carries business data only.
* Exit codes are stable and standardized (see ``EXIT_*`` constants).
* Errors are reported as
  ``agentm-terminal: error: <type>: <root cause>. <suggested fix>.``
* Format auto-detection: ``--format`` defaults to ``text`` when stdout
  is a TTY, ``json`` otherwise.

Example::

    # Interactive:
    agentm-terminal --connect unix:///tmp/gw.sock

    # Piped (auto json mode):
    printf '/help\\n' | agentm-terminal --connect unix:///tmp/gw.sock
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from typing import Annotated, Any

import typer

from agentm_channels import DEFAULT_SOCKET_URL, autoload_dotenv
from agentm_channels.client import AuthError, WireClient
from agentm_channels.client_cli import ConnectError, resolve_connect
from agentm_channels.wire import (
    KIND_BYE,
    KIND_DELIVERY_BATCH,
    KIND_ERROR,
    KIND_OUTBOUND,
    KIND_PING,
    KIND_PONG,
    Envelope,
)

from . import __version__
from .renderer import Renderer

# Pull ``.env`` into ``os.environ`` before typer parses argv. Idempotent.
autoload_dotenv()

# -- Exit codes (cli-design rule group 3) ------------------------------

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_USAGE = 2
EXIT_AUTH = 4
EXIT_SIGINT = 6
EXIT_CONNECT = 7

PROG = "agentm-terminal"

log = logging.getLogger("agentm_terminal")


def _err(kind: str, root: str, fix: str) -> None:
    """Print a structured error line to stderr.

    Format: ``agentm-terminal: error: <type>: <root cause>. <fix>.``
    """
    sys.stderr.write(f"{PROG}: error: {kind}: {root}. {fix}.\n")
    sys.stderr.flush()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{PROG} {__version__}")
        raise typer.Exit(code=EXIT_OK)


# -- typer app ---------------------------------------------------------

app = typer.Typer(
    name=PROG,
    add_completion=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _resolve_format(arg: str | None) -> str:
    if arg is not None:
        return arg
    return "text" if sys.stdout.isatty() else "json"


def _resolve_color(no_color_flag: bool, fmt: str) -> bool:
    if fmt == "json":
        return False
    if no_color_flag:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _resolve_connect(url: str, tls_ca: str | None):
    """Wrap :func:`resolve_connect` with CLI-shaped error reporting."""
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
                "Gateway URL. Supported schemes: unix:///abs/path/to/sock, "
                "ws://host:port/path, wss://host:port/path. Default: shared "
                "with `agentm-gateway` ($XDG_RUNTIME_DIR/agentm-gw.sock if "
                "set, else /tmp/agentm-gw-<uid>.sock). Env: AGENTM_SOCKET."
            ),
        ),
    ] = DEFAULT_SOCKET_URL,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            envvar="AGENTM_TOKEN",
            help=(
                "Bearer token sent in the hello envelope. Required for "
                "ws/wss gateways with token auth. Env: AGENTM_TOKEN. "
                "Note: visible in `ps`; prefer the env variable."
            ),
        ),
    ] = None,
    tls_ca: Annotated[
        str | None,
        typer.Option(
            "--tls-ca",
            metavar="PATH",
            help=(
                "CA bundle (PEM) to verify the gateway certificate. Only "
                "meaningful with wss://. Default: system trust store."
            ),
        ),
    ] = None,
    output_format: Annotated[
        str | None,
        typer.Option(
            "--format",
            envvar="AGENTM_FORMAT",
            help=(
                "Output format / frontend. Choices: text, json, textual. "
                "Default: 'text' on a TTY, 'json' otherwise. Use 'json' for "
                "scripts / agent drivers; 'textual' for the rich TUI."
            ),
        ),
    ] = None,
    no_color: Annotated[
        bool,
        typer.Option(
            "--no-color",
            help=(
                "Disable ANSI color in text mode. Also honoured via the "
                "NO_COLOR environment variable (https://no-color.org)."
            ),
        ),
    ] = False,
    sender_id: Annotated[
        str,
        typer.Option("--sender-id", help="Inbound sender id (default: 'local')."),
    ] = "local",
    chat_id: Annotated[
        str,
        typer.Option("--chat-id", help="Chat / session id (default: 'terminal')."),
    ] = "terminal",
    no_input: Annotated[
        bool,
        typer.Option(
            "--no-input",
            help=(
                "Exit immediately after the gateway handshake. Useful for "
                "liveness probes; not the default — the client normally "
                "reads stdin."
            ),
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Raise log level on stderr to INFO (default: WARNING).",
        ),
    ] = False,
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            is_eager=True,
            callback=_version_callback,
            help="Print version and exit.",
        ),
    ] = False,
) -> None:
    """Terminal client for the AgentM channels gateway.

    Connects over the v1 wire protocol (Unix socket) and renders
    outbound messages on stdout.

    Examples:

      Interactive (TTY auto-selects text format):
        agentm-terminal --connect unix:///tmp/gw.sock

      Piped (auto-selects JSON format):
        printf '/help\\n' | agentm-terminal --connect unix:///tmp/gw.sock
    """
    if output_format is not None and output_format not in ("text", "json", "textual"):
        _err(
            "bad-argument",
            f"--format {output_format!r} is not one of text/json/textual",
            "choose one of text, json, textual",
        )
        raise typer.Exit(code=EXIT_USAGE)

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        rc = asyncio.run(
            _arun(
                connect=connect,
                token=token,
                tls_ca=tls_ca,
                output_format=output_format,
                no_color=no_color,
                sender_id=sender_id,
                chat_id=chat_id,
                no_input=no_input,
            )
        )
    except KeyboardInterrupt:
        rc = EXIT_SIGINT
    raise typer.Exit(code=rc)


# -- async run loop ----------------------------------------------------


async def _arun(
    *,
    connect: str,
    token: str | None,
    tls_ca: str | None,
    output_format: str | None,
    no_color: bool,
    sender_id: str,
    chat_id: str,
    no_input: bool,
) -> int:
    fmt = _resolve_format(output_format)
    color = _resolve_color(no_color, fmt)
    renderer = (
        Renderer(fmt=fmt, color=color) if fmt in ("text", "json") else None
    )
    textual_outbound: asyncio.Queue[dict[str, Any]] | None = (
        asyncio.Queue() if fmt == "textual" else None
    )

    _spec, transport = _resolve_connect(connect, tls_ca)

    # Peer id: stable-ish within a single run. The gateway uses this as
    # the synthetic channel name registered on the bus; pairing it with
    # a uuid prevents collisions between concurrent terminal sessions.
    peer_id = f"terminal-{uuid.uuid4().hex[:8]}"

    stop_event = asyncio.Event()
    exit_code = EXIT_OK

    async def on_outbound(env: Envelope) -> None:
        # WireClient already unpacks delivery_batch items and dispatches
        # each as its own ``KIND_OUTBOUND`` envelope, so we only need to
        # handle the single-item case here.
        if env.kind == KIND_OUTBOUND:
            body = env.body if isinstance(env.body, dict) else {}
            if textual_outbound is not None:
                await textual_outbound.put(body)
                return
            try:
                assert renderer is not None
                renderer.render_outbound(body)
            except Exception:  # noqa: BLE001
                log.exception("renderer failed on envelope id=%s", env.id)
            return
        if env.kind == KIND_DELIVERY_BATCH:  # pragma: no cover — handled upstream
            return
        if env.kind == KIND_PING:
            return  # WireClient replies pong itself
        if env.kind == KIND_PONG:
            return
        if env.kind == KIND_BYE:
            log.info("gateway sent BYE; shutting down")
            stop_event.set()
            return
        if env.kind == KIND_ERROR:
            nonlocal exit_code
            body = env.body if isinstance(env.body, dict) else {}
            code = body.get("code", "unknown")
            message = body.get("message", "")
            _err(
                "gateway-error",
                f"gateway emitted error code={code!r} message={message!r}",
                "check the gateway logs (stderr on its side)",
            )
            exit_code = EXIT_GENERIC
            stop_event.set()
            return

    client = WireClient(
        transport=transport,
        peer_id=peer_id,
        peer_kind="chat_client",
        token=token,
        on_outbound=on_outbound,
    )

    try:
        await client.connect()
    except AuthError as exc:
        _err(
            "auth-failed",
            f"gateway rejected handshake (code={exc.code!r})",
            "verify --bind-allow-uid on the gateway covers this client's uid",
        )
        return EXIT_AUTH
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        _err(
            "connect-failed",
            f"cannot connect to {connect!r} ({exc.__class__.__name__})",
            "is the gateway running with --bind on that path",
        )
        return EXIT_CONNECT
    except OSError as exc:
        _err(
            "connect-failed",
            f"cannot connect to {connect!r}: {exc.strerror or exc}",
            "check the socket path and gateway state",
        )
        return EXIT_CONNECT

    if renderer is not None:
        renderer.ready()

    if no_input:
        await client.close()
        return EXIT_OK

    if fmt == "textual":
        assert textual_outbound is not None
        from .ui.textual import run_textual

        async def _send_inbound(body: dict[str, Any]) -> None:
            await client.send_inbound(
                body, env_id=f"in-{int(time.time() * 1_000_000)}"
            )

        try:
            code = await run_textual(
                send_inbound=_send_inbound,
                outbound_queue=textual_outbound,
                sender_id=sender_id,
                chat_id=chat_id,
            )
        finally:
            await client.close()
        return int(code)

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
        except (NotImplementedError, RuntimeError):  # pragma: no cover — Windows
            pass

    reader_task = asyncio.create_task(
        _stdin_reader(client, sender_id, chat_id, stop_event),
        name="terminal-stdin",
    )
    stop_task = asyncio.create_task(stop_event.wait(), name="terminal-stop")

    try:
        done, pending = await asyncio.wait(
            [reader_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
    finally:
        if renderer is not None:
            renderer.stopped()
        await client.close()

    if sigint_seen:
        return EXIT_SIGINT
    return exit_code


async def _stdin_reader(
    client: WireClient,
    sender_id: str,
    chat_id: str,
    stop_event: asyncio.Event,
) -> None:
    """Read stdin line by line, wrap into ``inbound`` envelopes."""
    loop = asyncio.get_running_loop()
    while not stop_event.is_set():
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
        except Exception:  # pragma: no cover — defensive
            log.exception("stdin read failed")
            stop_event.set()
            return
        if not line:
            stop_event.set()
            return
        content = line.rstrip("\n")
        if not content:
            continue
        button_value: str | None = None
        if content.startswith("="):
            button_value = content[1:].strip() or None
            display = f"[button click: {button_value or ''}]"
        else:
            display = content
        body: dict[str, Any] = {
            "channel": "terminal",
            "sender_id": sender_id,
            "chat_id": chat_id,
            "content": display,
        }
        if button_value is not None:
            body["button_value"] = button_value
        try:
            await client.send_inbound(
                body, env_id=f"in-{int(time.time() * 1_000_000)}"
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to send inbound; closing")
            stop_event.set()
            return


# -- entrypoint --------------------------------------------------------


def main() -> None:
    """Entry point referenced by the ``agentm-terminal`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
