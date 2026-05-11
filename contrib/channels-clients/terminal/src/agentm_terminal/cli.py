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
  is a TTY, ``json`` otherwise. Document explicitly in --help so the
  contract is discoverable.

Example::

    # Interactive:
    agentm-terminal --connect unix:///tmp/gw.sock

    # Piped (auto json mode):
    printf '/help\\n' | agentm-terminal --connect unix:///tmp/gw.sock
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from typing import Any
from urllib.parse import urlparse

from agentm_channels import DEFAULT_SOCKET_URL
from agentm_channels.client import AuthError, WireClient
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


# -- argv --------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    epilog = (
        "Examples:\n"
        "  Interactive (TTY auto-selects text format):\n"
        f"    {PROG} --connect unix:///tmp/gw.sock\n"
        "\n"
        "  Piped (auto-selects JSON format):\n"
        f"    printf '/help\\n' | {PROG} --connect unix:///tmp/gw.sock\n"
    )
    p = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Terminal client for the AgentM channels gateway. Connects "
            "over the v1 wire protocol (Unix socket) and renders outbound "
            "messages on stdout."
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--connect",
        default=DEFAULT_SOCKET_URL,
        metavar="URL",
        help=(
            "Gateway socket URL. v1 supports only unix:///abs/path/to/sock; "
            "other schemes are rejected with exit 2. "
            f"Default: ``{DEFAULT_SOCKET_URL}`` "
            "($XDG_RUNTIME_DIR/agentm-gw.sock if set, else "
            "/tmp/agentm-gw-<uid>.sock — matches `agentm-gateway`'s default)."
        ),
    )
    p.add_argument(
        "--format",
        choices=("text", "json", "textual"),
        default=None,
        help=(
            "Output format / frontend. Default: 'text' when stdout is a TTY, "
            "else 'json'. Use 'json' for scripts / agent drivers. Use "
            "'textual' for the rich TUI (requires the `textual` package)."
        ),
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help=(
            "Disable ANSI color in text mode. Also honoured via the "
            "NO_COLOR environment variable (https://no-color.org)."
        ),
    )
    p.add_argument(
        "--sender-id",
        default="local",
        help="Inbound sender id (default: 'local').",
    )
    p.add_argument(
        "--chat-id",
        default="terminal",
        help="Chat / session id (default: 'terminal').",
    )
    p.add_argument(
        "--no-input",
        action="store_true",
        help=(
            "Exit immediately after the gateway handshake. Useful for "
            "liveness probes; not the default — the client normally reads "
            "stdin."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Raise log level on stderr to INFO (default: WARNING).",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"{PROG} {__version__}",
    )
    return p


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


def _parse_connect_url(url: str) -> str:
    """Return the absolute socket path. Raises ``SystemExit(2)`` on
    invalid input (scheme, missing path)."""
    parsed = urlparse(url)
    if parsed.scheme != "unix":
        _err(
            "bad-argument",
            f"--connect scheme {parsed.scheme!r} is not supported",
            "use unix:///abs/path/to/sock (only unix:// is available in v1)",
        )
        raise SystemExit(EXIT_USAGE)
    # urlparse on ``unix:///abs/path`` → netloc="", path="/abs/path".
    socket_path = parsed.path or parsed.netloc
    if not socket_path or not socket_path.startswith("/"):
        _err(
            "bad-argument",
            f"--connect URL {url!r} has no absolute socket path",
            "use unix:///abs/path/to/sock",
        )
        raise SystemExit(EXIT_USAGE)
    return socket_path


# -- async run loop ----------------------------------------------------


async def _arun(args: argparse.Namespace) -> int:
    fmt = _resolve_format(args.format)
    color = _resolve_color(args.no_color, fmt)
    # The textual frontend has its own render path; the text/json
    # renderer is only constructed for the line-based modes.
    renderer = (
        Renderer(fmt=fmt, color=color) if fmt in ("text", "json") else None
    )
    textual_outbound: asyncio.Queue[dict[str, Any]] | None = (
        asyncio.Queue() if fmt == "textual" else None
    )

    socket_path = _parse_connect_url(args.connect)

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
        # WireClient's _dispatch handles delivery_batch → outbound itself,
        # but defensive branches keep the contract explicit.
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
        socket_path=socket_path,
        peer_id=peer_id,
        peer_kind="chat_client",
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
            f"cannot connect to {args.connect!r} ({exc.__class__.__name__})",
            "is the gateway running with --bind on that path",
        )
        return EXIT_CONNECT
    except OSError as exc:
        # ENOENT / ECONNREFUSED / EACCES on the socket path.
        _err(
            "connect-failed",
            f"cannot connect to {args.connect!r}: {exc.strerror or exc}",
            "check the socket path and gateway state",
        )
        return EXIT_CONNECT

    # Handshake succeeded. Announce ready (text/json modes only — the
    # textual app shows readiness via the title bar).
    if renderer is not None:
        renderer.ready()

    if args.no_input:
        await client.close()
        return EXIT_OK

    # Textual UI takes over the terminal — it owns stdin/stdout and
    # the event loop until the user quits. The shared stop_event /
    # signal handlers / stdin reader path is skipped entirely.
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
                sender_id=args.sender_id,
                chat_id=args.chat_id,
            )
        finally:
            await client.close()
        return int(code)

    # Install a SIGINT handler that flips to the EXIT_SIGINT code so a
    # piped user (Ctrl-C while reading stdin) gets a stable exit code.
    sigint_seen = False

    def _on_sigint() -> None:
        nonlocal sigint_seen
        sigint_seen = True
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_sigint if sig == signal.SIGINT else stop_event.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover — Windows
            pass

    reader_task = asyncio.create_task(
        _stdin_reader(client, args.sender_id, args.chat_id, stop_event),
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
            # EOF (Ctrl-D / piped input exhausted). Clean shutdown.
            stop_event.set()
            return
        content = line.rstrip("\n")
        if not content:
            continue
        button_value: str | None = None
        if content.startswith("="):
            # Round-trip escape for approval-button clicks. Matches the
            # legacy TerminalChannel format byte-for-byte.
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    except SystemExit as exc:
        # argparse already printed to stderr. Map to canonical usage code.
        if isinstance(exc.code, int):
            return EXIT_USAGE if exc.code == 2 else int(exc.code)
        return EXIT_USAGE

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        return asyncio.run(_arun(args))
    except KeyboardInterrupt:
        return EXIT_SIGINT
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return EXIT_GENERIC


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
