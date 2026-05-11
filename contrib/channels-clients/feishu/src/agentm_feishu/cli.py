"""``agentm-feishu`` — Feishu / Lark client process for the channels gateway.

Connects to an ``agentm-gateway --bind unix://…`` over the v1 wire
protocol, brings up the Feishu adapter, and proxies inbound messages /
card-button clicks into the gateway and outbound messages back to
Feishu as schema-2.0 interactive cards.

CLI design follows ``autoharness:cli-design``:

* All logs go to stderr. Stdout is unused (daemon).
* Exit codes are stable and standardized (see ``EXIT_*`` constants).
* Errors: ``agentm-feishu: error: <type>: <root cause>. <suggested fix>.``
* Secrets never appear in argv (rule group 5). The app secret comes
  from a file (``--app-secret PATH``) or the ``LARK_APP_SECRET`` env
  variable — never as a literal flag value.

Examples::

    # File-based secret (recommended):
    agentm-feishu \\
      --connect unix:///tmp/gw.sock \\
      --app-id cli_xxxx \\
      --app-secret /run/secrets/feishu_app_secret

    # Env-based secret:
    LARK_APP_ID=cli_xxxx LARK_APP_SECRET=... \\
      agentm-feishu --connect unix:///tmp/gw.sock
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import uuid
from urllib.parse import urlparse

from pathlib import Path

from agentm_channels import DEFAULT_SOCKET_URL, load_dotenv_files
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
from .adapter import FeishuAdapter, FeishuConfig
from .ws_patch import apply_ws_patch

# -- Exit codes (cli-design rule group 3) ------------------------------

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_USAGE = 2
EXIT_AUTH = 4
EXIT_SIGINT = 6
EXIT_CONNECT = 7

PROG = "agentm-feishu"

log = logging.getLogger("agentm_feishu")


def _err(kind: str, root: str, fix: str) -> None:
    """Print a structured error line to stderr."""
    sys.stderr.write(f"{PROG}: error: {kind}: {root}. {fix}.\n")
    sys.stderr.flush()


# -- argv --------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    epilog = (
        "Examples:\n"
        "  File-based secret (recommended):\n"
        f"    {PROG} --connect unix:///tmp/gw.sock \\\n"
        "      --app-id cli_xxxx \\\n"
        "      --app-secret /run/secrets/feishu_app_secret\n"
        "\n"
        "  Env-based credentials:\n"
        "    LARK_APP_ID=cli_xxxx LARK_APP_SECRET=... \\\n"
        f"      {PROG} --connect unix:///tmp/gw.sock\n"
    )
    p = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Feishu / Lark client for the AgentM channels gateway. "
            "Connects over the v1 wire protocol (Unix socket) and "
            "bridges Feishu events ↔ gateway envelopes."
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
            "(matches `agentm-gateway`'s default)."
        ),
    )
    p.add_argument(
        "--app-id",
        metavar="ID",
        default=None,
        help=(
            "Feishu app id. Falls back to the LARK_APP_ID env variable; "
            "missing → exit 2."
        ),
    )
    p.add_argument(
        "--app-secret",
        metavar="PATH",
        default=None,
        help=(
            "Path to a file containing the Feishu app secret. Required "
            "unless LARK_APP_SECRET is set in the environment. The file "
            "wins if both are present. Secrets are never accepted as "
            "literal CLI values."
        ),
    )
    p.add_argument(
        "--allow-from",
        action="append",
        metavar="PATTERN",
        default=None,
        help=(
            "Sender id allow-list for inbound (repeatable). Defaults to "
            "['*'] (allow everyone) — set explicit ids to harden."
        ),
    )
    p.add_argument(
        "--chat-id-prefix",
        default="feishu",
        metavar="STR",
        help="Channel name reported on inbound envelopes (default: feishu).",
    )
    p.add_argument(
        "--check-config",
        action="store_true",
        help=(
            "Validate flags / env and exit 0 without contacting the "
            "gateway or Feishu. Useful for systemd unit smoke tests."
        ),
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Raise log level on stderr to INFO (default: WARNING).",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"{PROG} {__version__}",
    )
    return p


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
    socket_path = parsed.path or parsed.netloc
    if not socket_path or not socket_path.startswith("/"):
        _err(
            "bad-argument",
            f"--connect URL {url!r} has no absolute socket path",
            "use unix:///abs/path/to/sock",
        )
        raise SystemExit(EXIT_USAGE)
    return socket_path


def _load_app_secret(path: str | None) -> str | None:
    """Load the Feishu app secret. File wins over env."""
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                secret = f.read().strip()
        except OSError as exc:
            _err(
                "bad-argument",
                f"cannot read --app-secret file {path!r}: {exc.strerror or exc}",
                "check the path and read permission",
            )
            raise SystemExit(EXIT_USAGE) from exc
        if not secret:
            _err(
                "bad-argument",
                f"--app-secret file {path!r} is empty",
                "write the secret into the file (no trailing newline required)",
            )
            raise SystemExit(EXIT_USAGE)
        return secret
    env_secret = os.environ.get("LARK_APP_SECRET", "").strip()
    return env_secret or None


def _resolve_config(args: argparse.Namespace) -> FeishuConfig:
    """Materialize the adapter config or exit 2."""
    app_id = args.app_id or os.environ.get("LARK_APP_ID", "").strip()
    if not app_id:
        _err(
            "bad-argument",
            "missing --app-id (and no LARK_APP_ID in env)",
            "pass --app-id cli_xxxx or set LARK_APP_ID",
        )
        raise SystemExit(EXIT_USAGE)
    app_secret = _load_app_secret(args.app_secret)
    if not app_secret:
        _err(
            "bad-argument",
            "missing app secret (no --app-secret file and LARK_APP_SECRET empty)",
            "pass --app-secret PATH or set LARK_APP_SECRET",
        )
        raise SystemExit(EXIT_USAGE)
    allow_from = list(args.allow_from) if args.allow_from else ["*"]
    return FeishuConfig(
        app_id=app_id,
        app_secret=app_secret,
        allow_from=allow_from,
        chat_id_prefix=args.chat_id_prefix,
        channel_name=args.chat_id_prefix,
    )


# -- async run loop ----------------------------------------------------


async def _arun(args: argparse.Namespace) -> int:
    socket_path = _parse_connect_url(args.connect)
    cfg = _resolve_config(args)

    if args.check_config:
        return EXIT_OK

    peer_id = f"feishu-{uuid.uuid4().hex[:8]}"
    stop_event = asyncio.Event()
    exit_code = EXIT_OK
    adapter: FeishuAdapter | None = None

    async def on_outbound(env: Envelope) -> None:
        nonlocal exit_code
        if env.kind == KIND_OUTBOUND:
            if adapter is None:
                return
            try:
                await adapter.handle_outbound(env)
            except Exception:  # noqa: BLE001
                log.exception("adapter.handle_outbound failed id=%s", env.id)
            return
        if env.kind == KIND_DELIVERY_BATCH:  # pragma: no cover — unwrapped upstream
            return
        if env.kind in (KIND_PING, KIND_PONG):
            return
        if env.kind == KIND_BYE:
            log.info("gateway sent BYE; shutting down")
            stop_event.set()
            return
        if env.kind == KIND_ERROR:
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
        _err(
            "connect-failed",
            f"cannot connect to {args.connect!r}: {exc.strerror or exc}",
            "check the socket path and gateway state",
        )
        return EXIT_CONNECT

    # Apply WS patch *before* constructing the adapter — the patch
    # rebinds ``lark_oapi.ws.client.loop`` and the adapter's
    # ``FeishuChannel`` constructor latches onto that module global.
    apply_ws_patch()
    adapter = FeishuAdapter(client=client, config=cfg)

    # Install signal handlers for clean shutdown.
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

    adapter_task = asyncio.create_task(adapter.start(), name="feishu-adapter")
    stop_task = asyncio.create_task(stop_event.wait(), name="feishu-stop")

    try:
        done, pending = await asyncio.wait(
            [adapter_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        # If the adapter task crashed (e.g. invalid Feishu credentials)
        # surface the exception class as an auth failure.
        if adapter_task in done and adapter_task.exception() is not None:
            adapter_exc = adapter_task.exception()
            assert adapter_exc is not None  # for mypy
            _err(
                "feishu-error",
                f"adapter failed: {adapter_exc.__class__.__name__}: {adapter_exc}",
                "check --app-id / --app-secret and Feishu app status",
            )
            # Distinguish credential / handshake failures from generic
            # crashes by class name — lark_oapi raises connect errors
            # without a dedicated type, so this is the best we can do.
            return EXIT_AUTH
        for t in pending:
            t.cancel()
    finally:
        try:
            await adapter.stop()
        except Exception:  # noqa: BLE001
            log.exception("adapter.stop raised")
        await client.close()

    if sigint_seen:
        return EXIT_SIGINT
    return exit_code


# -- entrypoint --------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    # Pick up LARK_APP_ID / LARK_APP_SECRET etc. from a .env in the
    # caller's cwd (or the workspace root) before arg parsing — matches
    # what agentm-gateway does.
    load_dotenv_files(Path.cwd())
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return EXIT_USAGE if exc.code == 2 else int(exc.code)
        return EXIT_USAGE

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # lark-oapi logs the recv loop's clean WebSocket close as ERROR
    # ("receive message loop exit, err: sent 1000 (OK); ...") which
    # looks like a crash but isn't — 1000 is "Normal Closure". Downgrade
    # those specific lines to INFO so operators don't get spooked.
    class _LarkCleanCloseFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.levelno == logging.ERROR and "1000 (OK)" in record.getMessage():
                record.levelno = logging.INFO
                record.levelname = "INFO"
            return True

    logging.getLogger("Lark").addFilter(_LarkCleanCloseFilter())

    # asyncio yells "Task was destroyed but it is pending!" for the
    # background tasks lark spawns on its private (daemon-thread) loop.
    # See ``adapter._cancel_lark_tasks`` for the full diagnosis — those
    # tasks have no actual remediation, they're already cancelled and
    # the daemon thread dies with the process. Suppress the line so
    # operators don't see a wall of red on Ctrl-C.
    class _LarkOrphanTaskFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            if "lark_oapi" in message and "Task was destroyed" in message:
                return False
            return True

    logging.getLogger("asyncio").addFilter(_LarkOrphanTaskFilter())

    # And one more orphan: lark's ``ExpiringCache._start_clear_cron``
    # coroutine never gets awaited because the task wrapping it dies
    # with the daemon thread. Python prints this through the warnings
    # module (not logging), so the asyncio filter above doesn't catch
    # it — file a dedicated filterwarnings entry.
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"coroutine 'ExpiringCache\._start_clear_cron' was never awaited",
        category=RuntimeWarning,
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
