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

import asyncio
import contextlib
import logging
import os
import signal
import sys
import uuid
import warnings
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
from .adapter import FeishuAdapter, FeishuConfig
from .ws_patch import apply_ws_patch

# Pull ``.env`` into ``os.environ`` before typer parses argv. Idempotent.
autoload_dotenv()

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
            raise typer.Exit(code=EXIT_USAGE) from exc
        if not secret:
            _err(
                "bad-argument",
                f"--app-secret file {path!r} is empty",
                "write the secret into the file (no trailing newline required)",
            )
            raise typer.Exit(code=EXIT_USAGE)
        return secret
    env_secret = os.environ.get("LARK_APP_SECRET", "").strip()
    return env_secret or None


def _resolve_config(
    *,
    app_id: str | None,
    app_secret_path: str | None,
    allow_from: list[str] | None,
    channel_name: str,
    scenario: str | None,
    session_scope: str,
) -> FeishuConfig:
    """Materialize the adapter config or exit 2."""
    resolved_app_id = app_id or os.environ.get("LARK_APP_ID", "").strip()
    if not resolved_app_id:
        _err(
            "bad-argument",
            "missing --app-id (and no LARK_APP_ID in env)",
            "pass --app-id cli_xxxx or set LARK_APP_ID",
        )
        raise typer.Exit(code=EXIT_USAGE)
    resolved_app_secret = _load_app_secret(app_secret_path)
    if not resolved_app_secret:
        _err(
            "bad-argument",
            "missing app secret (no --app-secret file and LARK_APP_SECRET empty)",
            "pass --app-secret PATH or set LARK_APP_SECRET",
        )
        raise typer.Exit(code=EXIT_USAGE)
    if session_scope not in ("chat", "user"):
        _err(
            "bad-argument",
            f"--session-scope {session_scope!r} is not 'chat' or 'user'",
            "choose 'chat' (shared session per chat) or 'user' (per sender)",
        )
        raise typer.Exit(code=EXIT_USAGE)
    return FeishuConfig(
        app_id=resolved_app_id,
        app_secret=resolved_app_secret,
        allow_from=list(allow_from) if allow_from else ["*"],
        channel_name=channel_name,
        scenario=scenario,
        session_scope=session_scope,
    )


# -- lark logging hygiene ----------------------------------------------


def _install_lark_log_filters() -> None:
    """Tame two known shutdown-noise sources from lark_oapi.

    Idempotent: the filter classes are unique-per-call but installing
    them twice on the same logger is harmless — they're cheap predicates.
    """

    class _LarkCleanCloseFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.levelno == logging.ERROR and "1000 (OK)" in record.getMessage():
                record.levelno = logging.INFO
                record.levelname = "INFO"
            return True

    logging.getLogger("Lark").addFilter(_LarkCleanCloseFilter())

    class _LarkOrphanTaskFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            if "lark_oapi" in message and "Task was destroyed" in message:
                return False
            return True

    logging.getLogger("asyncio").addFilter(_LarkOrphanTaskFilter())

    warnings.filterwarnings(
        "ignore",
        message=r"coroutine 'ExpiringCache\._start_clear_cron' was never awaited",
        category=RuntimeWarning,
    )


# -- typer app ---------------------------------------------------------

app = typer.Typer(
    name=PROG,
    add_completion=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)


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
                "with `agentm-gateway`. Env: AGENTM_SOCKET."
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
                "NOTE: CLI args leak into /proc and shell history — "
                "prefer --token-file or AGENTM_TOKEN."
            ),
        ),
    ] = None,
    token_file: Annotated[
        str | None,
        typer.Option(
            "--token-file",
            metavar="PATH",
            help=(
                "Read the bearer token from PATH (whitespace stripped). "
                "Mutually exclusive with --token. Preferred for production."
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
                "meaningful with wss://."
            ),
        ),
    ] = None,
    app_id: Annotated[
        str | None,
        typer.Option(
            "--app-id",
            envvar="LARK_APP_ID",
            metavar="ID",
            help=(
                "Feishu app id. Falls back to the LARK_APP_ID env variable; "
                "missing → exit 2."
            ),
        ),
    ] = None,
    app_secret: Annotated[
        str | None,
        typer.Option(
            "--app-secret",
            metavar="PATH",
            help=(
                "Path to a file containing the Feishu app secret. Required "
                "unless LARK_APP_SECRET is set in the environment. The file "
                "wins if both are present. Secrets are never accepted as "
                "literal CLI values."
            ),
        ),
    ] = None,
    allow_from: Annotated[
        list[str] | None,
        typer.Option(
            "--allow-from",
            metavar="PATTERN",
            help=(
                "Sender id allow-list for inbound (repeatable). Defaults "
                "to ['*'] (allow everyone) — set explicit ids to harden."
            ),
        ),
    ] = None,
    channel_name: Annotated[
        str,
        typer.Option(
            "--channel-name",
            metavar="STR",
            help="Channel name reported on inbound envelopes (default: feishu).",
        ),
    ] = "feishu",
    scenario: Annotated[
        str | None,
        typer.Option(
            "--scenario",
            envvar="AGENTM_SCENARIO",
            help=(
                "Scenario the gateway builds new sessions in (sent on the "
                "first message per chat). Defaults to 'chatbot' — the "
                "conversational persona+memory composition built for chat "
                "channels. Override with this flag or AGENTM_SCENARIO."
            ),
        ),
    ] = "chatbot",
    session_scope: Annotated[
        str,
        typer.Option(
            "--session-scope",
            help=(
                "How session_key is composed (§3.4): 'chat' = one session "
                "per chat (default); 'user' = per sender within a chat."
            ),
        ),
    ] = "chat",
    check_config: Annotated[
        bool,
        typer.Option(
            "--check-config",
            help=(
                "Validate flags / env and exit 0 without contacting the "
                "gateway or Feishu. Useful for systemd unit smoke tests."
            ),
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
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
    """Feishu / Lark client for the AgentM channels gateway.

    Connects over the v1 wire protocol (Unix socket) and bridges
    Feishu events ↔ gateway envelopes.

    Examples:

      File-based secret (recommended):
        agentm-feishu --connect unix:///tmp/gw.sock \\
          --app-id cli_xxxx \\
          --app-secret /run/secrets/feishu_app_secret

      Env-based credentials:
        LARK_APP_ID=cli_xxxx LARK_APP_SECRET=... \\
          agentm-feishu --connect unix:///tmp/gw.sock
    """
    from agentm.gateway import resolve_token

    try:
        effective_token = resolve_token(token, token_file)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _install_lark_log_filters()

    try:
        rc = asyncio.run(
            _arun(
                connect_opts=ConnectOptions(
                    connect=connect, token=effective_token, tls_ca=tls_ca
                ),
                app_id=app_id,
                app_secret_path=app_secret,
                allow_from=allow_from,
                channel_name=channel_name,
                scenario=scenario,
                session_scope=session_scope,
                check_config=check_config,
            )
        )
    except KeyboardInterrupt:
        rc = EXIT_SIGINT
    raise typer.Exit(code=rc)


# -- async run loop ----------------------------------------------------


async def _arun(
    *,
    connect_opts: ConnectOptions,
    app_id: str | None,
    app_secret_path: str | None,
    allow_from: list[str] | None,
    channel_name: str,
    scenario: str | None,
    session_scope: str,
    check_config: bool,
) -> int:
    connect = connect_opts.connect
    token = connect_opts.token
    _spec, transport = _resolve_connect(connect, connect_opts.tls_ca)
    cfg = _resolve_config(
        app_id=app_id,
        app_secret_path=app_secret_path,
        allow_from=allow_from,
        channel_name=channel_name,
        scenario=scenario,
        session_scope=session_scope,
    )

    if check_config:
        return EXIT_OK

    # Generated ONCE per process and reused on every reconnect (the wire
    # client never regenerates it), so a re-dial replays this peer's outbox.
    # A stable/configurable peer_name for cross-PROCESS restart continuity is
    # a separate future improvement.
    peer_name = f"feishu-{uuid.uuid4().hex[:8]}"
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
        if env.kind in (KIND_PING, KIND_PONG):
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
        transport=transport,
        peer_name=peer_name,
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

    apply_ws_patch()
    adapter = FeishuAdapter(client=client, config=cfg)

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
    # Reconnect supervisor: the Feishu adapter (long-poll) and this wire leg
    # are independent tasks. We already connected above, so the supervisor
    # runs reconnect-only (connect_first=False) — only the *gateway* link
    # re-dials on a drop (same peer_name → outbox replay); the adapter stays
    # alive across gateway restarts. The supervisor returns only on close()
    # or raises AuthError if a reconnect handshake is fatally rejected.
    reconnect_task = asyncio.create_task(
        client.run_reconnecting(connect_first=False), name="feishu-reconnect"
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
                "feishu-error",
                f"adapter failed: {adapter_exc.__class__.__name__}: {adapter_exc}",
                "check --app-id / --app-secret and Feishu app status",
            )
            return EXIT_AUTH
        if reconnect_task in done and reconnect_task.exception() is not None:
            reconnect_exc = reconnect_task.exception()
            assert reconnect_exc is not None
            if isinstance(reconnect_exc, AuthError):
                _err(
                    "auth-failed",
                    f"gateway rejected reconnect (code={reconnect_exc.code!r})",
                    "verify --bind-allow-uid on the gateway covers this uid",
                )
                return EXIT_AUTH
            _err(
                "gateway-error",
                f"reconnect supervisor failed: "
                f"{reconnect_exc.__class__.__name__}: {reconnect_exc}",
                "check the gateway state and logs",
            )
            return EXIT_GENERIC
        for t in pending:
            t.cancel()
    finally:
        try:
            await adapter.stop()
        except Exception:  # noqa: BLE001
            log.exception("adapter.stop raised")
        await client.close()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await reconnect_task

    if sigint_seen:
        return EXIT_SIGINT
    return exit_code


# -- entrypoint --------------------------------------------------------


def main() -> None:
    """Entry point referenced by the ``agentm-feishu`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
