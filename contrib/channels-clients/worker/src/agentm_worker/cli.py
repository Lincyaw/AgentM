"""``agentm-worker`` — agent-side worker process for the channels gateway.

Connects to an ``agentm-gateway --bind unix://…`` as a wire peer of
kind ``agent_worker``, advertises a list of scenarios it can handle,
and drives one :class:`agentm.harness.AgentSession` per chat for the
gateway's forwarded inbound messages.

CLI design follows ``autoharness:cli-design``:

* All logs go to stderr; stdout is reserved for business output (the
  worker has none — it speaks only over the wire).
* Standardized exit codes (see ``EXIT_*`` constants).
* Errors are reported as
  ``agentm-worker: error: <type>: <root cause>. <suggested fix>.``
* Non-interactive; SIGINT / SIGTERM trigger a clean shutdown.

Example::

    agentm-worker --connect unix:///tmp/gw.sock --scenario general_purpose
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from agentm.core.abi import EventBus
from agentm_channels import DEFAULT_SOCKET_URL, autoload_dotenv
from agentm_channels.client import AuthError, WireClient
from agentm_channels.client_cli import ConnectError, ConnectOptions, resolve_connect
from agentm_channels.wire import (
    KIND_BYE,
    KIND_ERROR,
    KIND_INBOUND,
    KIND_OUTBOUND,
    Envelope,
)

from . import __version__
from .runner import WorkerRunner

# Pull ``.env`` into ``os.environ`` before typer parses argv. Idempotent.
autoload_dotenv()

# -- Exit codes (cli-design rule group 3) ------------------------------

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_USAGE = 2
EXIT_AUTH = 4
EXIT_SIGINT = 6
EXIT_CONNECT = 7

PROG = "agentm-worker"

log = logging.getLogger("agentm_worker")


def _err(kind: str, root: str, fix: str) -> None:
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


# -- session factory --------------------------------------------------


def _build_session_factory(
    *,
    scenario: str | None,
    provider: str | None,
    model: str | None,
) -> Callable[[str, EventBus, str | None], Awaitable[Any]]:
    """Build the AgentSession factory the worker uses for every chat.

    Mirrors :func:`agentm_channels.cli._build_session_factory` — kept
    duplicated because the gateway CLI's helper is module-private. The
    worker keeps it independent so behaviour changes on the gateway
    side don't silently change worker behaviour.
    """
    from typing import cast as _cast

    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession
    from agentm.core.runtime.session_bootstrap import (
        make_default_session_store,
        resolve_session_state,
    )

    chosen_provider = provider or DEFAULT_PROVIDER_REGISTRY.default_provider().id
    chosen_model = model or DEFAULT_PROVIDER_REGISTRY.default_model(chosen_provider)

    async def factory(cwd: str, bus: EventBus, resume: str | None) -> Any:
        store = make_default_session_store(cwd)
        state = resolve_session_state(
            cwd=cwd, resume=resume, continue_recent=False, session_store=store
        )
        provider_spec = DEFAULT_PROVIDER_REGISTRY.build(
            chosen_provider, {"model": chosen_model}
        )
        config = AgentSessionConfig(
            cwd=cwd,
            provider=provider_spec,
            scenario=scenario,
            session_manager=_cast(Any, state),
            bus=bus,
        )
        return await AgentSession.create(config)

    return factory


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
    scenario: Annotated[
        list[str] | None,
        typer.Option(
            "--scenario",
            envvar="AGENTM_SCENARIO",
            metavar="NAME",
            help=(
                "Scenario this worker can handle. Repeatable. Advertised "
                "at hello as capabilities.scenarios. "
                "Default: ['general_purpose']. Env: AGENTM_SCENARIO "
                "(single scenario only via env)."
            ),
        ),
    ] = None,
    cwd: Annotated[
        str,
        typer.Option(
            "--cwd",
            envvar="AGENTM_CWD",
            metavar="PATH",
            help=(
                "Working directory passed to AgentSession.create for every "
                "session this worker hosts. Default: $PWD. All sessions on "
                "this worker share this cwd — start a separate worker for "
                "another project root. Env: AGENTM_CWD."
            ),
        ),
    ] = "",
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            envvar="AGENTM_PROVIDER",
            help=(
                "LLM provider id (anthropic / openai / …). "
                "Default: SDK default. Env: AGENTM_PROVIDER."
            ),
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            envvar="AGENTM_MODEL",
            help="LLM model id. Default: provider default. Env: AGENTM_MODEL.",
        ),
    ] = None,
    max_concurrency: Annotated[
        int,
        typer.Option(
            "--max-concurrency",
            metavar="N",
            help=(
                "Maximum in-flight session.prompt calls across all sessions "
                "on this worker. Default: 4."
            ),
        ),
    ] = 4,
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
    """Agent-side worker for the AgentM channels gateway.

    Connects over the v1 wire protocol (Unix socket), advertises a
    scenario list, and drives one AgentSession per chat for forwarded
    inbound messages.

    Examples:

      agentm-worker --connect unix:///tmp/gw.sock --scenario general_purpose

    Note: every session this worker hosts shares --cwd. To serve
    multiple project roots, start more workers.
    """
    from agentm_channels import resolve_token

    try:
        effective_token = resolve_token(token, token_file)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    resolved_cwd = cwd or str(Path.cwd())
    scenarios: list[str] = list(scenario) if scenario else ["general_purpose"]
    if not scenarios:
        _err(
            "bad-argument",
            "--scenario list is empty",
            "pass at least one --scenario NAME",
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
                connect_opts=ConnectOptions(
                    connect=connect, token=effective_token, tls_ca=tls_ca
                ),
                scenarios=scenarios,
                cwd=resolved_cwd,
                provider=provider,
                model=model,
                max_concurrency=max_concurrency,
            )
        )
    except KeyboardInterrupt:
        rc = EXIT_SIGINT
    raise typer.Exit(code=rc)


# -- async run loop ---------------------------------------------------


async def _arun(
    *,
    connect_opts: ConnectOptions,
    scenarios: list[str],
    cwd: str,
    provider: str | None,
    model: str | None,
    max_concurrency: int,
) -> int:
    connect = connect_opts.connect
    token = connect_opts.token
    _spec, transport = _resolve_connect(connect, connect_opts.tls_ca)

    peer_id = f"worker-{uuid.uuid4().hex[:8]}"

    stop_event = asyncio.Event()
    exit_code = EXIT_OK
    runner_ref: dict[str, WorkerRunner] = {}

    async def on_outbound(env: Envelope) -> None:
        if env.kind == KIND_INBOUND:
            runner = runner_ref.get("r")
            if runner is None:
                log.warning("inbound id=%s arrived before runner ready", env.id)
                return
            await runner.handle_inbound_envelope(env)
            return
        if env.kind == KIND_OUTBOUND:
            runner = runner_ref.get("r")
            if runner is None:
                log.warning("outbound id=%s arrived before runner ready", env.id)
                return
            await runner.handle_outbound_envelope(env)
            return
        if env.kind == KIND_BYE:
            log.info("gateway sent BYE; shutting down")
            stop_event.set()
            return
        if env.kind == KIND_ERROR:
            nonlocal exit_code
            body = env.body if isinstance(env.body, dict) else {}
            _err(
                "gateway-error",
                f"gateway emitted error code={body.get('code')!r} "
                f"message={body.get('message')!r}",
                "check the gateway logs (stderr on its side)",
            )
            exit_code = EXIT_GENERIC
            stop_event.set()

    client = WireClient(
        transport=transport,
        peer_id=peer_id,
        peer_kind="agent_worker",
        token=token,
        on_outbound=on_outbound,
        capabilities={"scenarios": scenarios, "cwd": cwd},
    )

    try:
        await client.connect()
    except AuthError as exc:
        _err(
            "auth-failed",
            f"gateway rejected handshake (code={exc.code!r})",
            "verify --bind-allow-uid on the gateway covers this worker's uid",
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

    factory = _build_session_factory(
        scenario=scenarios[0],
        provider=provider,
        model=model,
    )
    runner = WorkerRunner(
        client=client,
        cwd=cwd,
        scenario=scenarios[0],
        session_factory=factory,
        max_concurrency=int(max_concurrency),
    )
    runner_ref["r"] = runner
    await runner.start()

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

    log.info(
        "worker ready peer_id=%s scenarios=%s cwd=%s",
        peer_id,
        scenarios,
        cwd,
    )

    try:
        await stop_event.wait()
    finally:
        await runner.stop()
        await client.close()

    if sigint_seen:
        return EXIT_SIGINT
    return exit_code


# -- entrypoint --------------------------------------------------------


def main() -> None:
    """Entry point referenced by the ``agentm-worker`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
