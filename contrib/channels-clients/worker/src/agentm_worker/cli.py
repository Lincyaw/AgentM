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

import argparse
import asyncio
import logging
import os
import signal
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agentm.core.abi import EventBus
from pathlib import Path

from agentm_channels import DEFAULT_SOCKET_URL, load_dotenv_files
from agentm_channels.client import AuthError, WireClient
from agentm_channels.wire import (
    KIND_BYE,
    KIND_ERROR,
    KIND_INBOUND,
    KIND_OUTBOUND,
    Envelope,
)

from . import __version__
from .runner import WorkerRunner

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


# -- argv --------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    epilog = (
        "Examples:\n"
        f"  {PROG} --connect unix:///tmp/gw.sock --scenario general_purpose\n"
        "\n"
        "Note: every session this worker hosts shares --cwd. To serve\n"
        "multiple project roots, start more workers.\n"
    )
    p = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Agent-side worker for the AgentM channels gateway. Connects "
            "over the v1 wire protocol (Unix socket), advertises a "
            "scenario list, and drives one AgentSession per chat for "
            "forwarded inbound messages."
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
        "--scenario",
        action="append",
        metavar="NAME",
        default=None,
        help=(
            "Scenario this worker can handle. Repeatable. Advertised at "
            "hello as capabilities.scenarios. Default: ['general_purpose']."
        ),
    )
    p.add_argument(
        "--cwd",
        default=os.environ.get("AGENTM_WORKER_CWD") or str(Path.cwd()),
        metavar="PATH",
        help=(
            "Working directory passed to AgentSession.create for every "
            "session this worker hosts. Default: $PWD. All sessions on "
            "this worker share this cwd — start a separate worker for "
            "another project root."
        ),
    )
    p.add_argument(
        "--provider",
        default=os.environ.get("AGENTM_WORKER_PROVIDER")
        or os.environ.get("AGENTM_PROVIDER"),
        help=(
            "LLM provider id (anthropic / openai / …). Default: read "
            "AGENTM_WORKER_PROVIDER or AGENTM_PROVIDER, else SDK default."
        ),
    )
    p.add_argument(
        "--model",
        default=os.environ.get("AGENTM_WORKER_MODEL")
        or os.environ.get("AGENTM_MODEL"),
        help="LLM model id. Default: provider default.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Maximum in-flight session.prompt calls across all sessions "
            "on this worker. Default: 4."
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


def _parse_connect_url(url: str) -> str:
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


# -- async run loop ---------------------------------------------------


async def _arun(args: argparse.Namespace) -> int:
    socket_path = _parse_connect_url(args.connect)
    scenarios: list[str] = list(args.scenario or ["general_purpose"])
    if not scenarios:
        _err(
            "bad-argument",
            "--scenario list is empty",
            "pass at least one --scenario NAME",
        )
        return EXIT_USAGE

    peer_id = f"worker-{uuid.uuid4().hex[:8]}"

    stop_event = asyncio.Event()
    exit_code = EXIT_OK
    runner_ref: dict[str, WorkerRunner] = {}

    async def on_outbound(env: Envelope) -> None:
        # The gateway forwards inbound work as ``KIND_INBOUND``. With
        # Phase 6 A2A support, the gateway *also* routes
        # ``KIND_OUTBOUND`` envelopes back to this worker when a peer
        # delivers a reply to one of our ``peer_send`` calls — match by
        # ``correlation_id`` in the runner's pending-replies map.
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

    # Workers advertise scenarios + cwd in capabilities — the gateway's
    # WorkerRegistry uses this for routing decisions.
    client = WireClient(
        socket_path=socket_path,
        peer_id=peer_id,
        peer_kind="agent_worker",
        on_outbound=on_outbound,
        capabilities={"scenarios": scenarios, "cwd": args.cwd},
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

    factory = _build_session_factory(
        scenario=scenarios[0],
        provider=args.provider,
        model=args.model,
    )
    runner = WorkerRunner(
        client=client,
        cwd=args.cwd,
        scenario=scenarios[0],
        session_factory=factory,
        max_concurrency=int(args.max_concurrency),
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
        args.cwd,
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


def main(argv: list[str] | None = None) -> int:
    # Match agentm-gateway: pull .env vars (provider API keys, custom
    # scenario knobs) from cwd / workspace root before arg parsing.
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
