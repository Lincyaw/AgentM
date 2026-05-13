"""``agentm-gateway`` console entry.

Reads an optional YAML config, builds a :class:`MessageBus`, spins up
a :class:`ChannelManager` (wire-bridge mode only) and a :class:`Gateway`,
and runs until the process is signalled to stop (SIGINT / SIGTERM).

The gateway is a wire-protocol daemon. Channel platforms (Feishu,
terminal, …) run as separate client processes and connect to the gateway
over a Unix socket — there is no in-process channel path.

    # Terminal A:
    agentm-gateway --bind unix:///tmp/gw.sock --config gw.yaml

    # Terminal B:
    agentm-terminal --connect unix:///tmp/gw.sock

Minimal config (``gateway.yaml``)::

    cwd: ./workspace
    scenario: general_purpose
    provider: openai
    model: <provider-default>

    approval:
      require_approval: [bash]
      timeout_seconds: 300

``.env`` is auto-loaded from the cwd and the AgentM workspace root.

Exit codes (see also ``cli-design`` rule group 3):

* ``0`` — clean shutdown
* ``2`` — argument / config error (bad YAML, conflicting flags, …)
* ``130`` — SIGINT (Ctrl-C)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

import typer

from agentm.ai import DEFAULT_PROVIDER_REGISTRY
from agentm.core.abi import EventBus

from . import DEFAULT_SOCKET_URL, autoload_dotenv
from .approval import ApprovalPolicy
from .auth import TokenAuthenticator, UnixPeerCredAuthenticator
from .bus import MessageBus
from .gateway import Gateway, GatewayConfig
from .manager import ChannelManager
from .outbox import SqliteInbox, SqliteOutbox
from .server import Authenticator, WireServer
from .session_bindings import SessionBindingStore
from .transport import (
    ServerTransport,
    UnixServerTransport,
    WebSocketServerTransport,
)
from .wire_bridge import DEFAULT_MAX_A2A_HOPS, WireBridge

# Pull ``.env`` keys into ``os.environ`` BEFORE typer parses argv. The
# ``envvar=...`` defaults below are resolved at parse time (not at
# module-import time), so this only needs to run before ``app()``.
# Idempotent across reimports.
autoload_dotenv()


PROG = "agentm-gateway"

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_SIGINT = 130


logger = logging.getLogger(__name__)


def _default_provider() -> str:
    return DEFAULT_PROVIDER_REGISTRY.default_provider().id


def _default_model(provider: str) -> str:
    return DEFAULT_PROVIDER_REGISTRY.default_model(provider)


# -------- config loading ------------------------------------------------


def _expand_env(obj: Any) -> Any:
    """Recursively expand ``${VAR}`` references in YAML values."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config {path} must be a YAML mapping")
    return _expand_env(data)


def _retry_delays(spec: Any) -> tuple[float, ...] | None:
    if spec is None:
        return None
    if not isinstance(spec, list):
        raise SystemExit("send_retry_delays must be a list of seconds (floats)")
    try:
        return tuple(float(x) for x in spec)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"send_retry_delays: invalid entry: {exc}") from exc


def _commands_section(
    spec: Any,
) -> tuple[bool, list[str], list[str]]:
    """Parse the ``commands:`` block in gateway YAML.

    Schema (every key optional)::

        commands:
          skill_paths: [/extra/skills]      # extra dirs for /skill:* discovery
          atoms:
            enabled: false                  # surface /atom:install etc.
            allow: [permission, …]          # whitelist; "*" = every mountable

    Returns ``(atom_enabled, atom_allow, skill_paths)``.
    """
    if not isinstance(spec, dict):
        return False, [], []
    skill_paths = [str(p) for p in spec.get("skill_paths") or []]
    atoms = spec.get("atoms")
    if not isinstance(atoms, dict):
        return False, [], skill_paths
    enabled = bool(atoms.get("enabled", False))
    allow_raw = atoms.get("allow") or []
    if not isinstance(allow_raw, list):
        raise SystemExit(
            "commands.atoms.allow must be a list of atom names (or [\"*\"])."
        )
    return enabled, [str(x) for x in allow_raw], skill_paths


def _approval_policy(spec: Any) -> ApprovalPolicy:
    if not isinstance(spec, dict):
        return ApprovalPolicy()
    return ApprovalPolicy(
        always_allow=frozenset(str(x) for x in spec.get("always_allow", [])),
        always_block=frozenset(str(x) for x in spec.get("always_block", [])),
        require_approval=frozenset(str(x) for x in spec.get("require_approval", [])),
        timeout_seconds=float(spec.get("timeout_seconds", 300.0)),
    )


# -------- session factory ----------------------------------------------


def _build_session_factory(
    *, scenario: str | None, provider: str, model: str
) -> Callable[[str, EventBus, str | None], Awaitable[Any]]:
    from typing import cast as _cast

    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession
    from agentm.core.runtime.session_bootstrap import (
        make_default_session_store,
        resolve_session_state,
    )

    async def factory(cwd: str, bus: EventBus, resume: str | None) -> Any:
        store = make_default_session_store(cwd)
        state = resolve_session_state(
            cwd=cwd, resume=resume, continue_recent=False, session_store=store
        )
        provider_spec = DEFAULT_PROVIDER_REGISTRY.build(provider, {"model": model})
        config = AgentSessionConfig(
            cwd=cwd,
            provider=provider_spec,
            scenario=scenario,
            session_manager=_cast(Any, state),
            bus=bus,
        )
        return await AgentSession.create(config)

    return factory


# -------- --bind resolution -------------------------------------------


@dataclass(frozen=True)
class BindOptions:
    """Raw bind-related typer arguments, before resolution.

    ``BindSpec`` is the *resolved* shape (after YAML/CLI/defaults merge);
    ``BindOptions`` is the *raw* shape the typer callback collects so we
    don't push 8 kwargs through ``_arun``'s signature.
    """

    bind: str | None = None
    bind_allow_uid: list[int] | None = None
    bind_allow_any_uid: bool = False
    bind_token_file: str | None = None
    bind_allow_anonymous: bool = False
    tls_cert: str | None = None
    tls_key: str | None = None


@dataclass(frozen=True)
class BindSpec:
    """Resolved ``--bind`` configuration.

    For unix scheme: ``socket_path`` is set, ``allow_uids`` of ``None``
    means any-uid (peer-cred reads the kernel uid but doesn't filter).

    For ws/wss scheme: ``host`` / ``port`` / ``url`` are set, ``tokens``
    holds the bearer-token allow-list (empty set when anonymous mode
    is explicitly opted into via ``--bind-allow-anonymous``). The
    request path is not gated server-side — a reverse proxy owns URL
    routing — but the original URL is kept on ``url`` for log/check
    output. TLS material on the server side is cert + key only; there
    is no client-cert verification today (a server ``--tls-ca`` knob
    would be a premature feature).
    """

    scheme: str
    # unix
    socket_path: str = ""
    allow_uids: frozenset[int] | None = frozenset()
    # ws / wss
    host: str = ""
    port: int = 0
    url: str = ""
    tokens: frozenset[str] = frozenset()
    allow_anonymous: bool = False
    tls_cert: str | None = None
    tls_key: str | None = None


def _load_tokens_file(path: str) -> set[str]:
    """One token per line; blank lines and lines starting with ``#``
    are skipped. Raises ``SystemExit`` on read errors.
    """
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
    bind_token_file: str | None = None,
    bind_allow_anonymous: bool = False,
    tls_cert: str | None = None,
    tls_key: str | None = None,
    cfg: dict[str, Any],
) -> BindSpec:
    """Merge CLI flags > yaml ``bind:`` section > shared default.

    Supports three schemes: ``unix://``, ``ws://``, ``wss://``. Raises
    :class:`SystemExit` (exit code 2) on invalid input — unknown scheme,
    conflicting flags, TLS-with-unix, wss-without-cert, or ws/wss
    without token configuration.
    """
    yaml_bind = cfg.get("bind") if isinstance(cfg.get("bind"), dict) else None
    yaml_url = (yaml_bind or {}).get("socket")
    url = bind or yaml_url or DEFAULT_SOCKET_URL
    parsed = urlparse(str(url))
    scheme = parsed.scheme

    yaml_token_auth = (yaml_bind or {}).get("token_auth") if yaml_bind else None
    yaml_token_auth = yaml_token_auth if isinstance(yaml_token_auth, dict) else None
    yaml_tls = (yaml_bind or {}).get("tls") if yaml_bind else None
    yaml_tls = yaml_tls if isinstance(yaml_tls, dict) else None

    eff_tls_cert = tls_cert or (yaml_tls.get("cert") if yaml_tls else None)
    eff_tls_key = tls_key or (yaml_tls.get("key") if yaml_tls else None)

    if scheme == "unix":
        if eff_tls_cert or eff_tls_key:
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

        if cli_uids or cli_any:
            allow_uids: frozenset[int] | None = (
                None if cli_any else frozenset(int(x) for x in cli_uids)
            )
        elif yaml_bind is not None and (
            "allow_uids" in yaml_bind or "allow_any_uid" in yaml_bind
        ):
            if yaml_bind.get("allow_any_uid") and yaml_bind.get("allow_uids"):
                raise SystemExit(
                    "bind.allow_uids and bind.allow_any_uid are mutually exclusive."
                )
            if yaml_bind.get("allow_any_uid"):
                allow_uids = None
            else:
                raw = yaml_bind.get("allow_uids") or []
                if not isinstance(raw, list):
                    raise SystemExit("bind.allow_uids must be a list of integers.")
                try:
                    allow_uids = frozenset(int(x) for x in raw)
                except (TypeError, ValueError) as exc:
                    raise SystemExit(
                        f"bind.allow_uids: invalid entry: {exc}"
                    ) from exc
        else:
            # Most-secure default: current process uid only.
            allow_uids = frozenset({os.geteuid()})

        return BindSpec(
            scheme="unix",
            socket_path=socket_path,
            allow_uids=allow_uids,
        )

    if scheme not in ("ws", "wss"):
        raise SystemExit(
            f"--bind scheme {scheme!r} not supported; use unix://, ws://, or wss://."
        )

    if scheme == "wss" and (not eff_tls_cert or not eff_tls_key):
        raise SystemExit(
            "wss:// bind requires --tls-cert and --tls-key (or bind.tls.cert/"
            "bind.tls.key in YAML)."
        )
    if scheme == "ws" and (eff_tls_cert or eff_tls_key):
        raise SystemExit(
            "ws:// bind cannot use TLS; switch to wss:// or remove --tls-*."
        )

    # Token resolution: CLI flag > YAML.
    cli_any_uid = bool(bind_allow_uid) or bool(bind_allow_any_uid)
    if cli_any_uid:
        raise SystemExit(
            "--bind-allow-uid / --bind-allow-any-uid are unix-only; "
            "use --bind-token-file with ws://wss://."
        )

    tokens: set[str] = set()
    if bind_token_file:
        tokens.update(_load_tokens_file(bind_token_file))
    elif yaml_token_auth is not None:
        yaml_tokens = yaml_token_auth.get("tokens")
        yaml_tokens_file = yaml_token_auth.get("tokens_file")
        if isinstance(yaml_tokens, list):
            tokens.update(str(t).strip() for t in yaml_tokens if str(t).strip())
        if yaml_tokens_file:
            tokens.update(_load_tokens_file(str(yaml_tokens_file)))

    if not tokens and not bind_allow_anonymous:
        raise SystemExit(
            f"{scheme}:// bind requires --bind-token-file (or "
            "bind.token_auth in YAML). Pass --bind-allow-anonymous to "
            "opt out (NOT recommended — anyone reachable on the network "
            "can drive the gateway)."
        )

    host = parsed.hostname or "0.0.0.0"
    port = parsed.port or (443 if scheme == "wss" else 80)
    bind_url = f"{scheme}://{host}:{port}{parsed.path or '/'}"

    return BindSpec(
        scheme=scheme,
        host=host,
        port=port,
        url=bind_url,
        tokens=frozenset(tokens),
        allow_anonymous=bool(bind_allow_anonymous) and not tokens,
        tls_cert=eff_tls_cert,
        tls_key=eff_tls_key,
    )


def _build_server_transport(spec: BindSpec) -> ServerTransport:
    """Materialize a :class:`ServerTransport` for an already-validated
    :class:`BindSpec`. Pure factory — no side effects beyond constructing
    the transport (no socket bind yet)."""
    if spec.scheme == "unix":
        return UnixServerTransport(spec.socket_path)
    ssl_context = None
    if spec.scheme == "wss":
        import ssl

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        assert spec.tls_cert is not None and spec.tls_key is not None
        ssl_context.load_cert_chain(certfile=spec.tls_cert, keyfile=spec.tls_key)
    return WebSocketServerTransport(
        host=spec.host,
        port=spec.port,
        ssl_context=ssl_context,
    )


def _build_bind_authenticator(spec: BindSpec) -> Authenticator:
    if spec.scheme == "unix":
        return UnixPeerCredAuthenticator(
            allowed_uids=set(spec.allow_uids)
            if spec.allow_uids is not None
            else None
        )
    if spec.allow_anonymous:
        from .server import AllowAllAuthenticator

        return AllowAllAuthenticator()
    return TokenAuthenticator(allowed_tokens=set(spec.tokens))


# -------- typer app -----------------------------------------------------


app = typer.Typer(
    name=PROG,
    add_completion=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        try:
            from . import __version__
        except ImportError:
            __version__ = "0.1.0"
        typer.echo(f"{PROG} {__version__}")
        raise typer.Exit(code=EXIT_OK)


@app.command()
def cli(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            envvar="AGENTM_CONFIG",
            help="YAML config (env vars expanded). Env: AGENTM_CONFIG.",
        ),
    ] = None,
    cwd: Annotated[
        str,
        typer.Option(
            "--cwd",
            envvar="AGENTM_CWD",
            help="Working directory for sessions. Default: $PWD. Env: AGENTM_CWD.",
        ),
    ] = "",
    scenario: Annotated[
        str,
        typer.Option(
            "--scenario",
            envvar="AGENTM_SCENARIO",
            help="Default scenario. Env: AGENTM_SCENARIO.",
        ),
    ] = "general_purpose",
    state_dir: Annotated[
        Path | None,
        typer.Option(
            "--state-dir",
            envvar="AGENTM_STATE_DIR",
            help=(
                "Persistent gateway state directory (outbox/inbox/session "
                "bindings). Default: <cwd>/.agentm/channels. Env: AGENTM_STATE_DIR."
            ),
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            envvar="AGENTM_PROVIDER",
            help=(
                "LLM provider id. Default: SDK default. Env: AGENTM_PROVIDER."
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
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            envvar="AGENTM_LOG_LEVEL",
            help="Logging level on stderr. Env: AGENTM_LOG_LEVEL.",
        ),
    ] = "INFO",
    bind: Annotated[
        str | None,
        typer.Option(
            "--bind",
            envvar="AGENTM_SOCKET",
            help=(
                "Run the wire-protocol server on the given URL. Supported: "
                "unix:///abs/path/to/sock (default; peer-cred auth, current uid "
                "only unless --bind-allow-uid / --bind-allow-any-uid), "
                "ws://host:port/path (token auth, plaintext — reverse-proxy "
                "for TLS), wss://host:port/path (token auth, TLS via "
                "--tls-cert/--tls-key). Env: AGENTM_SOCKET."
            ),
        ),
    ] = None,
    bind_token_file: Annotated[
        Path | None,
        typer.Option(
            "--bind-token-file",
            metavar="PATH",
            help=(
                "Path to a file containing one bearer token per line "
                "(blank lines and '#' comments are skipped). Required for "
                "ws://wss:// binds unless --bind-allow-anonymous is set."
            ),
        ),
    ] = None,
    bind_allow_anonymous: Annotated[
        bool,
        typer.Option(
            "--bind-allow-anonymous",
            help=(
                "Allow ws/wss bind without tokens. NOT recommended — anyone "
                "reachable on the network can drive the gateway. Intended "
                "for behind-a-trusted-reverse-proxy setups only."
            ),
        ),
    ] = False,
    tls_cert: Annotated[
        Path | None,
        typer.Option(
            "--tls-cert",
            metavar="PATH",
            help="TLS certificate (PEM) for wss:// binds. Required for wss://.",
        ),
    ] = None,
    tls_key: Annotated[
        Path | None,
        typer.Option(
            "--tls-key",
            metavar="PATH",
            help="TLS private key (PEM) for wss:// binds. Required for wss://.",
        ),
    ] = None,
    bind_allow_uid: Annotated[
        list[int] | None,
        typer.Option(
            "--bind-allow-uid",
            metavar="UID",
            help=(
                "Add a uid to the peer-cred allow-list. Repeatable. "
                "Mutually exclusive with --bind-allow-any-uid."
            ),
        ),
    ] = None,
    bind_allow_any_uid: Annotated[
        bool,
        typer.Option(
            "--bind-allow-any-uid",
            help=(
                "Accept any local peer uid. Use only on a fully trusted "
                "host. Mutually exclusive with --bind-allow-uid."
            ),
        ),
    ] = False,
    no_inproc_worker: Annotated[
        bool,
        typer.Option(
            "--no-inproc-worker",
            help=(
                "Refuse inbound when no matching external worker is "
                "connected; emit a chat message explaining the gap "
                "instead. Default: run the agent in-process when no "
                "external worker is connected."
            ),
        ),
    ] = False,
    max_a2a_hops: Annotated[
        int,
        typer.Option(
            "--max-a2a-hops",
            metavar="N",
            help=(
                "Maximum agent-to-agent hop count for forwarded inbound "
                f"envelopes. Default: {DEFAULT_MAX_A2A_HOPS}."
            ),
        ),
    ] = DEFAULT_MAX_A2A_HOPS,
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help=(
                "Load config, validate, and exit. No channels are started "
                "and no LLM is invoked. Exits 0 when the config is OK, 2 "
                "when something is missing or malformed."
            ),
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
    """Multi-channel chat gateway for AgentM.

    Drives Feishu / terminal / future channels through one bus, one
    command router, one approval bridge.

    Run as a wire-protocol daemon and connect clients separately:

      agentm-gateway --bind unix:///tmp/gw.sock --config gw.yaml
      agentm-terminal --connect unix:///tmp/gw.sock

    For cross-host deployments, use WebSocket:

      agentm-gateway --bind ws://0.0.0.0:7777/agentm --bind-token-file /etc/agentm/tokens
      agentm-worker  --connect ws://gw.example.com:7777/agentm --token "$AGENTM_TOKEN"
    """
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    resolved_cwd = cwd or str(Path.cwd())

    bind_opts = BindOptions(
        bind=bind,
        bind_allow_uid=bind_allow_uid,
        bind_allow_any_uid=bind_allow_any_uid,
        bind_token_file=str(bind_token_file) if bind_token_file else None,
        bind_allow_anonymous=bind_allow_anonymous,
        tls_cert=str(tls_cert) if tls_cert else None,
        tls_key=str(tls_key) if tls_key else None,
    )

    try:
        rc = asyncio.run(
            _arun(
                config_path=config,
                cwd=resolved_cwd,
                scenario_flag=scenario,
                state_dir=state_dir,
                provider_flag=provider,
                model_flag=model,
                bind_opts=bind_opts,
                inproc_worker=not no_inproc_worker,
                max_a2a_hops=max_a2a_hops,
                check=check,
            )
        )
    except KeyboardInterrupt:
        rc = EXIT_SIGINT
    except SystemExit as exc:
        # Helpers (yaml loader, bind resolver) raise SystemExit("msg") on
        # config errors. argparse-era main() converted these to exit 2;
        # do the same here so the contract is preserved.
        if isinstance(exc.code, int):
            rc = exc.code
        elif exc.code is None or exc.code is True:
            rc = EXIT_CONFIG_ERROR
        else:
            sys.stderr.write(f"{exc.code}\n")
            rc = EXIT_CONFIG_ERROR
    raise typer.Exit(code=rc)


# -------- async run loop ----------------------------------------------


async def _arun(
    *,
    config_path: Path | None,
    cwd: str,
    scenario_flag: str,
    state_dir: Path | None,
    provider_flag: str | None,
    model_flag: str | None,
    bind_opts: BindOptions,
    inproc_worker: bool,
    max_a2a_hops: int,
    check: bool,
) -> int:
    cfg: dict[str, Any] = {}
    if config_path is not None:
        cfg = _load_yaml(config_path)

    # YAML cwd wins over the implicit $PWD default; explicit --cwd /
    # AGENTM_CWD beats YAML.
    implicit_cwd_default = str(Path.cwd())
    yaml_cwd = cfg.get("cwd")
    if cwd == implicit_cwd_default and isinstance(yaml_cwd, str) and yaml_cwd:
        resolved_cwd = yaml_cwd
    else:
        resolved_cwd = cwd

    bind_spec = _resolve_bind(
        bind=bind_opts.bind,
        bind_allow_uid=bind_opts.bind_allow_uid,
        bind_allow_any_uid=bind_opts.bind_allow_any_uid,
        bind_token_file=bind_opts.bind_token_file,
        bind_allow_anonymous=bind_opts.bind_allow_anonymous,
        tls_cert=bind_opts.tls_cert,
        tls_key=bind_opts.tls_key,
        cfg=cfg,
    )

    raw_channels = cfg.get("channels") or {}
    if not isinstance(raw_channels, dict):
        raise SystemExit(
            f"`channels:` must be a mapping, got {type(raw_channels).__name__}"
        )
    bad = sorted(n for n in raw_channels if n != "stub")
    if bad:
        raise SystemExit(
            f"config `channels:` block lists v0 in-process channels {bad}; "
            "those are removed. Run `agentm-gateway --bind unix:///path` "
            "and connect each platform as a separate client process "
            "(agentm-terminal, agentm-feishu)."
        )

    channels_cfg: dict[str, Any] = raw_channels

    resolved_provider = provider_flag or cfg.get("provider") or _default_provider()
    resolved_model = (
        model_flag or cfg.get("model") or _default_model(resolved_provider)
    )
    resolved_scenario = scenario_flag or cfg.get("scenario")

    bus = MessageBus()
    retry_delays = _retry_delays(cfg.get("send_retry_delays"))
    manager = (
        ChannelManager(channels_cfg, bus, send_retry_delays=retry_delays)
        if retry_delays is not None
        else ChannelManager(channels_cfg, bus)
    )
    atom_enabled, atom_allow, skill_paths = _commands_section(cfg.get("commands"))
    gateway = Gateway(
        bus=bus,
        config=GatewayConfig(
            cwd=resolved_cwd,
            scenario=resolved_scenario,
            state_dir=(
                state_dir
                or (Path(cfg["state_dir"]) if "state_dir" in cfg else None)
            ),
            approval_policy=_approval_policy(cfg.get("approval")),
            atom_commands_enabled=atom_enabled,
            atom_allow=atom_allow,
            skill_paths=skill_paths,
        ),
        session_factory=_build_session_factory(
            scenario=resolved_scenario,
            provider=resolved_provider,
            model=resolved_model,
        ),
    )

    resolved_state_dir = (
        state_dir
        or (Path(cfg["state_dir"]) if "state_dir" in cfg else None)
        or (Path(resolved_cwd) / ".agentm" / "channels")
    )

    if check:
        logger.info("config OK: channels=%s", sorted(manager.channels))
        bind_payload: dict[str, Any]
        if bind_spec.scheme == "unix":
            bind_payload = {
                "scheme": "unix",
                "socket": f"unix://{bind_spec.socket_path}",
                "allow_uids": (
                    sorted(bind_spec.allow_uids)
                    if bind_spec.allow_uids is not None
                    else None
                ),
            }
        else:
            bind_payload = {
                "scheme": bind_spec.scheme,
                "url": bind_spec.url,
                "token_count": len(bind_spec.tokens),
                "allow_anonymous": bind_spec.allow_anonymous,
                "tls": bind_spec.scheme == "wss",
            }
        check_payload: dict[str, Any] = {
            "kind": "check",
            "channels": sorted(manager.channels),
            "state_dir": str(resolved_state_dir),
            "bind": bind_payload,
        }
        sys.stdout.write(json.dumps(check_payload) + "\n")
        sys.stdout.flush()
        return EXIT_OK

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover — Windows.
            pass

    await manager.start()
    await gateway.start()

    resolved_state_dir.mkdir(parents=True, exist_ok=True)
    wire_outbox = SqliteOutbox(str(resolved_state_dir / "wire-outbox.sqlite"))
    wire_inbox = SqliteInbox(str(resolved_state_dir / "wire-inbox.sqlite"))
    session_bindings = SessionBindingStore(
        resolved_state_dir / "session-bindings.sqlite"
    )
    bridge = WireBridge(
        bus=bus,
        manager=manager,
        outbox=wire_outbox,
        bindings=session_bindings,
        scenario=resolved_scenario or "",
        allow_inproc=inproc_worker,
        max_a2a_hops=int(max_a2a_hops),
    )
    authenticator = _build_bind_authenticator(bind_spec)
    server_transport = _build_server_transport(bind_spec)
    wire_server = WireServer(
        transport=server_transport,
        outbox=wire_outbox,
        inbox=wire_inbox,
        on_inbound=bridge.handle_inbound,
        authenticator=authenticator,
        on_peer_hello=bridge.handle_peer_hello,
        on_peer_disconnect=bridge.handle_peer_disconnect,
        on_worker_outbound=bridge.handle_worker_outbound,
    )
    await wire_server.start()
    if bind_spec.scheme == "unix":
        logger.info(
            "wire server bound at unix://%s (allow_uids=%s)",
            bind_spec.socket_path,
            sorted(bind_spec.allow_uids)
            if bind_spec.allow_uids is not None
            else "any",
        )
    else:
        logger.info(
            "wire server bound at %s (auth=%s)",
            bind_spec.url,
            "anonymous"
            if bind_spec.allow_anonymous
            else f"token({len(bind_spec.tokens)})",
        )

    logger.info(
        "gateway running with channels: %s", sorted(manager.channels) or "(none)"
    )

    try:
        await stop_event.wait()
    finally:
        logger.info("gateway shutting down")
        await wire_server.stop()
        wire_outbox.close()
        wire_inbox.close()
        await gateway.stop()
        await manager.stop()
    return EXIT_OK


# -------- entrypoint --------------------------------------------------


def main() -> None:
    """Entry point referenced by the ``agentm-gateway`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
