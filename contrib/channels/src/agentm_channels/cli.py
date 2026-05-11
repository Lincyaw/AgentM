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
* ``2`` — argument / config error (no --bind, bad YAML, legacy channels: block)
* ``130`` — SIGINT (Ctrl-C)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agentm.ai import DEFAULT_PROVIDER_REGISTRY

from . import DEFAULT_SOCKET_URL
from agentm.core.abi import EventBus

from .approval import ApprovalPolicy
from .auth import UnixPeerCredAuthenticator
from .bus import MessageBus
from .gateway import Gateway, GatewayConfig
from .manager import ChannelManager
from .outbox import SqliteInbox, SqliteOutbox
from .server import WireServer
from .session_bindings import SessionBindingStore
from .wire_bridge import DEFAULT_MAX_A2A_HOPS, WireBridge


def _default_provider() -> str:
    return DEFAULT_PROVIDER_REGISTRY.default_provider().id


def _default_model(provider: str) -> str:
    return DEFAULT_PROVIDER_REGISTRY.default_model(provider)


logger = logging.getLogger(__name__)


# -------- config loading ------------------------------------------------


def _load_dotenv_files(cwd: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    candidates: list[Path] = [cwd / ".env"]
    walker = cwd.resolve()
    for _ in range(8):
        manifest = walker / "pyproject.toml"
        if manifest.exists():
            try:
                if "[tool.uv.workspace]" in manifest.read_text(encoding="utf-8"):
                    candidates.append(walker / ".env")
                    break
            except OSError:
                pass
        if walker.parent == walker:
            break
        walker = walker.parent
    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)


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
class BindSpec:
    """Resolved ``--bind`` configuration. ``allow_uids`` of ``None``
    means any-uid (peer-cred reads the kernel uid but doesn't filter)."""

    socket_path: str
    allow_uids: frozenset[int] | None


def _resolve_bind(
    args: argparse.Namespace, cfg: dict[str, Any]
) -> BindSpec:
    """Merge CLI flags > yaml ``bind:`` section > shared default.

    The gateway is always a wire-protocol daemon: when neither the CLI
    flag nor the YAML ``bind:`` section sets a socket, fall back to
    :data:`DEFAULT_SOCKET_URL`. Raises :class:`SystemExit` (exit code 2)
    on invalid input — TCP scheme, or conflicting allow-uid flags.
    """
    cli_url = args.bind
    yaml_bind = cfg.get("bind") if isinstance(cfg.get("bind"), dict) else None
    yaml_url = (yaml_bind or {}).get("socket")
    url = cli_url or yaml_url or DEFAULT_SOCKET_URL
    parsed = urlparse(str(url))
    if parsed.scheme != "unix":
        raise SystemExit(
            f"--bind scheme {parsed.scheme!r} not supported; only unix:// is "
            "available in v1 (TCP is deferred per "
            ".claude/designs/client-server-architecture.md §5.2)."
        )
    # urlparse on ``unix:///abs/path`` gives netloc="" and path="/abs/path".
    socket_path = parsed.path or parsed.netloc
    if not socket_path:
        raise SystemExit(
            f"--bind {url!r} has no socket path; use unix:///abs/path/to/sock"
        )

    cli_uids = list(args.bind_allow_uid or ())
    cli_any = bool(args.bind_allow_any_uid)
    if cli_uids and cli_any:
        raise SystemExit(
            "--bind-allow-uid and --bind-allow-any-uid are mutually exclusive."
        )

    if cli_uids or cli_any:
        # CLI side made an explicit decision; YAML is ignored to keep
        # the precedence rule honest.
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
                raise SystemExit(f"bind.allow_uids: invalid entry: {exc}") from exc
    else:
        # Most-secure default: current process uid only.
        allow_uids = frozenset({os.geteuid()})

    return BindSpec(socket_path=socket_path, allow_uids=allow_uids)


# -------- argv ---------------------------------------------------------


EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_SIGINT = 130


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="agentm-gateway",
        description=(
            "Multi-channel chat gateway for AgentM. "
            "Drives Feishu / terminal / future channels through one bus, "
            "one command router, one approval bridge."
        ),
        epilog=(
            "Run as a wire-protocol daemon and connect clients separately:\n"
            "  agentm-gateway --bind unix:///tmp/gw.sock --config gw.yaml\n"
            "  agentm-terminal --connect unix:///tmp/gw.sock\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, help="YAML config (env vars expanded).")
    p.add_argument(
        "--cwd",
        default=os.environ.get("AGENTM_GATEWAY_CWD") or str(Path.cwd()),
    )
    p.add_argument(
        "--scenario",
        default=os.environ.get("AGENTM_GATEWAY_SCENARIO", "general_purpose"),
    )
    p.add_argument("--state-dir", type=Path)
    p.add_argument(
        "--provider",
        default=os.environ.get("AGENTM_GATEWAY_PROVIDER")
        or os.environ.get("AGENTM_PROVIDER")
        or _default_provider(),
    )
    # Resolved against the *chosen* provider in :func:`_arun` once the
    # provider is final (config + env + CLI all merged); the bare
    # default here is None so an explicit ``--model`` still wins.
    p.add_argument("--model", default=None)
    p.add_argument(
        "--log-level", default=os.environ.get("AGENTM_GATEWAY_LOG_LEVEL", "INFO")
    )
    p.add_argument(
        "--bind",
        default=None,
        help=(
            "Run the wire-protocol server on the given URL. v1 supports "
            "only ``unix:///abs/path/to/sock`` (TCP is deferred). "
            f"When omitted, defaults to ``{DEFAULT_SOCKET_URL}`` "
            "($XDG_RUNTIME_DIR/agentm-gw.sock if set, else "
            "/tmp/agentm-gw-<uid>.sock). Authentication is Unix peer-cred; "
            "by default only the current process's uid may connect — "
            "override with --bind-allow-uid or --bind-allow-any-uid."
        ),
    )
    p.add_argument(
        "--bind-allow-uid",
        action="append",
        type=int,
        default=None,
        metavar="UID",
        help=(
            "Add a uid to the peer-cred allow-list. Repeatable. Ignored "
            "unless --bind is set. Mutually exclusive with "
            "--bind-allow-any-uid."
        ),
    )
    p.add_argument(
        "--bind-allow-any-uid",
        action="store_true",
        help=(
            "Accept any local peer uid. Use only on a fully trusted host. "
            "Mutually exclusive with --bind-allow-uid."
        ),
    )
    # Phase 5a: gateway can refuse inbound when no external worker is
    # connected. Default is unchanged (in-process worker); flip to
    # ``--no-inproc-worker`` to require an ``agentm-worker`` peer.
    p.add_argument(
        "--inproc-worker",
        dest="inproc_worker",
        action="store_true",
        default=True,
        help=(
            "Run the agent in-process when no external worker is "
            "connected (default). Behaviour matches Phases 1-4."
        ),
    )
    p.add_argument(
        "--no-inproc-worker",
        dest="inproc_worker",
        action="store_false",
        help=(
            "Refuse inbound when no matching external worker is "
            "connected; emit a chat message explaining the gap "
            "instead. Requires --bind."
        ),
    )
    p.add_argument(
        "--max-a2a-hops",
        type=int,
        default=DEFAULT_MAX_A2A_HOPS,
        metavar="N",
        help=(
            "Maximum agent-to-agent hop count for forwarded inbound "
            "envelopes. Each forward through the gateway increments "
            "the hop counter; envelopes that would exceed this cap are "
            f"dropped and a hop_limit_exceeded error is returned to the "
            f"sender. Default: {DEFAULT_MAX_A2A_HOPS}."
        ),
    )
    p.add_argument(
        "--check",
        action="store_true",
        help=(
            "Load config, validate, and exit. No channels are started "
            "and no LLM is invoked. Exits 0 when the config is OK, 2 "
            "when something is missing or malformed. Suitable as a "
            "pre-flight gate in CI or before promoting a deployment."
        ),
    )
    return p.parse_args(argv)


# -------- main ---------------------------------------------------------


async def _arun(args: argparse.Namespace) -> int:
    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = _load_yaml(args.config)
    # CLI > YAML > env default. The earlier ``args.cwd`` win was a
    # silent foot-gun: a YAML ``cwd: /tmp/foo`` would be ignored and
    # the gateway would scan the shell's pwd for ``.agentm/commands``
    # / ``.claude/skills``. Honour the YAML value when the CLI did
    # not override.
    yaml_cwd = cfg.get("cwd")
    args_cwd_default = (
        os.environ.get("AGENTM_GATEWAY_CWD") or str(Path.cwd())
    )
    if args.cwd == args_cwd_default and isinstance(yaml_cwd, str) and yaml_cwd:
        cwd = yaml_cwd
    else:
        cwd = args.cwd

    bind_spec = _resolve_bind(args, cfg)

    raw_channels = cfg.get("channels") or {}
    if not isinstance(raw_channels, dict):
        raise SystemExit(f"`channels:` must be a mapping, got {type(raw_channels).__name__}")
    # In-process platform channels are gone; only the test-only ``stub``
    # adapter stays usable through this YAML path (kept so the bind-flow
    # smoke tests can run without spinning up a real platform client).
    bad = sorted(n for n in raw_channels if n != "stub")
    if bad:
        raise SystemExit(
            f"config `channels:` block lists v0 in-process channels {bad}; "
            "those are removed. Run `agentm-gateway --bind unix:///path` "
            "and connect each platform as a separate client process "
            "(agentm-terminal, agentm-feishu)."
        )

    # bind_spec is always set now — _resolve_bind falls back to the
    # shared default socket URL when neither CLI nor YAML supplied one.
    assert bind_spec is not None
    channels_cfg: dict[str, Any] = raw_channels

    provider = args.provider or cfg.get("provider") or _default_provider()
    model = (
        args.model
        or os.environ.get("AGENTM_GATEWAY_MODEL")
        or os.environ.get("AGENTM_MODEL")
        or cfg.get("model")
        or _default_model(provider)
    )

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
            cwd=cwd,
            scenario=args.scenario or cfg.get("scenario"),
            state_dir=(
                args.state_dir
                or (Path(cfg["state_dir"]) if "state_dir" in cfg else None)
            ),
            approval_policy=_approval_policy(cfg.get("approval")),
            atom_commands_enabled=atom_enabled,
            atom_allow=atom_allow,
            skill_paths=skill_paths,
        ),
        session_factory=_build_session_factory(
            scenario=args.scenario or cfg.get("scenario"),
            provider=provider,
            model=model,
        ),
    )

    state_dir = (
        args.state_dir
        or (Path(cfg["state_dir"]) if "state_dir" in cfg else None)
        or (Path(cwd) / ".agentm" / "channels")
    )

    if args.check:
        # ``--check`` is a dry-run gate: everything above has already
        # parsed config, validated YAML, instantiated channels (which
        # is where allow_from / app_id / app_secret checks fire), and
        # built the session factory. If we got here, the config is
        # structurally OK. No channels are started → no LLM calls, no
        # network, no side effects.
        logger.info("config OK: channels=%s", sorted(manager.channels))
        check_payload: dict[str, Any] = {
            "kind": "check",
            "channels": sorted(manager.channels),
            "state_dir": str(state_dir),
        }
        if bind_spec is not None:
            check_payload["bind"] = {
                "socket": f"unix://{bind_spec.socket_path}",
                "allow_uids": (
                    sorted(bind_spec.allow_uids)
                    if bind_spec.allow_uids is not None
                    else None
                ),
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

    wire_server: WireServer | None = None
    wire_outbox: SqliteOutbox | None = None
    wire_inbox: SqliteInbox | None = None
    if bind_spec is not None:
        state_dir.mkdir(parents=True, exist_ok=True)
        wire_outbox = SqliteOutbox(str(state_dir / "wire-outbox.sqlite"))
        wire_inbox = SqliteInbox(str(state_dir / "wire-inbox.sqlite"))
        # Routing table: session_key → host_id + resume_id. Lives next
        # to the outbox so the whole gateway state directory is one
        # consistent unit on disk. Survives gateway restart by design.
        session_bindings = SessionBindingStore(
            state_dir / "session-bindings.sqlite"
        )
        bridge = WireBridge(
            bus=bus,
            manager=manager,
            outbox=wire_outbox,
            bindings=session_bindings,
            scenario=args.scenario or cfg.get("scenario") or "",
            allow_inproc=bool(getattr(args, "inproc_worker", True)),
            max_a2a_hops=int(
                getattr(args, "max_a2a_hops", DEFAULT_MAX_A2A_HOPS)
            ),
        )
        authenticator = UnixPeerCredAuthenticator(
            allowed_uids=set(bind_spec.allow_uids)
            if bind_spec.allow_uids is not None
            else None
        )
        wire_server = WireServer(
            socket_path=bind_spec.socket_path,
            outbox=wire_outbox,
            inbox=wire_inbox,
            on_inbound=bridge.handle_inbound,
            authenticator=authenticator,
            on_peer_hello=bridge.handle_peer_hello,
            on_peer_disconnect=bridge.handle_peer_disconnect,
            on_worker_outbound=bridge.handle_worker_outbound,
        )
        await wire_server.start()
        logger.info(
            "wire server bound at unix://%s (allow_uids=%s)",
            bind_spec.socket_path,
            sorted(bind_spec.allow_uids)
            if bind_spec.allow_uids is not None
            else "any",
        )

    logger.info(
        "gateway running with channels: %s", sorted(manager.channels) or "(none)"
    )

    try:
        await stop_event.wait()
    finally:
        logger.info("gateway shutting down")
        if wire_server is not None:
            await wire_server.stop()
        if wire_outbox is not None:
            wire_outbox.close()
        if wire_inbox is not None:
            wire_inbox.close()
        await gateway.stop()
        await manager.stop()
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # ``httpx`` is left at the configured level so operators can confirm
    # provider traffic is happening — silencing it made it impossible to
    # tell whether the LLM was being called when the agent appeared
    # idle. Per-channel libraries (Lark/Slack/…) tune their own loggers
    # in their channel modules if needed.
    _load_dotenv_files(Path(args.cwd))
    try:
        return asyncio.run(_arun(args))
    except KeyboardInterrupt:
        return EXIT_SIGINT
    except SystemExit as exc:
        # Config / argument errors raise SystemExit with a message. Map
        # to the standardized exit code so callers can distinguish
        # "config malformed" (2, retry hopeless) from "operational
        # failure" (other) without parsing stderr.
        if exc.code is None or exc.code is True:
            return EXIT_CONFIG_ERROR
        if isinstance(exc.code, int):
            return exc.code
        # Non-int SystemExit code → argparse / SystemExit("message").
        # argparse already printed to stderr; return the canonical
        # config-error code.
        sys.stderr.write(f"{exc.code}\n")
        return EXIT_CONFIG_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
