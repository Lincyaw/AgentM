"""``agentm-gateway`` console entry.

Reads a YAML config (or env vars), builds a :class:`MessageBus`,
spins up a :class:`ChannelManager` and a :class:`Gateway`, and runs
until the process is signalled to stop (SIGINT / SIGTERM) — *or*, in
``--terminal`` mode, until stdin closes (Ctrl-D / piped input
exhausted). The terminal-EOF path is what lets shell pipelines drive
the gateway without timeouts:

    printf '/help\\n/atom:list\\n' | \\
      agentm-gateway --terminal --terminal-format json --config gw.yaml

Minimal config (``gateway.yaml``)::

    cwd: ./workspace
    scenario: feishu_chat
    provider: openai
    model: <provider-default>

    approval:
      require_approval: [bash]
      timeout_seconds: 300

    channels:
      feishu:
        enabled: true
        app_id: ${LARK_APP_ID}
        app_secret: ${LARK_APP_SECRET}
        allow_from: ['*']

Env-var convenience: when no ``--config`` is passed, the CLI builds a
minimal config from ``LARK_APP_ID`` / ``LARK_APP_SECRET`` so a one-line
launch works::

    LARK_APP_ID=… LARK_APP_SECRET=… agentm-gateway --cwd ./work

``.env`` is auto-loaded from the cwd and the AgentM workspace root.

Exit codes (see also ``cli-design`` rule group 3):

* ``0`` — clean shutdown
* ``2`` — argument / config error (missing channels, bad YAML, empty allow_from)
* ``130`` — SIGINT (Ctrl-C)

JSON output contract (``--terminal --terminal-format json``):

* stdout is one JSON object per line; stderr carries logs only.
* ``{"kind":"ready"}`` — channel started, safe to start sending.
* ``{"kind":"message","content":"…","buttons":[…],"metadata":{…}}``
* ``{"kind":"turn_complete"}`` — emitted at the end of a real agent
  turn. Control commands (``/help`` / ``/status`` / …) do **not** emit
  ``turn_complete``; they emit one ``message`` object and that's it.
* ``{"kind":"stopped"}`` — final line; no further output is coming.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agentm.ai import DEFAULT_PROVIDER_REGISTRY
from agentm.core.abi import EventBus

from .approval import ApprovalPolicy
from .bus import MessageBus
from .gateway import Gateway, GatewayConfig
from .manager import ChannelManager


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

    from agentm.harness import (
        AgentSession,
        AgentSessionConfig,
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
            "Shell-driving recipe (no driver/wrapper needed):\n"
            "  printf '/help\\n/atom:list\\n' | \\\n"
            "    agentm-gateway --terminal --terminal-format json \\\n"
            "                   --config gw.yaml 2>/tmp/gw.err\n"
            "Stdout = one JSON object per line; stderr = logs."
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
        default=os.environ.get("AGENTM_GATEWAY_SCENARIO", "feishu_chat"),
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
        "--terminal",
        action="store_true",
        help=(
            "Run the gateway with only the terminal channel (stdin/stdout). "
            "Useful for local validation of slash commands and skill "
            "activation without configuring a chat platform."
        ),
    )
    p.add_argument(
        "--terminal-format",
        choices=("text", "json"),
        default=os.environ.get("AGENTM_GATEWAY_TERMINAL_FORMAT", "text"),
        help=(
            "Output format when --terminal is set. 'text' is human-"
            "friendly (ANSI colors, agent prefix). 'json' emits one "
            "JSON object per line — use this when a script or another "
            "agent is driving the gateway. See module docstring for the "
            "JSON contract."
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


def _channels_from_env(env: dict[str, str]) -> dict[str, Any]:
    """Build a minimal ``channels`` config from env vars (one-liner mode)."""
    channels: dict[str, Any] = {}
    if env.get("LARK_APP_ID") and env.get("LARK_APP_SECRET"):
        channels["feishu"] = {
            "enabled": True,
            "app_id": env["LARK_APP_ID"],
            "app_secret": env["LARK_APP_SECRET"],
            "allow_from": [
                x.strip()
                for x in env.get("LARK_ALLOW_FROM", "*").split(",")
                if x.strip()
            ]
            or ["*"],
        }
    return channels


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

    if args.terminal:
        # Terminal mode overrides any YAML/env channels — explicit flag
        # is unambiguous about intent (local validation, single user).
        channels_cfg: dict[str, Any] = {
            "terminal": {
                "enabled": True,
                "allow_from": ["*"],
                "format": args.terminal_format,
            }
        }
        # JSON mode is consumed by parsers; even WARNING-level logging
        # on stderr is the right home (operator can still see it via
        # ``2>``). Suppress INFO chatter unless the operator opted in.
        if args.log_level.upper() == "INFO":
            args.log_level = "WARNING"
    else:
        channels_cfg = cfg.get("channels") or _channels_from_env(dict(os.environ))
        if not channels_cfg:
            raise SystemExit(
                "no channels configured. Pass --config <yaml> with a "
                "`channels:` section, set LARK_APP_ID / LARK_APP_SECRET for "
                "the one-liner Feishu mode, or run with --terminal for local "
                "validation."
            )

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

    if args.check:
        # ``--check`` is a dry-run gate: everything above has already
        # parsed config, validated YAML, instantiated channels (which
        # is where allow_from / app_id / app_secret checks fire), and
        # built the session factory. If we got here, the config is
        # structurally OK. No channels are started → no LLM calls, no
        # network, no side effects.
        logger.info("config OK: channels=%s", sorted(manager.channels))
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
    logger.info(
        "gateway running with channels: %s", sorted(manager.channels) or "(none)"
    )

    # In ``--terminal`` mode, stdin EOF must teardown the daemon — that
    # is what makes ``printf '…' | agentm-gateway --terminal`` exit
    # cleanly without an external ``timeout``. The terminal channel
    # already sets its ``_stopped`` event on EOF; the manager's
    # per-channel start tasks complete when it does. So in terminal
    # mode we race ``stop_event`` (SIGINT/SIGTERM) against "any
    # channel task finished naturally". Other modes only honour
    # signals — Feishu's channel.start() blocks forever and an EOF
    # condition isn't meaningful.
    if args.terminal:
        channel_tasks = list(manager._tasks)  # type: ignore[attr-defined]
        signal_wait = asyncio.create_task(stop_event.wait(), name="gw-signal")
        try:
            done, _pending = await asyncio.wait(
                [signal_wait, *channel_tasks],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if signal_wait in done:
                logger.info("gateway shutting down (signal)")
            else:
                logger.info("gateway shutting down (stdin EOF)")
        finally:
            if not signal_wait.done():
                signal_wait.cancel()
            await gateway.stop()
            await manager.stop()
        return EXIT_OK

    try:
        await stop_event.wait()
    finally:
        logger.info("gateway shutting down")
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
