"""``agentm-gateway`` console entry.

Reads a YAML config (or env vars), builds a :class:`MessageBus`,
spins up a :class:`ChannelManager` and a :class:`Gateway`, and runs
until the process is signalled to stop.

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


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="agentm-gateway",
        description="Multi-channel chat gateway for AgentM.",
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
    cwd = args.cwd
    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = _load_yaml(args.config)

    if args.terminal:
        # Terminal mode overrides any YAML/env channels — explicit flag
        # is unambiguous about intent (local validation, single user).
        channels_cfg: dict[str, Any] = {
            "terminal": {"enabled": True, "allow_from": ["*"]}
        }
        # Also: a terminal session by definition has just one chat.
        # Logging to stdout at INFO would interleave with agent replies
        # and make the channel unreadable; drop the level unless the
        # operator explicitly raised it.
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
    try:
        await stop_event.wait()
    finally:
        logger.info("gateway shutting down")
        await gateway.stop()
        await manager.stop()
    return 0


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
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
