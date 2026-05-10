"""``agentm-gateway`` console entry.

Reads a YAML config (or env vars), builds a :class:`MessageBus`,
spins up a :class:`ChannelManager` and a :class:`Gateway`, and runs
until the process is signalled to stop.

Minimal config (``gateway.yaml``)::

    cwd: ./workspace
    scenario: feishu_chat
    provider: openai
    model: Doubao-Seed-2.0-pro

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
launch works:

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

from agentm.core.abi import EventBus

from .approval import ApprovalPolicy
from .bus import MessageBus
from .gateway import Gateway, GatewayConfig
from .manager import ChannelManager


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

    from agentm.cli import (
        DEFAULT_PROVIDER_REGISTRY,
        _make_default_session_store,
        _resolve_session_state,
    )
    from agentm.harness import AgentSession, AgentSessionConfig

    async def factory(cwd: str, bus: EventBus, resume: str | None) -> Any:
        store = _make_default_session_store(cwd)
        state = _resolve_session_state(
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
        or os.environ.get("AGENTM_PROVIDER", "anthropic"),
    )
    p.add_argument(
        "--model",
        default=os.environ.get("AGENTM_GATEWAY_MODEL")
        or os.environ.get("AGENTM_MODEL", "claude-sonnet-4-5-20250929"),
    )
    p.add_argument(
        "--log-level", default=os.environ.get("AGENTM_GATEWAY_LOG_LEVEL", "INFO")
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

    channels_cfg = cfg.get("channels") or _channels_from_env(dict(os.environ))
    if not channels_cfg:
        raise SystemExit(
            "no channels configured. Pass --config <yaml> with a `channels:` "
            "section, or set LARK_APP_ID / LARK_APP_SECRET for the one-liner "
            "Feishu mode."
        )

    bus = MessageBus()
    manager = ChannelManager(channels_cfg, bus)
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
        ),
        session_factory=_build_session_factory(
            scenario=args.scenario or cfg.get("scenario"),
            provider=args.provider or cfg.get("provider", "anthropic"),
            model=args.model or cfg.get("model", "claude-sonnet-4-5-20250929"),
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
    logging.getLogger("Lark").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _load_dotenv_files(Path(args.cwd))
    try:
        return asyncio.run(_arun(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
