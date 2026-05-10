"""Console entry — ``agentm-feishu-gateway``.

Reads gateway config from a YAML file or environment, constructs a
:class:`FeishuChatSource`, and runs :class:`FeishuGateway`. The session
factory wires each chat to an :class:`agentm.harness.AgentSession`
mounted on the configured scenario (``feishu_chat`` by default).

Configuration (any of these, with later sources winning):

1. ``--config <path>`` — YAML file. Supports ``app_id`` / ``app_secret``
   / ``scenario`` / ``cwd`` / ``state_dir`` / ``approval`` /
   ``response_in_card``.
2. Environment: ``LARK_APP_ID``, ``LARK_APP_SECRET``,
   ``AGENTM_FEISHU_SCENARIO``, ``AGENTM_FEISHU_CWD``.
3. CLI flags — see ``--help``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agentm.core.abi import EventBus

from .approval import ApprovalPolicy
from .chat_source import ChatSource
from .feishu_source import FeishuChatSource
from .gateway import FeishuGateway, GatewayConfig


logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # local import: pyyaml is a runtime dep but only needed here.

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config file {path} must contain a YAML mapping")
    return data


def _approval_policy(spec: Any) -> ApprovalPolicy:
    if not isinstance(spec, dict):
        return ApprovalPolicy()
    return ApprovalPolicy(
        always_allow=frozenset(str(x) for x in spec.get("always_allow", [])),
        always_block=frozenset(str(x) for x in spec.get("always_block", [])),
        require_approval=frozenset(str(x) for x in spec.get("require_approval", [])),
        timeout_seconds=float(spec.get("timeout_seconds", 300.0)),
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentm-feishu-gateway",
        description="Long-running daemon mediating between Feishu chats and AgentM.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config file. Wins over env vars; CLI flags win over both.",
    )
    parser.add_argument("--app-id", help="Feishu app ID (or LARK_APP_ID env)")
    parser.add_argument("--app-secret", help="Feishu app secret (or LARK_APP_SECRET env)")
    parser.add_argument(
        "--cwd",
        default=os.environ.get("AGENTM_FEISHU_CWD") or str(Path.cwd()),
        help="Working directory each AgentSession runs against.",
    )
    parser.add_argument(
        "--scenario",
        default=os.environ.get("AGENTM_FEISHU_SCENARIO", "feishu_chat"),
        help="Scenario manifest mounted on each session.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Where to persist chat→session map (default: <cwd>/.agentm/feishu/).",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("AGENTM_FEISHU_PROVIDER", "anthropic"),
        help="LLM provider name (default: anthropic).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AGENTM_FEISHU_MODEL", "claude-sonnet-4-5-20250929"),
        help="LLM model id passed to the provider.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("AGENTM_FEISHU_LOG_LEVEL", "INFO"),
    )
    return parser.parse_args(argv)


def _build_session_factory(
    *,
    scenario: str | None,
    provider: str,
    model: str,
) -> Callable[[str, EventBus, str | None], Awaitable[Any]]:
    # Imported lazily so tests can drive the gateway with a fake factory
    # without importing the full harness at module load.
    from typing import cast as _cast

    from agentm.cli import (
        DEFAULT_PROVIDER_REGISTRY,
        _make_default_session_store,
        _resolve_session_state,
    )
    from agentm.harness import AgentSession, AgentSessionConfig

    async def factory(cwd: str, bus: EventBus, resume: str | None) -> Any:
        store = _make_default_session_store(cwd)
        session_state = _resolve_session_state(
            cwd=cwd,
            resume=resume,
            continue_recent=False,
            session_store=store,
        )
        provider_spec = DEFAULT_PROVIDER_REGISTRY.build(provider, {"model": model})
        config = AgentSessionConfig(
            cwd=cwd,
            provider=provider_spec,
            scenario=scenario,
            session_manager=_cast(Any, session_state),
            bus=bus,
        )
        return await AgentSession.create(config)

    return factory


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = _load_yaml(args.config)

    app_id = args.app_id or cfg.get("app_id") or os.environ.get("LARK_APP_ID")
    app_secret = (
        args.app_secret or cfg.get("app_secret") or os.environ.get("LARK_APP_SECRET")
    )
    if not app_id or not app_secret:
        raise SystemExit(
            "missing Feishu credentials: pass --app-id/--app-secret, "
            "set LARK_APP_ID / LARK_APP_SECRET, or supply them in --config."
        )

    scenario = args.scenario or cfg.get("scenario")
    cwd = args.cwd or cfg.get("cwd") or str(Path.cwd())
    state_dir = args.state_dir or (Path(cfg["state_dir"]) if "state_dir" in cfg else None)
    policy = _approval_policy(cfg.get("approval"))
    response_in_card = bool(cfg.get("response_in_card", False))

    source: ChatSource = FeishuChatSource(app_id=app_id, app_secret=app_secret)
    gateway_config = GatewayConfig(
        cwd=cwd,
        scenario=scenario,
        state_dir=state_dir,
        approval_policy=policy,
        response_in_card=response_in_card,
    )
    gateway = FeishuGateway(
        source=source,
        config=gateway_config,
        session_factory=_build_session_factory(
            scenario=scenario, provider=args.provider, model=args.model
        ),
    )

    try:
        asyncio.run(gateway.run())
    except KeyboardInterrupt:
        logger.info("interrupted; shutting down")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
