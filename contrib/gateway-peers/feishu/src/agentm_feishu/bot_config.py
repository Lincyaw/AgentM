"""Parse ``[feishu.bots.*]`` from config.toml into a list of FeishuConfig.

Multi-bot support: a single ``agentm-feishu`` process can host N Feishu bots,
each configured under ``[feishu.bots.<name>]`` in the user's config.toml
(``$AGENTM_HOME/config.toml`` or ``~/.agentm/config.toml``).

This module reads the TOML file independently — it does NOT import or modify
``agentm.core.lib.user_config`` (that module is for model profiles).
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

from .adapter import FeishuConfig

log = logging.getLogger("agentm_feishu.bot_config")


def _agentm_home() -> Path:
    home = os.environ.get("AGENTM_HOME")
    return Path(home) if home else Path.home() / ".agentm"


def _read_secret(raw: dict[str, Any], bot_name: str) -> str | None:
    """Resolve app_secret from the bot config table.

    Precedence: app_secret_file > app_secret > env LARK_APP_SECRET_{BOT_NAME_UPPER}
    """
    secret_file = raw.get("app_secret_file")
    if isinstance(secret_file, str) and secret_file:
        try:
            return Path(secret_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            log.warning(
                "[feishu.bots.%s] cannot read app_secret_file %r: %s",
                bot_name,
                secret_file,
                exc,
            )
            return None

    direct = raw.get("app_secret")
    if isinstance(direct, str) and direct:
        return direct

    env_key = f"LARK_APP_SECRET_{bot_name.upper()}"
    env_val = os.environ.get(env_key, "").strip()
    return env_val or None


def load_bot_configs() -> list[tuple[str, FeishuConfig]]:
    """Load ``[feishu.bots.*]`` from config.toml.

    Returns a list of ``(bot_name, FeishuConfig)`` tuples.
    Returns an empty list if the section is missing or the file doesn't exist.
    """
    path = _agentm_home() / "config.toml"
    if not path.is_file():
        return []

    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        log.warning("config.toml: failed to parse %s", path, exc_info=True)
        return []

    feishu = data.get("feishu")
    if not isinstance(feishu, dict):
        return []

    bots = feishu.get("bots")
    if not isinstance(bots, dict):
        return []

    result: list[tuple[str, FeishuConfig]] = []
    for name, raw in bots.items():
        if not isinstance(raw, dict):
            log.warning(
                "config.toml: [feishu.bots.%s] is not a table; skipped", name
            )
            continue

        app_id = raw.get("app_id")
        if not isinstance(app_id, str) or not app_id:
            log.warning(
                "config.toml: [feishu.bots.%s] missing app_id; skipped", name
            )
            continue

        app_secret = _read_secret(raw, name)
        if not app_secret:
            log.warning(
                "config.toml: [feishu.bots.%s] no app_secret resolved "
                "(set app_secret, app_secret_file, or LARK_APP_SECRET_%s); skipped",
                name,
                name.upper(),
            )
            continue

        channel_name = raw.get("channel_name")
        if not isinstance(channel_name, str) or not channel_name:
            channel_name = name  # default channel_name = bot key name

        scenario = raw.get("scenario")
        if not isinstance(scenario, str):
            scenario = None

        session_scope = raw.get("session_scope", "chat")
        if session_scope not in ("chat", "user"):
            log.warning(
                "config.toml: [feishu.bots.%s] session_scope %r invalid; using 'chat'",
                name,
                session_scope,
            )
            session_scope = "chat"

        raw_allow = raw.get("allow_from")
        if isinstance(raw_allow, list):
            allow_from = [str(x) for x in raw_allow]
        else:
            allow_from = ["*"]

        result.append((
            name,
            FeishuConfig(
                app_id=app_id,
                app_secret=app_secret,
                channel_name=channel_name,
                scenario=scenario,
                session_scope=session_scope,
                allow_from=allow_from,
            ),
        ))

    return result


__all__ = ["load_bot_configs"]
