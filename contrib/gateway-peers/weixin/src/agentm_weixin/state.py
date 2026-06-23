"""Persistent state management for weixin accounts, sync buffers, and context tokens.

Storage layout under ``~/.agentm/weixin/``:
  accounts.json           — list of registered account IDs
  accounts/<id>.json      — per-account credentials (token, baseUrl, userId)
  accounts/<id>.sync.json — getUpdates incremental cursor
  accounts/<id>.ctx.json  — context token cache (survives restarts)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


def _state_dir() -> Path:
    home = os.environ.get("AGENTM_HOME", "")
    if home:
        return Path(home) / "weixin"
    return Path.home() / ".agentm" / "weixin"


# -- Account data --------------------------------------------------------


@dataclass(slots=True)
class AccountData:
    token: str = ""
    base_url: str = ""
    cdn_base_url: str = ""
    user_id: str = ""


def _accounts_dir() -> Path:
    return _state_dir() / "accounts"


def _account_index_path() -> Path:
    return _state_dir() / "accounts.json"


def list_account_ids() -> list[str]:
    path = _account_index_path()
    try:
        if not path.exists():
            return []
        data = json.loads(path.read_text("utf-8"))
        if isinstance(data, list):
            return [s for s in data if isinstance(s, str) and s.strip()]
    except Exception:
        logger.warning(f"failed to read account index {path}")
    return []


def register_account_id(account_id: str) -> None:
    existing = list_account_ids()
    if account_id in existing:
        return
    existing.append(account_id)
    path = _account_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2), "utf-8")


def load_account(account_id: str) -> AccountData | None:
    path = _accounts_dir() / f"{account_id}.json"
    try:
        if not path.exists():
            return None
        d = json.loads(path.read_text("utf-8"))
        return AccountData(
            token=d.get("token", ""),
            base_url=d.get("base_url", ""),
            cdn_base_url=d.get("cdn_base_url", ""),
            user_id=d.get("user_id", ""),
        )
    except Exception:
        logger.warning(f"failed to load account {account_id}")
        return None


def save_account(account_id: str, data: AccountData) -> None:
    path = _accounts_dir() / f"{account_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "token": data.token,
        "base_url": data.base_url,
        "cdn_base_url": data.cdn_base_url,
        "user_id": data.user_id,
    }
    path.write_text(json.dumps(d, indent=2), "utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


# -- Sync buffer ----------------------------------------------------------


def load_sync_buf(account_id: str) -> str:
    path = _accounts_dir() / f"{account_id}.sync.json"
    try:
        if not path.exists():
            return ""
        d = json.loads(path.read_text("utf-8"))
        return d.get("get_updates_buf", "")
    except Exception:
        logger.warning(f"failed to load sync buf for {account_id}")
        return ""


def save_sync_buf(account_id: str, buf: str) -> None:
    path = _accounts_dir() / f"{account_id}.sync.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"get_updates_buf": buf}), "utf-8")


# -- Context tokens -------------------------------------------------------


class ContextTokenStore:
    """In-memory + disk-backed context token cache.

    The iLink API issues a ``context_token`` on every inbound message that
    MUST be echoed on every outbound send to the same user.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def _key(self, account_id: str, user_id: str) -> str:
        return f"{account_id}:{user_id}"

    def _path(self, account_id: str) -> Path:
        return _accounts_dir() / f"{account_id}.ctx.json"

    def set(self, account_id: str, user_id: str, token: str) -> None:
        self._store[self._key(account_id, user_id)] = token
        self._persist(account_id)

    def get(self, account_id: str, user_id: str) -> str | None:
        return self._store.get(self._key(account_id, user_id))

    def restore(self, account_id: str) -> None:
        path = self._path(account_id)
        try:
            if not path.exists():
                return
            data = json.loads(path.read_text("utf-8"))
            prefix = f"{account_id}:"
            count = 0
            for uid, tok in data.items():
                if isinstance(tok, str) and tok:
                    self._store[prefix + uid] = tok
                    count += 1
            logger.info(f"restored {count} context tokens for {account_id}")
        except Exception:
            logger.warning(f"failed to restore context tokens for {account_id}")

    def _persist(self, account_id: str) -> None:
        prefix = f"{account_id}:"
        tokens = {}
        for k, v in self._store.items():
            if k.startswith(prefix):
                tokens[k[len(prefix):]] = v
        path = self._path(account_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(tokens), "utf-8")
        except Exception:
            logger.warning(f"failed to persist context tokens for {account_id}")
