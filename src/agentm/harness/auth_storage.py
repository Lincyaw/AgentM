from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.ai.env_api_keys import find_env_keys, get_env_api_key
from agentm.ai.oauth import get_oauth_provider
from agentm.ai.types import OAuthCredentials


@dataclass(frozen=True, slots=True)
class AuthStatus:
    configured: bool
    source: str | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedCredential:
    api_key: str | None
    status: AuthStatus


class AuthStorage:
    def __init__(self, auth_path: Path | None = None) -> None:
        self._auth_path = auth_path or _default_auth_path()
        self._runtime_overrides: dict[str, str] = {}
        self._fallback_resolver: Any = None
        self._data = self._load()

    @classmethod
    def create(cls, auth_path: str | None = None) -> "AuthStorage":
        return cls(Path(auth_path) if auth_path is not None else None)

    @classmethod
    def in_memory(
        cls, data: dict[str, dict[str, Any]] | None = None
    ) -> "AuthStorage":
        storage = cls(auth_path=Path("/tmp/agentm-auth-storage.json"))
        storage._auth_path = Path("/dev/null")
        storage._data = dict(data or {})
        return storage

    def path(self) -> Path:
        return self._auth_path

    def set_runtime_api_key(self, provider: str, api_key: str) -> None:
        self._runtime_overrides[provider] = api_key

    def remove_runtime_api_key(self, provider: str) -> None:
        self._runtime_overrides.pop(provider, None)

    def set_fallback_resolver(self, resolver: Any) -> None:
        self._fallback_resolver = resolver

    def reload(self) -> None:
        self._data = self._load()

    def get_status(self, provider: str) -> AuthStatus:
        return self.resolve(provider).status

    def resolve(self, provider: str) -> ResolvedCredential:
        stored = self._resolve_sync(provider)
        return stored

    async def resolve_async(self, provider: str) -> ResolvedCredential:
        runtime = self._runtime_overrides.get(provider)
        if runtime:
            return ResolvedCredential(
                api_key=runtime,
                status=AuthStatus(True, source="runtime", label="runtime override"),
            )

        stored = self._data.get(provider)
        if isinstance(stored, dict) and stored.get("type") == "oauth":
            oauth_provider = get_oauth_provider(provider)
            if oauth_provider is not None:
                credentials = _coerce_oauth_credentials(stored)
                if credentials is not None:
                    if _expired(credentials):
                        credentials = await oauth_provider.refresh_token(credentials)
                        self._data[provider] = {"type": "oauth", **credentials}
                        self._save()
                    api_key = oauth_provider.get_api_key(credentials)
                    return ResolvedCredential(
                        api_key=api_key,
                        status=AuthStatus(
                            True, source="stored", label="stored OAuth token"
                        ),
                    )

        return self._resolve_sync(provider)

    def _resolve_sync(self, provider: str) -> ResolvedCredential:
        runtime = self._runtime_overrides.get(provider)
        if runtime:
            return ResolvedCredential(
                api_key=runtime,
                status=AuthStatus(True, source="runtime", label="runtime override"),
            )

        stored = self._data.get(provider)
        if isinstance(stored, dict):
            kind = stored.get("type")
            if kind == "api_key":
                key = stored.get("key")
                if isinstance(key, str) and key:
                    return ResolvedCredential(
                        api_key=key,
                        status=AuthStatus(True, source="stored", label="stored API key"),
                    )
            if kind == "oauth":
                oauth_provider = get_oauth_provider(provider)
                if oauth_provider is not None:
                    credentials = _coerce_oauth_credentials(stored)
                    if credentials is not None:
                        api_key = oauth_provider.get_api_key(credentials)
                        return ResolvedCredential(
                            api_key=api_key,
                            status=AuthStatus(True, source="stored", label="stored OAuth token"),
                        )

        env_key = get_env_api_key(provider)
        if env_key:
            env_names = find_env_keys(provider)
            label = env_names[0] if env_names else "environment"
            return ResolvedCredential(
                api_key=env_key,
                status=AuthStatus(True, source="environment", label=label),
            )

        if self._fallback_resolver is not None:
            key = self._fallback_resolver(provider)
            if isinstance(key, str) and key:
                return ResolvedCredential(
                    api_key=key,
                    status=AuthStatus(True, source="fallback", label="fallback resolver"),
                )

        return ResolvedCredential(api_key=None, status=AuthStatus(False))

    def store_api_key(self, provider: str, api_key: str) -> None:
        self._data[provider] = {"type": "api_key", "key": api_key}
        self._save()

    def store_oauth(self, provider: str, credentials: OAuthCredentials) -> None:
        self._data[provider] = {"type": "oauth", **credentials}
        self._save()

    def remove(self, provider: str) -> None:
        self._data.pop(provider, None)
        self._save()

    def list_credentials(self) -> dict[str, dict[str, Any]]:
        return dict(self._data)

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self._auth_path.exists() or self._auth_path == Path("/dev/null"):
            return {}
        try:
            payload = json.loads(self._auth_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if isinstance(payload, dict):
            return {
                str(key): value
                for key, value in payload.items()
                if isinstance(value, dict)
            }
        return {}

    def _save(self) -> None:
        if self._auth_path == Path("/dev/null"):
            return
        self._auth_path.parent.mkdir(parents=True, exist_ok=True)
        self._auth_path.write_text(
            json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.chmod(self._auth_path, 0o600)


def _default_auth_path() -> Path:
    configured = os.environ.get("AGENTM_AUTH_PATH")
    if configured:
        return Path(configured)
    return Path.home() / ".agentm" / "auth.json"


def _expired(credentials: OAuthCredentials) -> bool:
    import time

    return int(time.time() * 1000) >= credentials["expires"]


def _coerce_oauth_credentials(data: dict[str, Any]) -> OAuthCredentials | None:
    refresh = data.get("refresh")
    access = data.get("access")
    expires = data.get("expires")
    if isinstance(refresh, str) and isinstance(access, str) and isinstance(expires, int):
        return {"refresh": refresh, "access": access, "expires": expires}
    return None
