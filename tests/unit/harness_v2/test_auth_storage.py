from __future__ import annotations

import time

import pytest

from agentm.ai.oauth import get_oauth_provider
from agentm.harness.auth_storage import AuthStorage


@pytest.mark.asyncio
async def test_auth_storage_prefers_stored_oauth_and_refreshes(monkeypatch) -> None:
    storage = AuthStorage.in_memory(
        {
            "anthropic": {
                "type": "oauth",
                "access": "old-access",
                "refresh": "refresh-token",
                "expires": 0,
            }
        }
    )
    provider = get_oauth_provider("anthropic")
    assert provider is not None

    async def _refresh(credentials):
        assert credentials["refresh"] == "refresh-token"
        return {
            "access": "new-access",
            "refresh": "refresh-token-2",
            "expires": int(time.time() * 1000) + 60_000,
        }

    monkeypatch.setattr(provider, "refresh_token", _refresh)

    resolved = await storage.resolve_async("anthropic")
    assert resolved.api_key == "new-access"
    assert storage.list_credentials()["anthropic"]["refresh"] == "refresh-token-2"


def test_auth_storage_falls_back_to_environment(monkeypatch) -> None:
    storage = AuthStorage.in_memory()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    resolved = storage.resolve("openai")
    assert resolved.api_key == "sk-openai"
    assert resolved.status.source == "environment"
