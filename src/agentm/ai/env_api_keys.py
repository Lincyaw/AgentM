"""Environment-based API key discovery backed by provider descriptors."""

from __future__ import annotations

from collections.abc import Mapping

from agentm.ai.types import DEFAULT_PROVIDER_REGISTRY


def find_env_keys(provider: str, env: Mapping[str, str] | None = None) -> list[str] | None:
    """Return env var names that are currently set for ``provider``."""

    return DEFAULT_PROVIDER_REGISTRY.find_env_keys(provider, env)


def get_env_api_key(provider: str, env: Mapping[str, str] | None = None) -> str | None:
    """Return the API key for ``provider`` from the highest-priority env var."""

    return DEFAULT_PROVIDER_REGISTRY.get_env_api_key(provider, env)
