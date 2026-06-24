"""Resolve AgentM model profiles for host-side experiment runners."""

from __future__ import annotations

from typing import Any

__all__ = ["build_profile_provider"]


def build_profile_provider(name: str) -> tuple[str, dict[str, Any]]:
    """Resolve a config.toml profile name to a ``(module, config)`` provider."""

    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.lib import resolve_model_profile

    profile = resolve_model_profile(name)
    if profile is None:
        raise ValueError(
            f"no ~/.agentm/config.toml profile named {name!r}; "
            "define it under [models.<name>] or pass an existing profile"
        )
    return DEFAULT_PROVIDER_REGISTRY.build(profile.provider, profile.to_build_config())
