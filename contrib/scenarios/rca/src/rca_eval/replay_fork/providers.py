"""Build provider tuples from ``~/.agentm/config.toml`` profiles.

The replay-fork experiment runs two models in one process: the harness
(extractor + auditor) on one endpoint and the main agent on another. Rather
than hand-assemble extension configs -- and risk the ambient ``OPENAI_*``
env bleeding the agent's endpoint into the harness -- reuse the exact path
``agentm --model <profile>`` takes: resolve the named profile, then let the
provider registry assemble the ``(module, config)`` tuple. The profile's
``base_url`` / ``api_key`` / ``context_window`` travel inside the config
dict, which the OpenAI extension prefers over the environment, so the two
models stay cleanly separated.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_profile_provider"]

def build_profile_provider(name: str) -> tuple[str, dict[str, Any]]:
    """Resolve a config.toml profile name to a ``(module, config)`` provider.

    Raises ``ValueError`` if no profile by that name exists -- a typo in the
    harness-model flag should fail loudly, not silently fall back to the
    agent's provider.
    """
    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.lib import resolve_model_profile

    profile = resolve_model_profile(name)
    if profile is None:
        raise ValueError(
            f"no ~/.agentm/config.toml profile named {name!r}; "
            "define it under [models.<name>] or pass an existing profile"
        )
    return DEFAULT_PROVIDER_REGISTRY.build(profile.provider, profile.to_build_config())
