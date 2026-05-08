"""Builtin ``inherit_provider`` atom: re-publish a parent's active provider.

Wired automatically by :func:`AgentSession._spawn_child_session` when a
caller passes ``AgentSessionConfig(provider=None)`` — the spawn factory
injects ``provider=("agentm.extensions.builtin.inherit_provider",
{"provider": <parent_ProviderConfig>})`` so the child loads this module
as its provider extension and ``install()`` re-registers the parent's
:class:`ProviderConfig` verbatim. No re-authentication, no second LLM
gateway handshake.

Before this atom existed, every spawning extension (the ``sub_agent``
builtin, the ``llmharness.adapters.agentm`` adapter) hand-rolled the
same trick — synthesizing ``provider=(__name__, {"_bridge_provider":
provider})`` and adding a marker-detecting branch to its own ``install``.
Three implementations of one idea is two too many; this atom is the
canonical home.

This atom never installs handlers — it does exactly one thing,
synchronously, on install: re-register the supplied provider. No
``api.on(...)``, no tool registrations, no event subscriptions.
"""

from __future__ import annotations

from typing import Any

from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI, ExtensionLoadError, ProviderConfig

MANIFEST = ExtensionManifest(
    name="inherit_provider",
    description=(
        "Re-publish a parent session's active LLM provider into a child "
        "session without re-authenticating. Wired automatically by "
        "spawn_child_session when AgentSessionConfig.provider is None."
    ),
    registers=("provider:<inherited>",),
    config_schema={
        "type": "object",
        "properties": {
            "provider": {
                "description": (
                    "ProviderConfig instance from the parent session. "
                    "Injected by the spawn factory; not user-settable."
                ),
            },
        },
        "required": ["provider"],
        "additionalProperties": False,
    },
    api_version=1,
    tier=1,
)


# Config key the spawn factory uses to hand the parent provider over. Public
# so the factory and tests can reference it without a magic-string copy.
PARENT_PROVIDER_CONFIG_KEY = "provider"


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    provider = config.get(PARENT_PROVIDER_CONFIG_KEY)
    if not isinstance(provider, ProviderConfig):
        raise ExtensionLoadError(
            __name__,
            ValueError(
                "inherit_provider requires a ProviderConfig at "
                f"config[{PARENT_PROVIDER_CONFIG_KEY!r}]; "
                f"got {type(provider).__name__}. The spawn factory injects "
                "this value automatically — direct callers should pass "
                "AgentSessionConfig(provider=None) and let the factory wire it."
            ),
        )
    api.register_provider(provider.name, provider)


__all__ = ["MANIFEST", "PARENT_PROVIDER_CONFIG_KEY", "install"]
