"""Builtin ``inherit_provider`` atom: re-publish a parent's active provider.

Wired automatically by :func:`AgentSession._spawn_child_session` when a
caller passes ``AgentSessionConfig(provider=None)``: the spawn factory
injects this module as the child's provider extension and ``install()``
re-registers the parent's :class:`ProviderConfig` verbatim — no
re-authentication, no second LLM gateway handshake.
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from agentm.core.abi.roles import PARENT_PROVIDER_CONFIG_KEY, PROVIDER_INHERITOR
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI, ExtensionLoadError, ProviderConfig


class InheritProviderConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    parent_provider_config: Any  # ProviderConfig, injected by the spawn factory


MANIFEST = ExtensionManifest(
    name="inherit_provider",
    description=(
        "Re-publish a parent session's active LLM provider into a child "
        "session without re-authenticating. Wired automatically by "
        "spawn_child_session when AgentSessionConfig.provider is None."
    ),
    registers=("provider:<inherited>",),
    config_schema=InheritProviderConfig,
    requires=(),  # Leaf provider shim: consumes only injected config.
    api_version=1,
    tier=1,
    provides_role=(PROVIDER_INHERITOR,),
)


def install(api: ExtensionAPI, config: InheritProviderConfig) -> None:
    provider = config.parent_provider_config
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


__all__: Final = ["MANIFEST", "install"]
