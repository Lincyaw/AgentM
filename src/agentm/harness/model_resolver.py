from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentm.ai.types import ProviderDefinition, ResolvedAuth
from agentm.core.kernel.stream import Model
from agentm.harness.auth_guidance import format_no_api_key_found_message
from agentm.harness.auth_storage import AuthStorage
from agentm.harness.extension import ProviderConfig
from agentm.harness.model_registry import ModelRegistry


@dataclass(frozen=True, slots=True)
class ResolvedProvider:
    provider: ProviderDefinition
    model: Model
    auth: ResolvedAuth | None
    config: ProviderConfig


class ModelResolver:
    def __init__(
        self,
        *,
        model_registry: ModelRegistry | None = None,
        auth_storage: AuthStorage | None = None,
    ) -> None:
        self._registry = model_registry or ModelRegistry()
        self._auth = auth_storage or AuthStorage.create()

    async def resolve(
        self,
        provider_id: str,
        *,
        model_id: str | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> ResolvedProvider:
        definition = self._registry.get_provider(provider_id)
        if definition is None:
            raise KeyError(f"Unknown provider: {provider_id}")
        model = self._registry.get_model(provider_id, model_id)
        resolved_credential = await self._auth.resolve_async(provider_id)
        auth = (
            ResolvedAuth(
                api_key=resolved_credential.api_key,
                source=resolved_credential.status.source or "unknown",
                label=resolved_credential.status.label,
            )
            if resolved_credential.api_key is not None
            else None
        )
        config_dict = dict(provider_config or {})
        if definition.requires_auth and auth is None and not config_dict.get("api_key"):
            raise RuntimeError(format_no_api_key_found_message(provider_id))
        stream_fn = definition.stream_factory(model, config_dict, auth)
        provider_config_obj = ProviderConfig(
            stream_fn=stream_fn,
            model=model,
            name=provider_id,
        )
        return ResolvedProvider(
            provider=definition,
            model=model,
            auth=auth,
            config=provider_config_obj,
        )
