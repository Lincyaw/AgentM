# code-health: ignore-file[AM025] -- provider configs arrive from atoms as plain service values
"""Provider registry — named LLM providers and the binding they resolve to.

A session talks to one model at a time.  Which one is decided here: atoms
register named ``ProviderConfig`` values as session services, an optional
``ProviderResolver`` picks between them, and the active choice becomes the
session's ``stream_fn``/``model`` pair.

Once the session commits its first turn the choice is frozen into a
``ProviderSessionIdentity``, so a trajectory cannot silently change model
half-way through.  Everything after the freeze is checked against that
identity rather than re-resolved.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from agentm.core.abi.catalog import ActiveSetFingerprint
from agentm.core.abi.provider import (
    ProviderConfig,
    ProviderResolver,
    ProviderSessionIdentity,
)
from agentm.core.abi.roles import (
    PROVIDER_RESOLVER_SERVICE,
    PROVIDER_SESSION_IDENTITY,
)
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.stream import Model, StreamFn
from agentm.core.abi.trajectory import Turn

_SERVICE_PREFIX = "provider:"


@dataclass(frozen=True, slots=True)
class ProviderSnapshot:
    """Provider state captured before one atom installation."""

    stream_fn: StreamFn | None
    model: Model | None
    active_provider_name: str | None
    identity: ProviderSessionIdentity | None
    owners: dict[str, str | None]


class ProviderRegistry:
    """The session's provider registrations plus the active stream/model."""

    def __init__(
        self,
        *,
        services: ServiceRegistry,
        committed_turns: Callable[[], Sequence[Turn]],
        active_set: Callable[[], ActiveSetFingerprint | None],
        emit_register_event: Callable[[str, str, dict[str, object]], None],
        stream_fn: StreamFn | None = None,
        model: Model | None = None,
        identity: ProviderSessionIdentity | None = None,
    ) -> None:
        self._services = services
        self._committed_turns = committed_turns
        self._active_set = active_set
        self._emit_register_event = emit_register_event
        self.stream_fn = stream_fn
        self.model = model
        self._active_name: str | None = None
        self._owners: dict[str, str | None] = {}
        inherited = services.get_role(PROVIDER_SESSION_IDENTITY)
        self._identity = identity if identity is not None else inherited
        if self._identity is not None:
            services.bind(PROVIDER_SESSION_IDENTITY, self._identity, replace=True)

    # --- Registration ---

    def register(
        self,
        name: str,
        config: ProviderConfig,
        *,
        replace: bool = False,
    ) -> None:
        """Register an LLM provider and refresh the active provider."""
        if not isinstance(name, str) or not name:
            raise ValueError("provider registry name must be a non-empty string")
        if not isinstance(config, ProviderConfig):
            raise TypeError("provider config must be ProviderConfig")
        if config.name != name:
            raise ValueError(
                f"provider registry name {name!r} does not match "
                f"ProviderConfig.name {config.name!r}"
            )
        key = f"{_SERVICE_PREFIX}{name}"
        previous = self._services.get(key)
        if previous is not None and not replace:
            raise ValueError(
                f"provider {name!r} is already registered in this session; give "
                "each provider a unique name (config['name']) or pass replace=True"
            )
        prospective = self.configs()
        prospective[name] = config
        if self._identity is None:
            self.resolve_name(prospective)
        elif (
            self._identity.name == name
            and self._identity.model_id is not None
            and config.model.id != self._identity.model_id
        ):
            raise RuntimeError(
                "cannot replace the session-bound provider with model "
                f"{config.model.id!r}; expected {self._identity.model_id!r}"
            )
        from agentm.core.runtime.extension import current_installing_extension

        previous_owner = self._owners.get(name)
        self._services.register(key, config, scope="session")
        self._owners[name] = current_installing_extension() or None
        try:
            self.activate()
        except BaseException:
            if previous is None:
                self._services.unregister(key)
                self._owners.pop(name, None)
            else:
                self._services.register(key, previous, scope="session")
                self._owners[name] = previous_owner
            raise
        self._emit_register_event("provider", name, {"provider": config})

    def has(self, name: str) -> bool:
        return self._services.get(f"{_SERVICE_PREFIX}{name}") is not None

    def get(self, name: str | None = None) -> ProviderConfig | None:
        if name is None:
            self.activate()
        provider_name = name or self._active_name
        if provider_name is None:
            return None
        provider = self._services.get(f"{_SERVICE_PREFIX}{provider_name}")
        return provider if isinstance(provider, ProviderConfig) else None

    def names(self) -> list[str]:
        return sorted(
            name[len(_SERVICE_PREFIX) :]
            for name in self._services.names()
            if name.startswith(_SERVICE_PREFIX)
        )

    def configs(self) -> dict[str, ProviderConfig]:
        providers: dict[str, ProviderConfig] = {}
        for service_name in self._services.names():
            if not service_name.startswith(_SERVICE_PREFIX):
                continue
            provider = self._services.get(service_name)
            if isinstance(provider, ProviderConfig):
                providers[service_name[len(_SERVICE_PREFIX) :]] = provider
        return providers

    def owners(self) -> dict[str, str | None]:
        """Which atom registered each provider, by provider name."""

        return dict(self._owners)

    # --- Selection ---

    def _resolver(self) -> ProviderResolver | None:
        candidate = self._services.get(PROVIDER_RESOLVER_SERVICE)
        return candidate if isinstance(candidate, ProviderResolver) else None

    def resolve_name(self, providers: dict[str, ProviderConfig]) -> str:
        if not providers:
            raise LookupError("cannot resolve an empty provider registry")
        resolver = self._resolver()
        if resolver is not None:
            selected = resolver.resolve_provider(providers)
            if selected is None:
                raise LookupError(
                    "provider resolver returned None for a non-empty registry"
                )
            if selected not in providers:
                raise LookupError(
                    f"provider resolver selected unregistered provider {selected!r}"
                )
            return selected
        if len(providers) == 1:
            return next(iter(providers))
        raise RuntimeError(
            "multiple providers are registered; configure a ProviderResolver "
            "instead of relying on registration order"
        )

    def activate(self) -> None:
        providers = self.configs()
        if not providers:
            if (
                self._identity is not None
                and self.model is not None
                and self._identity.model_id is not None
                and self.model.id != self._identity.model_id
            ):
                raise RuntimeError(
                    "cannot activate provider: session is bound to model "
                    f"{self._identity.model_id!r}, got {self.model.id!r}"
                )
            return
        self.freeze_after_commits()
        if self._identity is not None:
            provider = providers.get(self._identity.name)
            if provider is None:
                if not self._committed_turns():
                    return
                raise RuntimeError(
                    "cannot activate provider: session is bound to provider "
                    f"{self._identity.name!r}, but it is not registered"
                )
            model_id = provider.model.id
            if (
                self._identity.model_id is not None
                and model_id != self._identity.model_id
            ):
                raise RuntimeError(
                    "cannot activate provider: session is bound to model "
                    f"{self._identity.model_id!r}, got {model_id!r}"
                )
            self._active_name = self._identity.name
            self.stream_fn = provider.stream_fn
            self.model = provider.model
            return
        selected = self.resolve_name(providers)
        if selected is None:
            return
        provider = providers[selected]
        self._active_name = selected
        self.stream_fn = provider.stream_fn
        self.model = provider.model

    # --- Identity freezing ---

    def on_turn_committed(self, _: object) -> None:
        """Bus hook: a committed turn locks the session to its provider."""

        self.freeze_after_commits()

    def freeze_after_commits(self) -> None:
        turns = self._committed_turns()
        if self._identity is not None or not turns:
            return
        providers = self.configs()
        first_model_id = next(
            (turn.meta.model_id for turn in turns if turn.meta.model_id),
            None,
        )
        if not providers:
            if self.model is None:
                return
            active_set = self._active_set()
            self.set_identity(
                ProviderSessionIdentity(
                    name=self._active_name or "direct",
                    model_id=first_model_id or self.model.id,
                    active_set_digest=(
                        active_set.digest if active_set is not None else None
                    ),
                    frozen_after_turn_index=turns[0].index,
                )
            )
            return
        provider_name = self._active_name
        if provider_name is None or provider_name not in providers:
            raise RuntimeError(
                "cannot freeze provider identity: committed turns have no "
                "active registered provider"
            )
        provider = providers[provider_name]
        if first_model_id is not None and provider.model.id != first_model_id:
            raise RuntimeError(
                "cannot freeze provider identity: committed model "
                f"{first_model_id!r} does not match selected provider model "
                f"{provider.model.id!r}"
            )
        active_set = self._active_set()
        self.set_identity(
            ProviderSessionIdentity(
                name=provider_name,
                model_id=first_model_id or provider.model.id,
                active_set_digest=active_set.digest if active_set is not None else None,
                frozen_after_turn_index=turns[0].index,
            )
        )

    def set_identity(self, identity: ProviderSessionIdentity) -> None:
        self._identity = identity
        self._services.bind(PROVIDER_SESSION_IDENTITY, identity, replace=True)

    def session_identity(self) -> ProviderSessionIdentity | None:
        """Return the provider/model identity bound to this session, if known."""

        self.freeze_after_commits()
        if self._identity is not None:
            active_set = self._active_set()
            if active_set is not None:
                bound_digest = self._identity.active_set_digest
                if bound_digest is None:
                    self.set_identity(
                        replace(self._identity, active_set_digest=active_set.digest)
                    )
                elif bound_digest != active_set.digest:
                    raise RuntimeError(
                        "provider identity active set does not match the session: "
                        f"{bound_digest} != {active_set.digest}"
                    )
            return self._identity
        if self._active_name is None:
            self.activate()
        if self.model is None:
            return None
        active_set = self._active_set()
        return ProviderSessionIdentity(
            name=self._active_name or "direct",
            model_id=self.model.id,
            active_set_digest=active_set.digest if active_set is not None else None,
        )

    # --- Install rollback ---

    def capture(self) -> ProviderSnapshot:
        return ProviderSnapshot(
            stream_fn=self.stream_fn,
            model=self.model,
            active_provider_name=self._active_name,
            identity=self._identity,
            owners=dict(self._owners),
        )

    def restore(self, snapshot: ProviderSnapshot) -> None:
        self.stream_fn = snapshot.stream_fn
        self.model = snapshot.model
        self._active_name = snapshot.active_provider_name
        self._identity = snapshot.identity
        self._owners = dict(snapshot.owners)


__all__ = ["ProviderRegistry", "ProviderSnapshot"]
