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

from loguru import logger

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
        refile_service: Callable[[str], None],
        stream_fn: StreamFn | None = None,
        model: Model | None = None,
        identity: ProviderSessionIdentity | None = None,
    ) -> None:
        self._services = services
        self._committed_turns = committed_turns
        self._active_set = active_set
        self._emit_register_event = emit_register_event
        self._refile_service = refile_service
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
        into: ServiceRegistry | None = None,
        owner: str | None = None,
    ) -> None:
        """Register an LLM provider and refresh the active provider.

        ``into`` is the registry the backing ``provider:<name>`` service is
        written to, and ``owner`` is who that registry belongs to. Both come
        from the atom context performing the write, never from an argument the
        atom chose: this method is runtime-internal and unreachable from
        ``AtomAPI``, which offers ``register_provider(name, config)`` and
        nothing else. Omitting them is the host writing as itself.
        """

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
        target = self._services if into is None else into
        previous = self._services.get(key)
        # What this write is about to shadow *in the registry being written*,
        # which is the only thing an undo of it may put back. ``previous`` is
        # what the whole chain resolves, and that entry may live in another
        # node: writing it here would copy another context's registration into
        # this one, where it would win by being the newer write and be attributed
        # to whoever owns this table.
        held = target.own_table().get(key)
        shadowed = None if held is None else held.service
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
        previous_owner = self._owners.get(name)
        target.register(key, config, scope="session")
        self._owners[name] = owner
        try:
            self.activate()
        except BaseException:
            if shadowed is None:
                # Taking the write out is the whole undo: whatever the chain
                # resolved before, in whichever node held it, is uncovered by
                # its removal. Nothing fires for a removal, so the ledger is
                # told who the key belongs to now.
                target.unregister(key)
                self._refile_service(key)
            else:
                target.register(key, shadowed, scope="session")
            if previous is None:
                self._owners.pop(name, None)
                if self._active_name == name:
                    self._active_name = None
            else:
                self._owners[name] = previous_owner
            raise
        self._emit_register_event("provider", name, {"provider": config})

    def unregister(self, name: str) -> None:
        """Drop one provider binding, its ownership record, and its active name.

        The active ``stream_fn``/``model`` pair is deliberately left alone: it
        is what the running driver is already streaming through, and a session
        that lost its model mid-turn would be worse off than one whose model
        outlives the atom that named it.

        The active *name* is not left alone, because it is not merely read back
        — the next committed turn freezes it into the session identity, which is
        durable, serialized into the trajectory, and validated on resume. A name
        no registration backs must not reach that identity. What the session is
        left with is a stream function nothing names, which is what ``"direct"``
        already means; the next ``activate()`` re-resolves a name from whatever
        providers remain.
        """

        self._services.unregister(f"{_SERVICE_PREFIX}{name}")
        self._owners.pop(name, None)
        if self._active_name == name:
            self._active_name = None

    def note_owner(self, name: str, owner: str | None) -> None:
        """Re-file one provider under the atom whose registration now resolves.

        Unlinking a context uncovers whatever it was shadowing, and this index
        is the one account of provider ownership that no chain resolves for
        itself. Not a caller's choice of name: the session reads it off the
        context tree and passes what it found.
        """

        self._owners[name] = owner

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
        """Bind the session to its provider once a turn has been committed.

        Reachable from ``activate()`` and, independently, from the
        turn-committed bus hook, so it cannot assume anything has re-resolved
        the active provider first. An identity is minted only for a name the
        registry can still account for: ``"direct"`` when no named provider is
        active, otherwise a name present in ``configs()``. A name absent from
        ``configs()`` is refused rather than frozen — freezing it would write a
        provider registered nowhere into the trajectory, where resume validates
        it and the next ``activate()`` raises for it.

        One consequence is worth stating because it is a behaviour change and
        the freeze is one-way. A session whose provider atom departs before its
        first observed commit now freezes ``"direct"`` where it previously
        froze the atom's name, so resuming that trajectory with the atom back
        in the composition fails the install. Every alternative in that state
        also fails — the departed name would fail resume validation instead —
        so the trade is deliberate, not an oversight.
        """

        turns = self._committed_turns()
        if self._identity is not None or not turns:
            return
        providers = self.configs()
        first_model_id = next(
            (turn.meta.model_id for turn in turns if turn.meta.model_id),
            None,
        )
        if not providers:
            if self._active_name is not None:
                logger.warning(
                    "refusing to freeze provider identity: {!r} is still the "
                    "active provider but is registered nowhere; a later "
                    "activate() will raise for it",
                    self._active_name,
                )
                return
            if self.model is None:
                return
            active_set = self._active_set()
            self.set_identity(
                ProviderSessionIdentity(
                    name="direct",
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
