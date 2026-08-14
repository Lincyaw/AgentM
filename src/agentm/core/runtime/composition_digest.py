"""Composition digest — what a session actually holds, as one comparable value.

``ActiveSetFingerprint`` digests the *plan*: which atoms a composition asked
for, at which versions, with which config.  Nothing digests the *result*.  This
module does, so that "install an atom, uninstall it, and check the session came
back" becomes a single equality rather than a hand-written list of assertions
that grows a hole every time a new kind of registration appears.

The digest is a read.  It must never change what it measures, which rules out
the obvious accessors: ``ProviderRegistry.get`` re-resolves the active
provider, and ``session_identity`` freezes an identity into the trajectory.
Both are read *around* here rather than through.

Several of the tables read here have no public reader — the bus's
subscriptions, the codec's registered sources, the provider registry's active
name.  They are read directly rather than by widening ``core/abi`` with
accessors that would exist for one diagnostic consumer: this module sits in the
layer that already force-clears the bus at shutdown, and the composability
target shape means to replace those tables rather than publish them.

What is covered, and in what order
----------------------------------

Order is part of the value wherever a read can observe it, and dropped where no
read can.  Tools are advertised to the model in list order; context policies
transform in list order; bus handlers dispatch in list order per channel; bus
observers fire in list order.  Those stay ordered.  Services, trigger
renderers, trigger codecs, providers and termination causes are addressed by
key and nothing distinguishes two insertion orders, so they are sorted.

What this digest deliberately excludes
--------------------------------------

Each exclusion is a claim that the excluded thing cannot distinguish two
sessions that are otherwise the same.  A claim that turns out to be false is a
bug in this module, not a licence to widen the list.

* ``EventBus._next_seq`` and ``TriggerQueue._seq``.  Both are monotonic
  counters whose only job is to break ties within one priority band.  Their
  effect is entirely captured by the *position* of each subscription in the
  per-channel list, which is digested.  The absolute value grows with every
  subscription ever made, so including it would make every digest differ from
  every other digest.

* ``id()`` of any object.  Freshly constructed objects get fresh addresses, so
  an atom reinstalled from the same source would never compare equal.  ``id()``
  appears below only as a *lookup key* into the install ledger, which keys
  ownership that way, and never as a digested value.

* Session id, root/parent session id, turn ids, timestamps, and dispatch ids.
  These identify an occurrence, not a composition.

* ``EventBus._frozen_clear``.  A lifecycle latch set once by ``start()`` and
  cleared once by shutdown; no install path writes it (the atom-facing bus
  facade refuses ``freeze_clear`` outright).

* ``SessionRuntime.system``, ``_max_turns``, ``_max_tool_calls``, ``_thinking``.
  Assigned in ``__init__`` and never again — an atom influences the system
  prompt by answering ``BeforeRunEvent``, which is a subscription, and that is
  digested.

* Anything already emitted.  Every ``register_*`` synchronously emits an
  ``ApiRegisterEvent``; an emission has left the system by the time it is
  observable and there is no un-emit, so it is outside any inverse and outside
  this digest by construction.  What a *handler* did with the event is inside,
  because a handler can only act through the same writes digested here.

* A trigger in flight.  That is turn state, not composition state; both digests
  in a revertibility check are taken with no turn running.

* The difference between two objects that share a name.  Handlers, policies and
  service values are digested by qualname (see ``_identity``), so swapping one
  closure for another defined in the same function, or one service value for a
  different instance of the same class, is invisible here.  This is the known
  coarseness of the equivalence rather than a claim that such a swap is safe.

What is deliberately *not* excluded is the one residue the runtime creates on
purpose: uninstalling an atom leaves its trigger codecs registered, so a
committed turn naming that source stays decodable.  Absorbing that into the
digest would make every future codec leak invisible too, so it stays visible
and a caller declares it — see ``agentm.testing.assert_revertible``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from agentm.core.abi.services import ServiceScope
from agentm.core.runtime.session_core import SessionRuntime


@dataclass(frozen=True, slots=True)
class AtomEntry:
    """One installed atom, as the ledger accounts for it."""

    module_path: str
    name: str | None
    runtime: bool


@dataclass(frozen=True, slots=True)
class ToolEntry:
    """One advertised tool and the atom that put it there."""

    name: str
    owner: str | None


@dataclass(frozen=True, slots=True)
class PolicyEntry:
    """One context policy, at the priority its position was sorted by."""

    priority: int | None
    policy: str
    owner: str | None


@dataclass(frozen=True, slots=True)
class SourceEntry:
    """One trigger source binding — a renderer or a codec."""

    source: str
    implementation: str
    owner: str | None


@dataclass(frozen=True, slots=True)
class ServiceEntry:
    """One service key with its scope, its owner, and what it resolves to."""

    key: str
    scope: ServiceScope | None
    owner: str | None
    value: str


@dataclass(frozen=True, slots=True)
class SubscriptionEntry:
    """One bus subscription, in the position it dispatches from."""

    channel: str
    priority: int
    owner: str | None
    handler: str


@dataclass(frozen=True, slots=True)
class ObserverEntry:
    """One bus observer, in the position it is notified from."""

    observer: str
    owner: str | None


@dataclass(frozen=True, slots=True)
class ProviderEntry:
    """One registered provider and the model it names."""

    name: str
    owner: str | None
    model_id: str


@dataclass(frozen=True, slots=True)
class CompositionDigest:
    """Everything a session holds that an installation could have written."""

    atoms: tuple[AtomEntry, ...]
    composition_specs: tuple[str, ...]
    tools: tuple[ToolEntry, ...]
    context_policies: tuple[PolicyEntry, ...]
    trigger_renderers: tuple[SourceEntry, ...]
    trigger_codecs: tuple[SourceEntry, ...]
    termination_causes: tuple[str, ...]
    services: tuple[ServiceEntry, ...]
    subscriptions: tuple[SubscriptionEntry, ...]
    observers: tuple[ObserverEntry, ...]
    providers: tuple[ProviderEntry, ...]
    active_provider: str | None
    active_model_id: str | None
    active_stream_fn: str | None
    provider_identity: str | None
    cleanup_callbacks: tuple[str, ...]
    pending_atom_installs: tuple[str, ...]
    queued_triggers: int
    pending_background_work: int
    background_tasks: tuple[str, ...]


def composition_digest(session: SessionRuntime) -> CompositionDigest:
    """Digest what ``session`` currently holds, without disturbing any of it."""

    # ``capture`` is the ledger's own snapshot: the same ownership maps the
    # install-failure rollback restores, read here rather than restored. Its
    # tool and policy maps are keyed by ``id()``, which is why the lookups
    # below take ids of live objects — an id is a key into this snapshot and
    # never reaches the digest.
    ledger = session._extensions.capture()
    bus = session.bus
    codec = session.codec
    providers = session._providers

    provider_configs = providers.configs()
    provider_owners = providers.owners()
    # Read the registry's fields rather than ``get()``/``session_identity()``:
    # both of those re-resolve, and one of them freezes an identity into the
    # session. A digest that mutates its subject cannot witness anything.
    identity = providers._identity
    atom_names = {path: name for name, path in ledger.atom_names.items()}

    return CompositionDigest(
        atoms=tuple(
            AtomEntry(
                module_path=module_path,
                name=atom_names.get(module_path),
                runtime=module_path in ledger.runtime_module_paths,
            )
            for module_path in ledger.module_paths
        ),
        # The replayable specs are what a child or a fork is rebuilt from. They
        # track the installed set but are dropped by a separate line in
        # ``forget``, so one left behind would resurrect a departed atom in
        # every session spawned afterwards.
        composition_specs=tuple(spec.module_path for spec in ledger.specs),
        tools=tuple(
            ToolEntry(name=tool.name, owner=ledger.tool_owners.get(id(tool)))
            for tool in session.tools
        ),
        context_policies=tuple(
            PolicyEntry(
                priority=ledger.context_policy_priorities.get(id(policy)),
                policy=_identity(policy),
                owner=ledger.context_policy_owners.get(id(policy)),
            )
            for policy in session.context_policies
        ),
        trigger_renderers=tuple(
            SourceEntry(
                source=source,
                implementation=_identity(session.trigger_renderers[source]),
                owner=ledger.trigger_renderer_owners.get(source),
            )
            for source in sorted(session.trigger_renderers)
        ),
        trigger_codecs=tuple(
            SourceEntry(
                source=source,
                implementation=_identity(codec._trigger_codecs[source]),
                owner=ledger.trigger_codec_owners.get(source),
            )
            for source in sorted(codec._trigger_codecs)
        ),
        termination_causes=tuple(sorted(codec._cause_types)),
        services=tuple(
            ServiceEntry(
                key=key,
                scope=session.services.scope(key),
                owner=ledger.service_owners.get(key),
                value=_identity(session.services.get(key)),
            )
            for key in sorted(session.services.names())
        ),
        # Per-channel list order is dispatch order; the channels themselves are
        # sorted because no emission spans two of them.
        subscriptions=tuple(
            SubscriptionEntry(
                channel=channel,
                priority=subscription.priority,
                owner=subscription.owner,
                handler=_identity(subscription.handler),
            )
            for channel in sorted(bus._handlers)
            for subscription in bus._handlers[channel]
        ),
        observers=tuple(
            ObserverEntry(
                observer=_identity(record.observer),
                owner=record.owner,
            )
            for record in bus._observers
        ),
        providers=tuple(
            ProviderEntry(
                name=name,
                owner=provider_owners.get(name),
                model_id=provider_configs[name].model.id,
            )
            for name in sorted(provider_configs)
        ),
        active_provider=providers._active_name,
        active_model_id=None if providers.model is None else providers.model.id,
        active_stream_fn=(
            None if providers.stream_fn is None else _identity(providers.stream_fn)
        ),
        provider_identity=(
            None if identity is None else f"{identity.name}/{identity.model_id}"
        ),
        cleanup_callbacks=tuple(
            _identity(callback) for callback in session._cleanup_callbacks
        ),
        pending_atom_installs=tuple(
            install.atom_name for install in session._pending_atom_installs
        ),
        queued_triggers=session.triggers._queue.qsize(),
        pending_background_work=session.triggers._pending_work,
        background_tasks=_background_tasks(),
    )


def _identity(value: object) -> str:
    """A stable, printable name for something the digest can only name.

    Handlers, observers, cleanup callbacks, policies, codecs and service values
    are arbitrary objects. Their addresses change on every reinstall, so the
    digest names them instead: the qualname where there is one, the type's
    qualname otherwise. Two distinct closures defined in the same function are
    indistinguishable here; that is the known coarseness of this equivalence.
    """

    if value is None:
        return "none"
    # Only some callables carry ``__qualname__`` -- a plain function and a
    # bound method do, an instance with ``__call__`` does not -- and no typed
    # accessor spans both.
    name = getattr(value, "__qualname__", None)  # code-health: ignore[AM021]
    if type(name) is str:
        return name
    return type(value).__qualname__


def _background_tasks() -> tuple[str, ...]:
    """Coroutine names of the tasks still pending on the running loop.

    A background task is the archetypal write the install ledger cannot see: an
    atom that starts one keeps running after everything it registered has been
    detached. Names are sorted because two tasks started in either order are the
    same residue.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        logger.debug(
            "composition digest taken outside a running event loop; "
            "background tasks are not covered by this digest"
        )
        return ()
    current = asyncio.current_task()
    return tuple(
        sorted(
            _identity(task.get_coro())
            for task in asyncio.all_tasks()
            if task is not current and not task.done()
        )
    )


__all__ = [
    "AtomEntry",
    "CompositionDigest",
    "ObserverEntry",
    "PolicyEntry",
    "ProviderEntry",
    "ServiceEntry",
    "SourceEntry",
    "SubscriptionEntry",
    "ToolEntry",
    "composition_digest",
]
