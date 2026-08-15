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

Order is part of the value wherever a read can observe it, and dropped where it
is not what the read is about.  Tools are advertised to the model in list
order; context policies transform in list order; bus handlers dispatch in list
order per channel; bus observers fire in list order.  Those stay ordered.
Services, trigger renderers, trigger codecs, providers and termination causes
are addressed by key, so they are sorted by key here.

Sorting those is a deliberate narrowing rather than a claim that nothing can
observe their order: ``ServiceRegistry.names()`` and
``ProviderRegistry.configs()`` are both public and both hand out insertion
order, and the second reaches a third-party resolver.  What the digest says is
that the same keys resolve to the same things, not that they were written in
the same sequence — an atom that reverts cleanly and leaves a differently
ordered ``names()`` is not a leak, and treating it as one would make every
revertibility check depend on install order.

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

* The difference between two opaque objects that share a name.  Handlers,
  observers, policies and codecs are digested by qualname (see ``_identity``),
  so swapping one closure for another defined in the same function is invisible
  here.  This is the known coarseness of the equivalence rather than a claim
  that such a swap is safe.  Service *values* are the exception: they are
  digested structurally (see ``_service_value``), because a service is data as
  often as it is behaviour and a session boundary like ``tool_allowlist`` is
  read for its content every turn.

* The attributes of an object that is neither a dataclass nor a container.  The
  structural walk opens the JSON-ish types and ``dataclasses.fields``; a class
  that satisfies a Protocol by holding mutable attributes is still digested by
  its type name alone, so a service that mutates itself in place reads here as
  unchanged.  Naming that class is the honest limit of a walk that must not run
  user code to look inside.

* What a closure captured.  A plain function is named, not opened.  Opening one
  would make the digest depend on whatever happened to be in scope where the
  function was defined — a session id, a temporary path — and two sessions
  built the same way would stop comparing equal, which is the comparison a
  revertibility check across two probe sessions rests on.  Nothing a real
  composition registers needs it: both shipped providers build their
  ``StreamFn`` as a ``@dataclass(slots=True)``, so the credential that made
  this worth looking at is opened by the dataclass branch.

What is deliberately *not* excluded is the one residue the runtime creates on
purpose: uninstalling an atom leaves its trigger codecs registered, so a
committed turn naming that source stays decodable.  Absorbing that into the
digest would make every future codec leak invisible too, so it stays visible
and a caller declares it — see ``agentm.testing.assert_revertible``.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Final, cast

from loguru import logger

from agentm.core.abi.effects import EffectLog
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
    """One service key with its scope, its owner, and what it resolves to.

    ``value`` is the structural digest of the registered object, not the object
    — see ``_service_value`` for what it can and cannot tell apart.
    """

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
    """One registered provider and the model it names.

    Deliberately not the whole ``ProviderConfig``: the registry keeps its
    configs in the service registry, so the same object is already digested for
    content as ``service:provider:<name>``, and restating it here would report
    one change as two.  What is *only* here is the ownership, which lives in
    the registry's own index rather than in the service.
    """

    name: str
    owner: str | None
    model_id: str


@dataclass(frozen=True, slots=True)
class EffectRecord:
    """One write an atom context recorded that no table of the runtime holds.

    Liveness, not heldness: it is read off the *linked* contexts, so an atom
    that has been unlinked contributes nothing here however much its own
    context object still holds. A record that survives an uninstall therefore
    means the write survived it too.

    Which is why a ``retain`` reason is not how deliberate residue shows up
    after the fact. It reads here while the atom is installed -- what the
    session is holding, and on whose word -- and it leaves with its context,
    because an uninstalled atom holds nothing. The residue itself stays visible
    where it actually landed: the one retained write the runtime makes is a
    trigger codec, and an uninstalled atom's codec is still a
    ``trigger_codecs`` entry, still attributed to the atom that registered it.
    Two accounts of one fact, each true of a different moment.

    ``settled`` is False for an async body that was queued and has not run; one
    of those is a write that has not happened yet, which is exactly the kind of
    residue a caller comparing two digests needs to see.
    """

    owner: str
    provides: str
    retain: str
    settled: bool


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
    service_write_observer: str | None
    effects: tuple[EffectRecord, ...]
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
    # Ownership is read off the context tree, not off the ledger: a thing
    # belongs to the context whose table it is in, and that is the only account
    # a detach acts on. The ledger keeps a parallel one, which the attribution
    # tests compare against this rather than this depending on.
    owners = session.ownership()
    linked = session.linked_contexts()

    provider_configs = providers.configs()
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
            ToolEntry(name=tool.name, owner=owners.tool(tool)) for tool in session.tools
        ),
        context_policies=tuple(
            PolicyEntry(
                priority=row.priority,
                policy=_identity(row.policy),
                owner=owners.policy(row.policy),
            )
            for row in session.policy_rows()
        ),
        trigger_renderers=tuple(
            SourceEntry(
                source=source,
                implementation=_identity(session.trigger_renderers[source]),
                owner=owners.renderer(source),
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
                owner=owners.service(key),
                value=_service_value(session.services.get(key)),
            )
            for key in sorted(session.services.names())
        ),
        # The observer is what attributes every later service write. It is
        # public on the registry an atom holds, so an atom can replace it or
        # clear it, and every write after that would be attributed to nobody
        # while the digest reported nothing amiss.
        service_write_observer=(
            None
            if session.services._write_observer is None
            else _identity(session.services._write_observer)
        ),
        effects=tuple(
            record
            for context in linked
            for record in _effect_records(context.module_path, context.effects)
        ),
        # Per-channel list order is dispatch order; the channels themselves are
        # sorted because no emission spans two of them.
        # Read through the bus's merged view rather than its own table: what
        # dispatches on a channel is the session's own subscriptions plus every
        # linked context's, and only the merge is in dispatch order.
        subscriptions=tuple(
            SubscriptionEntry(
                channel=channel,
                priority=subscription.priority,
                owner=subscription.owner,
                handler=_identity(subscription.handler),
            )
            for channel in bus.channels()
            for subscription in bus.subscriptions(channel)
        ),
        observers=tuple(
            ObserverEntry(
                observer=_identity(record.observer),
                owner=record.owner,
            )
            for record in bus.all_observers()
        ),
        providers=tuple(
            ProviderEntry(
                name=name,
                owner=owners.service(f"provider:{name}"),
                model_id=provider_configs[name].model.id,
            )
            for name in sorted(provider_configs)
        ),
        active_provider=providers._active_name,
        active_model_id=None if providers.model is None else providers.model.id,
        # Digested for content rather than named: this and the backing
        # ``service:provider:<name>`` value are the only two places a provider
        # rebuilt with the same name and model but a different base URL or
        # credential can show up, because both of those are fields of the
        # stream function and nothing else here reads them.
        active_stream_fn=(
            None if providers.stream_fn is None else _service_value(providers.stream_fn)
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


def _effect_records(owner: str, log: EffectLog) -> list[EffectRecord]:
    """One context's recorded writes: what ran, then what is still queued."""

    records = [
        EffectRecord(
            owner=owner,
            provides=entry.provides,
            retain=entry.retain,
            settled=True,
        )
        for entry in log.entries
    ]
    records.extend(
        EffectRecord(
            owner=owner,
            provides=handle.provides,
            retain=handle.retain,
            settled=False,
        )
        for handle in log.unsettled
    )
    return records


def _identity(value: object) -> str:
    """A stable, printable name for something the digest can only name.

    Handlers, observers, cleanup callbacks, policies and codecs are arbitrary
    objects, as is a service value the structural walk cannot open. Their
    addresses change on every reinstall, so the digest names them instead: the
    qualname where there is one, the type's qualname otherwise. Two distinct
    closures defined in the same function are indistinguishable here; that is
    the known coarseness of this equivalence.
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


_VALUE_DEPTH: Final[int] = 8
"""How far ``_service_value`` walks into a nested value before giving up.

A bound depth is what makes the walk safe on a value that contains itself, and
eight levels is past anything a service holds in practice.
"""

_UNSET: Final[object] = object()
"""Stands for a dataclass field that was declared and never assigned."""


def _service_value(value: object) -> str:
    """A content-sensitive digest of one registered service.

    Naming a service by its type, the way ``_identity`` names a handler, makes
    every data service look alike: ``"v1"`` and ``"v2"`` are both ``str``, and
    ``tool_allowlist`` — a boundary the driver reads every turn — would compare
    equal however it was rewritten.  Values built out of the JSON-ish types are
    therefore walked structurally, as are ``dataclasses.fields``, and
    everything else falls back to the type name.

    What the walk still cannot see, stated so a caller does not read more into
    an equality than it carries: a class that satisfies a Protocol by holding
    mutable attributes has no fields to enumerate, so it digests by its type
    name alone and a service that mutates itself in place reads as unchanged.
    Nor can it see inside a callable — a closure, a bound method, or an
    instance with ``__call__`` — because what it holds is reachable only by
    naming attributes the digest has no schema for, or by opening cells whose
    contents are as often session-specific as they are compositional.

    The walk is hashed rather than kept verbatim for two reasons: a service can
    hold a credential, and a failure message that printed one would leak it into
    every log that captured the test output; and a value's ``repr`` has no
    bound, while a digest entry has to stay readable next to a hundred others.
    The kind is kept in front of the hash so a difference still says what sort
    of thing changed.

    Structural equality here follows Python's: ordered containers keep their
    order, sets and mappings do not, and a list never digests as the tuple with
    the same elements.  Two distinct opaque instances of one class still digest
    alike, nested inside a container as much as at the top level.
    """

    kind = "none" if value is None else type(value).__qualname__
    encoded = _encode(value, _VALUE_DEPTH).encode("utf-8", errors="surrogatepass")
    return f"{kind}/{hashlib.sha256(encoded).hexdigest()[:16]}"


def _encode(value: object, depth: int) -> str:
    """Canonical text for ``value``: structural where it can be, named where not."""

    if value is None:
        return "none"
    kind = type(value)
    # Exact types rather than ``isinstance``: a subclass may carry state this
    # walk would not see, and naming it is the honest answer for one.
    if kind is bool or kind is int or kind is float or kind is str or kind is bytes:
        return f"{kind.__name__}:{value!r}"
    if depth <= 0:
        # Deeper than the walk goes, or a value that contains itself.
        return f"deep:{_identity(value)}"
    if kind is tuple or kind is list:
        items = cast("Sequence[object]", value)
        return (
            f"{kind.__name__}:[{','.join(_encode(item, depth - 1) for item in items)}]"
        )
    if kind is set or kind is frozenset:
        members = cast("AbstractSet[object]", value)
        encoded = sorted(_encode(member, depth - 1) for member in members)
        return f"{kind.__name__}:{{{','.join(encoded)}}}"
    if kind is dict or kind is MappingProxyType:
        mapping = cast("Mapping[object, object]", value)
        pairs = sorted(
            f"{_encode(key, depth - 1)}={_encode(item, depth - 1)}"
            for key, item in mapping.items()
        )
        return f"{kind.__name__}:{{{','.join(pairs)}}}"
    if is_dataclass(kind) and "__dataclass_fields__" in kind.__dict__:
        # Where the services in a real composition actually live. Walking only
        # the JSON-ish types left this branch unreached by every service a
        # session holds, so the digest was structural in principle and opaque
        # in practice. Declaration order is kept because it is fixed by the
        # class rather than by the write, so it cannot make two equal values
        # digest apart.
        #
        # The ``__dict__`` check is what keeps this branch to the exact type,
        # the way every branch above is: ``is_dataclass`` alone follows the
        # MRO and answers yes for an undecorated subclass of a dataclass, and
        # ``fields()`` would then enumerate the base's fields and silently skip
        # whatever state the subclass added, reporting a changed value as
        # unchanged. Such a subclass is named instead, which is the same answer
        # the walk gives every other type it cannot open. It stays paired with
        # ``is_dataclass`` because that is what says ``fields()`` may be called.
        attributes = ",".join(
            f"{spec.name}={_encode_attribute(value, spec.name, depth - 1)}"
            for spec in fields(kind)
        )
        return f"{kind.__qualname__}:({attributes})"
    return f"opaque:{_identity(value)}"


def _encode_attribute(value: object, name: str, depth: int) -> str:
    """One dataclass field, or the fact that it was never assigned.

    A slotted dataclass can be constructed with a field left unset, and reading
    one raises rather than returning a default; the absence is stable, so it is
    digested as itself instead of failing the read.
    """

    item = getattr(value, name, _UNSET)  # code-health: ignore[AM021]
    return "unset" if item is _UNSET else _encode(item, depth)


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
    "EffectRecord",
    "ObserverEntry",
    "PolicyEntry",
    "ProviderEntry",
    "ServiceEntry",
    "SourceEntry",
    "SubscriptionEntry",
    "ToolEntry",
    "composition_digest",
]
