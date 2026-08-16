# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Typed service registry — scoped dependency injection for atoms.

Services are the atom-to-atom communication channel when atoms cannot import
each other directly. Each service can also declare whether it is local to one
session or inherited by child sessions.

Well-known boundaries are described once by a ``ServiceRole`` (key + protocol
+ scope) declared in ``agentm.core.abi.roles``; ``bind``/``get_role``/
``require_role`` consume the descriptor so call sites never restate scope or
protocol.

A registry is one node of a context tree, not a flat table.  An atom writes
into its own registry and the session sees the write because that registry is
*linked* into the session's; detaching the atom unlinks it, and no write the
atom makes afterwards reaches anybody.  Reading resolves the other way — own
table first, then the chain — so an atom that has been unlinked still reads
everything the session holds, which is what lets a superseded atom find its
successor.

Which value a key resolves to is decided by write order, not by link order:
every write takes the next number from one counter, and the highest-numbered
write of a key wins.  Last-writer-wins is what a single flat table gave, and
keeping it means linking a registry cannot change what a key already resolved
to except by writing it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeVar, cast, overload

T = TypeVar("T")
ServiceScope = Literal["session", "tree"]
"""Whether a service is local to one session or inherited by child sessions."""
_INHERITED_SCOPES: Final[frozenset[ServiceScope]] = frozenset({"tree"})

_WRITE_ORDER = itertools.count()
"""One monotonic counter across every registry in the process.

It decides which of two writes to the same key across two linked registries is
the later one, and it orders ``names()``.  Process-wide rather than per-tree
because a registry is linked after it is written to, so a counter owned by the
tree would not have been reachable when the write happened.  Only differences
between two numbers are ever read; the absolute value is never digested,
persisted, or compared across processes.
"""


class WriteObserver(Protocol):
    """Callback fired after every write into a registry.

    ``role_bind`` separates a deliberate role binding from a plain
    ``register``: both are writes the owning session must attribute, only the
    first is a boundary decision worth announcing.
    """

    def __call__(
        self,
        key: str,
        service: object,
        scope: ServiceScope,
        *,
        role_bind: bool,
    ) -> None: ...


class ServiceNotFound(KeyError):
    """Raised when a required service is not registered."""


class ServiceTypeMismatch(TypeError):
    """Raised when a registered service doesn't match the expected protocol."""


@dataclass(frozen=True, slots=True)
class ServiceRole[T]:
    """Single-source descriptor for one well-known service boundary.

    ``scope`` is the canonical default; ``bind`` accepts an explicit override
    for the rare composition that needs a different lifetime.
    """

    key: str
    protocol: type[T] | None = None
    scope: ServiceScope = "tree"

    @property
    def capability(self) -> str:
        """This role as a manifest ``requires``/``registers`` entry.

        The role owns the service key, so an atom that depends on or provides
        the boundary declares it from here instead of restating the key as a
        ``"service:..."`` literal that nothing keeps in sync.
        """

        return f"service:{self.key}"


@dataclass(frozen=True, slots=True)
class ServiceLayer:
    """A write that decorates a key rather than replacing what it holds.

    The alternative, and what several atoms did before this existed, is to read
    the key, wrap what is there, and write the wrapper back.  That captures the
    chain in a closure: the order is install order, and taking out a link in
    the middle is unrepresentable -- the outer wrapper goes on calling the
    inner one long after the atom that registered it has been detached, with
    its context unlinked and its services gone.

    A layer states the decoration instead of performing it.  ``build`` takes
    whatever the key resolves to underneath and returns the decorated value,
    so the chain is *derived* at every read: it is the fold of the layers that
    are present right now, over the base that is there right now.  Removing an
    atom removes its layer, and the next read folds without it.

    One key, one writer per layer, so layers are set-shaped -- which is what
    makes them commute.  They fold by ``ServiceEntry.rank`` and then by write
    order, so an atom that declared it needs another folds outside it whatever
    order the composition happened to list them in, and two layers at one rank
    are two the graph does not order -- which is the case where the fold really
    does have to commute.
    """

    build: Callable[[object], object]


@dataclass(frozen=True, slots=True)
class ServiceEntry:
    """One registration: the value, how it was checked, and when it was written.

    ``order`` is the whole reason this is a value and not a bare service: it is
    what decides which of two writes to one key across two nodes wins, so an
    entry that is moved out of a table and put back has to be put back *as this
    object*.  Re-registering the value would mint a fresh number and hand it a
    key it may have lost.
    """

    service: object
    protocol: type | None = None
    scope: ServiceScope = "tree"
    order: int = -1


class ServiceRegistry:
    """Typed, named service registry with runtime protocol checks."""

    __slots__ = (
        "_layers",
        "_linked",
        "_parent",
        "_roles",
        "_services",
        "_write_observer",
        "rank",
    )

    def __init__(self, *, parent: ServiceRegistry | None = None) -> None:
        self._services: dict[str, ServiceEntry] = {}
        #: The dependency depth of whoever owns this node. Read at fold time
        #: rather than stamped onto entries, because it is a function of the
        #: atoms present right now: one arriving can deepen an atom that
        #: declared it comes after it, and a number copied into an entry at
        #: write time would be a picture taken before that happened.
        self.rank = 0
        #: Decorations of a key, kept beside the entries rather than among
        #: them: a layer does not compete for the key, and a node has to be
        #: able to hold a base and a layer for one name, or two layers.
        self._layers: dict[str, list[ServiceEntry]] = {}
        #: Which of this node's keys were bound as roles rather than plain
        #: registrations. A role is a cell -- one model, one executor -- and
        #: two unordered atoms binding one is refused; a plain key shadows on
        #: purpose and uncovers when the writer above it leaves.
        self._roles: set[str] = set()
        self._write_observer: WriteObserver | None = None
        #: The registry this one resolves through when it holds no entry for a
        #: key. Set once, at construction, by whoever derives a context; there
        #: is no way to reparent, so nothing about a read can change under it.
        self._parent = parent
        #: Registries whose writes this one aggregates. Held by the parent
        #: rather than flagged on the child: linking is the parent's data, and
        #: unlinking is removing an element from a list rather than setting a
        #: bit on the thing removed.
        self._linked: list[ServiceRegistry] = []

    # --- Linking ---

    def link(self, child: ServiceRegistry) -> None:
        """Aggregate ``child``'s writes into this registry's reads."""

        if child is self:
            raise ValueError("a service registry cannot be linked into itself")
        if any(existing is child for existing in self._linked):
            return
        self._linked.append(child)

    def unlink(self, child: ServiceRegistry) -> None:
        """Stop aggregating ``child``; safe to repeat."""

        self._linked = [existing for existing in self._linked if existing is not child]

    def take_own(self) -> ServiceRegistry:
        """Move this registry's own table into a fresh one, leaving this empty.

        What unlinking a context does to its services: they leave the tree
        without being unregistered one by one, and the caller decides whether
        they come back.
        """

        moved = ServiceRegistry()
        moved._services = self._services
        self._services = {}
        return moved

    def give_own(self, other: ServiceRegistry) -> None:
        """Take an own table back from a registry ``take_own`` moved it into.

        Anything written since keeps its value, because it was written later
        and later wins; the order numbers already say so.
        """

        restored = dict(other._services)
        restored.update(self._services)
        self._services = restored
        other._services = {}

    def own_table(self) -> dict[str, ServiceEntry]:
        """This node's own entries, chain excluded, with their write order.

        For the reader that has to say which node a key resolves *out of*,
        across nodes that cannot see each other: the answer is the highest
        order, the same tiebreak ``_lookup`` applies, and asking each node "is
        this yours" would answer in a different order from resolution.  The
        entries are frozen, so handing them out is handing out what they say,
        and iterating it is this node's own keys.
        """

        return dict(self._services)

    def swap_entry(self, name: str, entry: ServiceEntry | None) -> ServiceEntry | None:
        """Make this node's own entry for ``name`` be ``entry``; return the old.

        The single-key ``take_own``/``give_own``, and there for the same reason:
        a caller about to shadow a key *in the table that holds it* has to be
        able to put back exactly what was there.  ``register`` cannot do that --
        it mints the next write order, so the entry it puts back is newer than
        the one it replaced and wins a key that entry had lost.  The undo of a
        swap is the same swap with what it handed back, which is why this is one
        operation and not two.

        No observer fires either way: moving an entry is not a write, and who
        the key belongs to afterwards is a property of the tree rather than of
        the node it moved in or out of.  The caller reads that off the tree and
        re-files it, the way a removal already does.
        """

        held = self._services.pop(name, None)
        if entry is not None:
            self._services[name] = entry
        return held

    # --- Resolution ---

    def _lookup(
        self,
        name: str,
        *,
        skip: ServiceRegistry | None = None,
    ) -> ServiceEntry | None:
        """The winning entry for ``name`` across this node and its chain.

        ``skip`` is the node the lookup arrived from, so a child asking its
        parent does not see itself a second time through the parent's link
        list. Depth is bounded at two — a session and the contexts linked into
        it — so this recursion cannot run away.
        """

        best = _node_entry(self, name)
        for child in self._linked:
            if child is skip:
                continue
            entry = _node_entry(child, name)
            if entry is not None and (best is None or entry.order > best.order):
                best = entry
        parent = self._parent
        if parent is not None and parent is not skip:
            entry = parent._lookup(name, skip=self)
            if entry is not None and (best is None or entry.order > best.order):
                best = entry
        return best

    def _resolved(
        self,
        *,
        skip: ServiceRegistry | None = None,
    ) -> dict[str, ServiceEntry]:
        """Every key this node can see, each at its winning entry."""

        resolved: dict[str, ServiceEntry] = {}

        def _offer(candidates: dict[str, ServiceEntry]) -> None:
            for key, entry in candidates.items():
                held = resolved.get(key)
                if held is None or entry.order > held.order:
                    resolved[key] = entry

        def _offer_node(node: ServiceRegistry) -> None:
            # Layers count: a key nothing binds but something decorates is a
            # key this registry holds, and a reader that listed only the bound
            # ones would not see it at all.
            _offer(node._services)
            for key, rows in node._layers.items():
                for entry in rows:
                    held = resolved.get(key)
                    if held is None or entry.order > held.order:
                        resolved[key] = entry

        _offer_node(self)
        for child in self._linked:
            if child is skip:
                continue
            _offer_node(child)
        parent = self._parent
        if parent is not None and parent is not skip:
            _offer(parent._resolved(skip=self))
        return resolved

    def set_write_observer(self, observer: WriteObserver | None) -> None:
        """Install a callback fired after every write, ``register`` included.

        The owning session uses this to attribute each service to the atom
        that put it there, and to announce role bindings. Attribution has to
        happen at the mutation site: a session cannot ask N call sites to
        remember to report themselves. Observers are per-registry and are not
        copied by ``inherit_from`` / ``update_from``.
        """

        self._write_observer = observer

    def register(
        self,
        name: str,
        service: object,
        protocol: type | None = None,
        *,
        scope: ServiceScope = "tree",
    ) -> None:
        """Register a service by name.

        ``protocol`` is optional — when given, ``isinstance(service,
        protocol)`` is checked (works with ``@runtime_checkable``
        Protocol classes and regular base classes).

        ``scope="session"`` services are local runtime state and stay behind.
        ``scope="tree"`` services are inherited by spawned child sessions.
        Re-registering the same name replaces the previous service.
        """

        self._write(name, service, protocol, scope=scope, role_bind=False)

    def _write(
        self,
        name: str,
        service: object,
        protocol: type | None,
        *,
        scope: ServiceScope,
        role_bind: bool,
    ) -> None:
        if protocol is not None and not isinstance(service, protocol):
            raise ServiceTypeMismatch(
                f"service {name!r}: {type(service).__name__} does not "
                f"satisfy {protocol.__name__}"
            )
        self._services[name] = ServiceEntry(
            service=service,
            protocol=protocol,
            scope=scope,
            order=next(_WRITE_ORDER),
        )
        if role_bind:
            self._roles.add(name)
        if self._write_observer is not None:
            self._write_observer(name, service, scope, role_bind=role_bind)

    def own_roles(self) -> frozenset[str]:
        """The keys this node bound as roles, chain excluded."""

        return frozenset(self._roles)

    def own_layers(self) -> dict[str, tuple[ServiceEntry, ...]]:
        """This node's own layers, chain excluded, oldest write first.

        For a reader that has to say what one context contributes to a key it
        does not own -- a digest that reported only the fold would call a
        session with two layers equal to one with the same net executor.
        """

        return {name: tuple(rows) for name, rows in self._layers.items() if rows}

    def layer(
        self,
        name: str,
        build: Callable[[object], object],
        *,
        scope: ServiceScope = "tree",
    ) -> None:
        """Decorate ``name`` rather than replacing what it holds.

        For the case several atoms genuinely share: each wants to wrap the tool
        executor, or the permission policy, without any of them owning it. Each
        writes its own layer into its own table, so the writes do not collide
        and none of them holds another's value; the key resolves to the fold.

        What this buys is that a layer can be taken out of the middle. Nothing
        else can: a wrapper built by reading the key and writing itself back is
        a link in a chain nobody else can see, and detaching the atom that made
        it leaves the link in place.
        """

        entry = ServiceEntry(
            service=ServiceLayer(build=build),
            protocol=None,
            scope=scope,
            order=next(_WRITE_ORDER),
        )
        self._layers.setdefault(name, []).append(entry)
        if self._write_observer is not None:
            self._write_observer(name, entry.service, scope, role_bind=False)

    def bind(
        self,
        role: ServiceRole[T],
        service: T,
        *,
        replace: bool = False,
        scope: ServiceScope | None = None,
    ) -> None:
        """Bind an implementation to a well-known role.

        Unlike ``register``, binding an already-bound role without
        ``replace=True`` raises — boundary bindings are deliberate, not
        last-writer-wins.
        """

        if self.has(role.key) and not replace:
            raise ValueError(f"service {role.key!r} already bound")
        effective_scope = role.scope if scope is None else scope
        self._write(
            role.key,
            service,
            role.protocol,
            scope=effective_scope,
            role_bind=True,
        )

    @overload
    def get(self, name: str) -> object | None: ...

    @overload
    def get(self, name: str, protocol: type[T]) -> T | None: ...

    def get(self, name: str, protocol: type | None = None) -> object | None:
        """Look up a service.  Returns None if not found.

        When ``protocol`` is given, validates the stored service against it.
        A registered service with the wrong type is a broken composition, not
        an absent optional capability, so it raises ``ServiceTypeMismatch``.
        """

        entry = self._lookup(name)
        if entry is None:
            return None
        service = entry.service
        if type(service) is ServiceLayer:
            service = _fold(self, name)
            if service is None:
                return None
        if protocol is not None and not isinstance(service, protocol):
            raise ServiceTypeMismatch(
                f"service {name!r}: expected {protocol.__name__}, got "
                f"{type(service).__name__}"
            )
        return service

    def get_role(self, role: ServiceRole[T]) -> T | None:
        """Look up a role binding; None when absent, typed by the role."""

        if role.protocol is None:
            return cast("T | None", self.get(role.key))
        return self.get(role.key, role.protocol)

    def require_role(self, role: ServiceRole[T]) -> T:
        """Look up a role binding or raise ``ServiceNotFound``."""

        service = self.get_role(role)
        if service is None:
            raise ServiceNotFound(role.key)
        return service

    def require(self, name: str, protocol: type[T]) -> T:
        """Look up a service or raise ``ServiceNotFound``."""

        service = self.get(name, protocol)
        if service is None:
            raise ServiceNotFound(name)
        return service  # type: ignore[return-value]

    def has(self, name: str) -> bool:
        return self._lookup(name) is not None

    def unregister(self, name: str) -> object | None:
        """Remove a service from this registry's own table.

        Own table only. A key another node of the chain holds is not this
        node's to take away — removing it there would be one context reaching
        into another's, which is the thing the tree exists to prevent. A caller
        that means to shadow it writes its own value instead.
        """

        entry = self._services.pop(name, None)
        return None if entry is None else entry.service

    def names(self) -> list[str]:
        """Every key the chain resolves, in the order they were written."""

        resolved = self._resolved()
        return sorted(resolved, key=lambda name: resolved[name].order)

    def scope(self, name: str) -> ServiceScope | None:
        entry = self._lookup(name)
        return None if entry is None else entry.scope

    def update_from(self, other: ServiceRegistry) -> None:
        """Merge everything ``other`` resolves into this one (other wins)."""

        self._services.update(other._resolved())

    def inherit_from(self, other: ServiceRegistry) -> None:
        """Merge inherited services and leave session-local state behind."""
        self._services.update(
            {
                name: entry
                for name, entry in other._resolved().items()
                if entry.scope in _INHERITED_SCOPES
            }
        )


__all__ = [
    "ServiceEntry",
    "ServiceNotFound",
    "ServiceRegistry",
    "ServiceRole",
    "ServiceScope",
    "ServiceTypeMismatch",
    "WriteObserver",
]


def _node_entry(registry: ServiceRegistry, name: str) -> ServiceEntry | None:
    """This node's newest write to ``name``, layer or value.

    A layer is a write to the key even though it does not replace what the
    key holds, so every reader that asks "is this key here, and whose is
    it" has to count it.  Only ``get`` cares about the difference, and it
    asks by looking at what it got back.
    """

    best = registry._services.get(name)
    for entry in registry._layers.get(name, ()):
        if best is None or entry.order > best.order:
            best = entry
    return best


def _base_entry(
    registry: ServiceRegistry,
    name: str,
    *,
    skip: ServiceRegistry | None = None,
) -> ServiceEntry | None:
    """What the layers decorate: the newest write that is not a layer."""

    best = registry._services.get(name)
    for child in registry._linked:
        if child is skip:
            continue
        entry = child._services.get(name)
        if entry is not None and (best is None or entry.order > best.order):
            best = entry
    parent = registry._parent
    if parent is not None and parent is not skip:
        entry = _base_entry(parent, name, skip=registry)
        if entry is not None and (best is None or entry.order > best.order):
            best = entry
    return best


def _collect(
    registry: ServiceRegistry,
    name: str,
    *,
    skip: ServiceRegistry | None = None,
) -> list[ServiceEntry]:
    """Every *layer* on ``name`` anywhere in the chain, innermost first.

    Layers do not compete for the key -- they decorate whatever holds it -- so
    they are kept beside the entries rather than among them.  Two layers from
    one node, and a base and a layer in one node, both have to be
    representable, and a single slot per key cannot represent either.

    Ordered by the writing node's rank and then by write order: the graph first
    where it has something to say, and the composition's own listing for the
    pairs it does not order.  The rank is read here rather than stamped on the
    entry, so an atom arriving and deepening one that declared it comes after
    it changes the fold rather than leaving an older number behind.
    """

    return [entry for _rank, _order, entry in _layered(registry, name, skip=skip)]


def _layered(
    registry: ServiceRegistry,
    name: str,
    *,
    skip: ServiceRegistry | None = None,
) -> list[tuple[int, int, ServiceEntry]]:
    found: list[tuple[int, int, ServiceEntry]] = [
        (registry.rank, entry.order, entry) for entry in registry._layers.get(name, ())
    ]
    for child in registry._linked:
        if child is skip:
            continue
        found.extend(
            (child.rank, entry.order, entry) for entry in child._layers.get(name, ())
        )
    parent = registry._parent
    if parent is not None and parent is not skip:
        found.extend(_layered(parent, name, skip=registry))
    found.sort(key=lambda row: (row[0], row[1]))
    return found


def _fold(registry: ServiceRegistry, name: str) -> object | None:
    """A layered key's value: every layer applied over the current base.

    The base is the newest write that is not a layer, so a later ``bind``
    of a fresh implementation is decorated by the layers that are present
    rather than escaping them -- a layer says "wrap this key", not "wrap
    the value I happened to find".
    """

    found = _base_entry(registry, name)
    value: object | None = None if found is None else found.service
    for entry in _collect(registry, name):
        layer = entry.service
        if type(layer) is ServiceLayer:
            value = layer.build(value)
    return value
