# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Typed service registry — scoped dependency injection for atoms.

Services are the atom-to-atom communication channel when atoms cannot import
each other directly. Each service can also declare whether it is local to one
session or inherited by child sessions.

Well-known boundaries are described once by a ``ServiceRole`` (key + protocol
+ scope) declared in ``agentm.core.abi.roles``; ``bind``/``get_role``/
``require_role`` consume the descriptor so call sites never restate scope or
protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeVar, cast, overload

T = TypeVar("T")
ServiceScope = Literal["session", "tree"]
"""Whether a service is local to one session or inherited by child sessions."""
_INHERITED_SCOPES: Final[frozenset[ServiceScope]] = frozenset({"tree"})


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
class _ServiceEntry:
    service: object
    protocol: type | None = None
    scope: ServiceScope = "tree"


class ServiceRegistry:
    """Typed, named service registry with runtime protocol checks."""

    __slots__ = ("_services", "_write_observer")

    def __init__(self) -> None:
        self._services: dict[str, _ServiceEntry] = {}
        self._write_observer: WriteObserver | None = None

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
        self._services[name] = _ServiceEntry(
            service=service,
            protocol=protocol,
            scope=scope,
        )
        if self._write_observer is not None:
            self._write_observer(name, service, scope, role_bind=role_bind)

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

        entry = self._services.get(name)
        if entry is None:
            return None
        service = entry.service
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
        return name in self._services

    def unregister(self, name: str) -> object | None:
        """Remove a service and return the previous value, if present."""

        entry = self._services.pop(name, None)
        return None if entry is None else entry.service

    def names(self) -> list[str]:
        return list(self._services.keys())

    def scope(self, name: str) -> ServiceScope | None:
        entry = self._services.get(name)
        return None if entry is None else entry.scope

    def copy(self) -> ServiceRegistry:
        """Snapshot registrations and observer for composition transactions."""

        copied = ServiceRegistry()
        copied._services = dict(self._services)
        copied._write_observer = self._write_observer
        return copied

    def replace_from(self, other: ServiceRegistry) -> None:
        """Restore registrations and observer from ``other``."""

        self._services = dict(other._services)
        self._write_observer = other._write_observer

    def update_from(self, other: ServiceRegistry) -> None:
        """Merge another registry into this one (other wins on conflict)."""
        self._services.update(other._services)

    def inherit_from(self, other: ServiceRegistry) -> None:
        """Merge inherited services and leave session-local state behind."""
        self._services.update(
            {
                name: entry
                for name, entry in other._services.items()
                if entry.scope in _INHERITED_SCOPES
            }
        )


__all__ = [
    "ServiceNotFound",
    "ServiceRegistry",
    "ServiceRole",
    "ServiceScope",
    "ServiceTypeMismatch",
    "WriteObserver",
]
