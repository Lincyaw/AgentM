# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Typed service registry — scoped dependency injection for atoms.

Services are the atom-to-atom communication channel when atoms cannot import
each other directly. Each service can also declare whether it is local to one
session or inherited by child sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeVar, overload

T = TypeVar("T")
ServiceScope = Literal["session", "tree", "host", "process", "resource"]
_INHERITED_SCOPES: Final[frozenset[ServiceScope]] = frozenset(
    {"tree", "host", "process", "resource"}
)


class ServiceNotFound(KeyError):
    """Raised when a required service is not registered."""


class ServiceTypeMismatch(TypeError):
    """Raised when a registered service doesn't match the expected protocol."""


@dataclass(frozen=True, slots=True)
class _ServiceEntry:
    service: object
    protocol: type | None = None
    scope: ServiceScope = "tree"


class ServiceRegistry:
    """Typed, named service registry with runtime protocol checks."""

    __slots__ = ("_services",)

    def __init__(self) -> None:
        self._services: dict[str, _ServiceEntry] = {}

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
        ``scope="tree"`` services are session-tree capabilities.
        ``scope="host"`` services are host-owned injection points.
        ``scope="process"`` services are process-wide runtime capabilities.
        ``scope="resource"`` services are tied to an external resource boundary.
        All non-session scopes are inherited by spawned child sessions.
        Re-registering the same name replaces the previous service.
        """

        if protocol is not None:
            if not isinstance(service, protocol):
                raise ServiceTypeMismatch(
                    f"service {name!r}: {type(service).__name__} does not "
                    f"satisfy {protocol.__name__}"
                )
        self._services[name] = _ServiceEntry(
            service=service,
            protocol=protocol,
            scope=scope,
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

    def update_from(self, other: "ServiceRegistry") -> None:
        """Merge another registry into this one (other wins on conflict)."""
        self._services.update(other._services)

    def inherit_from(self, other: "ServiceRegistry") -> None:
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
    "ServiceScope",
    "ServiceTypeMismatch",
]
