"""Typed service registry — session-scoped dependency injection.

Replaces the untyped ``dict[str, Any]`` service pattern from v1.
Services are the atom-to-atom communication channel when §11 forbids
direct imports.

Each service is registered with a Protocol type so consumers get
type-safe access.  The registry validates at registration time.
"""

from __future__ import annotations

from typing import TypeVar, overload

from loguru import logger

T = TypeVar("T")


class ServiceNotFound(KeyError):
    """Raised when a required service is not registered."""


class ServiceTypeMismatch(TypeError):
    """Raised when a registered service doesn't match the expected protocol."""


class ServiceRegistry:
    """Typed, named service registry with runtime protocol checks."""

    __slots__ = ("_services", "_protocols")

    def __init__(self) -> None:
        self._services: dict[str, object] = {}
        self._protocols: dict[str, type] = {}

    def register(self, name: str, service: object, protocol: type | None = None) -> None:
        """Register a service by name.

        ``protocol`` is optional — when given, ``isinstance(service,
        protocol)`` is checked (works with ``@runtime_checkable``
        Protocol classes and regular base classes).

        Re-registering the same name replaces the previous service.
        """

        if protocol is not None:
            if not isinstance(service, protocol):
                raise ServiceTypeMismatch(
                    f"service {name!r}: {type(service).__name__} does not "
                    f"satisfy {protocol.__name__}"
                )
            self._protocols[name] = protocol
        self._services[name] = service

    @overload
    def get(self, name: str) -> object | None: ...

    @overload
    def get(self, name: str, protocol: type[T]) -> T | None: ...

    def get(self, name: str, protocol: type | None = None) -> object | None:
        """Look up a service.  Returns None if not found.

        When ``protocol`` is given, validates the stored service
        against it — a mismatch logs a warning and returns None rather
        than handing back a wrong-typed object.
        """

        service = self._services.get(name)
        if service is None:
            return None
        if protocol is not None and not isinstance(service, protocol):
            logger.warning(
                "service {!r}: expected {}, got {}",
                name,
                protocol.__name__,
                type(service).__name__,
            )
            return None
        return service

    def require(self, name: str, protocol: type[T]) -> T:
        """Look up a service or raise ``ServiceNotFound``."""

        service = self.get(name, protocol)
        if service is None:
            raise ServiceNotFound(name)
        return service  # type: ignore[return-value]

    def has(self, name: str) -> bool:
        return name in self._services

    def names(self) -> list[str]:
        return list(self._services.keys())

    def update_from(self, other: "ServiceRegistry") -> None:
        """Merge another registry into this one (other wins on conflict)."""
        self._services.update(other._services)
        self._protocols.update(other._protocols)


__all__ = [
    "ServiceNotFound",
    "ServiceRegistry",
    "ServiceTypeMismatch",
]
