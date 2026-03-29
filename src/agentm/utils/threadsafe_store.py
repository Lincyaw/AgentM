"""Thread-safe store base class for run-scoped shared data.

Provides a generic, thread-safe foundation for stores that need to be
shared across agents within a single run. Subclasses define their own
data types and operations while inheriting the locking infrastructure.
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class ThreadSafeStore(Generic[K, V]):
    """Thread-safe, run-scoped store base class.

    Subclasses define their own value type and business logic while
    inheriting the locking infrastructure and basic CRUD operations.

    Example:
        class HypothesisStore(ThreadSafeStore[str, HypothesisEntry]):
            def update(self, ...) -> HypothesisEntry:
                with self._lock:
                    # custom logic
                    self._data[id] = entry
                    return entry
    """

    def __init__(self) -> None:
        self._data: dict[K, V] = {}
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        """Return the value for *key*, or ``None`` if not found."""
        with self._lock:
            return self._data.get(key)

    def get_all(self) -> dict[K, V]:
        """Return a shallow copy of all stored data."""
        with self._lock:
            return dict(self._data)

    def remove(self, key: K) -> bool:
        """Remove *key* from the store. Returns True if it existed."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries from the store."""
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        """Return the number of entries in the store."""
        with self._lock:
            return len(self._data)

    def __contains__(self, key: K) -> bool:
        """Return True if *key* exists in the store."""
        with self._lock:
            return key in self._data
