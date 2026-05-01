from __future__ import annotations

import threading
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class ThreadSafeStore(Generic[K, V]):
    """Thread-safe store base class for run-scoped shared data."""

    def __init__(self) -> None:
        self._data: dict[K, V] = {}
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        with self._lock:
            return self._data.get(key)

    def get_all(self) -> dict[K, V]:
        with self._lock:
            return dict(self._data)

    def remove(self, key: K) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._data
