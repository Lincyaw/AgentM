"""Knowledge Store tools for path-based namespace operations.

Path utility functions are fully implemented (pure functions).
"""

from __future__ import annotations

import contextvars
from typing import Any

STORE_ROOT: tuple[str, ...] = ("knowledge",)

# Module-level store reference, set by builder at startup
_store_var: contextvars.ContextVar[Any | None] = contextvars.ContextVar("knowledge_store", default=None)


# --- Path utility functions (fully implemented) ---


def path_to_namespace(path: str) -> tuple[str, ...]:
    """Convert filesystem-like path to LangGraph Store namespace tuple.

    "/" -> ("knowledge",)
    "/failure_pattern/database/" -> ("knowledge", "failure_pattern", "database")
    """
    parts = [p for p in path.strip("/").split("/") if p]
    return STORE_ROOT + tuple(parts)


def path_to_namespace_and_key(path: str) -> tuple[tuple[str, ...], str]:
    """Split path into namespace + key.

    "/failure_pattern/database/connection_pool_exhaustion"
    -> (("knowledge", "failure_pattern", "database"), "connection_pool_exhaustion")
    """
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts:
        raise ValueError(
            "Path must contain at least one segment for a key. "
            "Use path_to_namespace() for namespace-only paths."
        )
    return STORE_ROOT + tuple(parts[:-1]), parts[-1]


def namespace_to_path(namespace: tuple[str, ...]) -> str:
    """Convert namespace tuple back to path string.

    ("knowledge", "failure_pattern", "database") -> "/failure_pattern/database/"
    """
    parts = namespace[len(STORE_ROOT) :]
    return "/" + "/".join(parts) + "/" if parts else "/"


# --- Knowledge tool implementations ---


def set_store(store: Any) -> None:
    """Set the LangGraph Store backend. Called by builder at startup."""
    _store_var.set(store)


def knowledge_search(
    query: str,
    path: str = "/",
    filter: dict[str, Any] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Semantic search over the knowledge base."""
    store = _store_var.get()
    if store is None:
        raise RuntimeError("Knowledge store not initialized — call set_store() first")
    namespace_prefix = path_to_namespace(path)
    results = store.search(namespace_prefix=namespace_prefix, query=query, filter=filter, limit=limit)
    return [
        {
            "key": r.key,
            "namespace": namespace_to_path(r.namespace),
            "value": r.value,
            "score": getattr(r, "score", None),
        }
        for r in results
    ]


def knowledge_list(path: str = "/") -> dict[str, Any]:
    """List the structure of the knowledge base at a given path."""
    store = _store_var.get()
    if store is None:
        raise RuntimeError("Knowledge store not initialized — call set_store() first")
    namespace = path_to_namespace(path)
    children = store.list_namespaces(prefix=namespace, max_depth=len(namespace) + 1)
    entries = store.search(namespace_prefix=namespace, limit=100)
    return {
        "path": path,
        "sub_paths": [namespace_to_path(ns) for ns in children],
        "entries": [{"key": e.key, "value": e.value} for e in entries],
    }


def knowledge_read(path: str) -> dict[str, Any]:
    """Read a specific knowledge entry by its full path."""
    store = _store_var.get()
    if store is None:
        raise RuntimeError("Knowledge store not initialized — call set_store() first")
    namespace, key = path_to_namespace_and_key(path)
    item = store.get(namespace=namespace, key=key)
    if item is None:
        return {"error": f"Not found: {path}"}
    return item.value


def knowledge_write(
    path: str,
    entry: dict[str, Any],
    merge: bool = False,
) -> str:
    """Create or update a knowledge entry."""
    store = _store_var.get()
    if store is None:
        raise RuntimeError("Knowledge store not initialized — call set_store() first")
    namespace, key = path_to_namespace_and_key(path)
    if merge:
        existing = store.get(namespace=namespace, key=key)
        if existing is not None:
            merged = {**existing.value, **entry}
            entry = merged
    store.put(namespace=namespace, key=key, value=entry, index=["title", "description"])
    return f"Written: {path}"


def knowledge_delete(path: str) -> str:
    """Delete a knowledge entry."""
    store = _store_var.get()
    if store is None:
        raise RuntimeError("Knowledge store not initialized — call set_store() first")
    namespace, key = path_to_namespace_and_key(path)
    store.delete(namespace=namespace, key=key)
    return f"Deleted: {path}"
