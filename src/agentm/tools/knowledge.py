"""Knowledge Store tools for path-based namespace operations.

Path utility functions are fully implemented (pure functions).
"""

from __future__ import annotations

from typing import Any, Optional

STORE_ROOT: tuple[str, ...] = ("knowledge",)

# Module-level store reference, set by builder at startup
_store = None


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
    return STORE_ROOT + tuple(parts[:-1]), parts[-1]


def namespace_to_path(namespace: tuple[str, ...]) -> str:
    """Convert namespace tuple back to path string.

    ("knowledge", "failure_pattern", "database") -> "/failure_pattern/database/"
    """
    parts = namespace[len(STORE_ROOT) :]
    return "/" + "/".join(parts) + "/" if parts else "/"


# --- Knowledge tool implementations ---


def knowledge_search(
    query: str,
    path: str = "/",
    filter: Optional[dict[str, Any]] = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Semantic search over the knowledge base."""
    namespace_prefix = path_to_namespace(path)
    results = _store.search(namespace_prefix=namespace_prefix, query=query, filter=filter, limit=limit)
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
    namespace = path_to_namespace(path)
    children = _store.list_namespaces(prefix=namespace, max_depth=len(namespace) + 1)
    entries = _store.search(namespace_prefix=namespace, limit=100)
    return {
        "path": path,
        "sub_paths": [namespace_to_path(ns) for ns in children],
        "entries": [{"key": e.key, "value": e.value} for e in entries],
    }


def knowledge_read(path: str) -> dict[str, Any]:
    """Read a specific knowledge entry by its full path."""
    namespace, key = path_to_namespace_and_key(path)
    item = _store.get(namespace=namespace, key=key)
    if item is None:
        return {"error": f"Not found: {path}"}
    return item.value


def knowledge_write(
    path: str,
    entry: dict[str, Any],
    merge: bool = False,
) -> str:
    """Create or update a knowledge entry."""
    namespace, key = path_to_namespace_and_key(path)
    if merge:
        existing = _store.get(namespace=namespace, key=key)
        if existing is not None:
            merged = {**existing.value, **entry}
            entry = merged
    _store.put(namespace=namespace, key=key, value=entry, index=["title", "description"])
    return f"Written: {path}"


def knowledge_delete(path: str) -> str:
    """Delete a knowledge entry."""
    namespace, key = path_to_namespace_and_key(path)
    _store.delete(namespace=namespace, key=key)
    return f"Deleted: {path}"
