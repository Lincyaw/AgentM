"""Constitution boundary parser.

Loads ``core-manifest.yaml`` and exposes :func:`is_constitution_path` —
the predicate self-mod APIs route through to decide whether a path is on
the constitution side of the boundary.

Layer purity: this module imports only stdlib + ``yaml``. It does **not**
touch the filesystem at import time and does not reach into the
extensions / runtime packages.

Path resolution policy lives in the runtime. The runtime pushes a
manifest path onto :data:`_MANIFEST_PATH_VAR` (a :class:`ContextVar`) at
session start. A ``ContextVar`` rather than a module global so concurrent
sessions in different cwds don't race on a process-wide write — each
asyncio task / thread sees the path bound by its own session.

Caching policy: the YAML is small (single-digit kilobytes) and parsed in
~milliseconds; :func:`is_constitution_path` fires at most once per write
operation, not per turn. We therefore parse on every call rather than
maintain a process-global cache. The earlier ``functools.cache`` keyed
on path alone had two bugs the constitution boundary cannot tolerate:
(1) two concurrent sessions binding different cwds via
``configure_manifest_path`` would ``cache_clear`` each other, and
(2) an on-disk manifest edit (the exact path a self-modifying agent
would take) returned a stale parse because the key didn't include
``mtime_ns``. Dropping the cache eliminates both failure modes for
negligible cost.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath

import yaml

# Test/runtime seam — default ``None`` so importing this module never
# touches the filesystem. The runtime binds this per-session via
# :func:`configure_manifest_path`; tests use :func:`override_manifest_path`
# for scoped overrides.
_MANIFEST_PATH_VAR: ContextVar[Path | None] = ContextVar(
    "agentm_core_manifest_path", default=None
)


class CoreManifestPathUnsetError(RuntimeError):
    """Raised when manifest helpers run before a manifest path is configured."""


@dataclass(frozen=True, slots=True)
class CoreManifest:
    version: int
    constitution_paths: tuple[str, ...]
    extension_api_current: int
    extension_api_grace: int
    tier_2_atoms: tuple[str, ...]
    managed_globs: tuple[str, ...] = ()


def current_manifest_path() -> Path | None:
    """Return the manifest path bound to the current context, if any."""

    return _MANIFEST_PATH_VAR.get()


def load_core_manifest(manifest_path: Path | None = None) -> CoreManifest:
    """Load the manifest from ``manifest_path`` or the current context.

    Passing ``manifest_path`` explicitly is preferred — the runtime owns
    the policy of where ``core-manifest.yaml`` lives. The fall-back to
    :data:`_MANIFEST_PATH_VAR` exists for callers that don't have a
    handle (atoms reading the manifest indirectly via
    ``is_constitution_path``).
    """

    path = manifest_path if manifest_path is not None else _MANIFEST_PATH_VAR.get()
    if path is None:
        raise CoreManifestPathUnsetError(
            "core-manifest.yaml path not configured: call "
            "configure_manifest_path() during runtime startup, use "
            "override_manifest_path() for scoped overrides, or pass "
            "manifest_path explicitly."
        )
    return _parse_manifest(path)


def reload_manifest(manifest_path: Path | None = None) -> CoreManifest:
    """Reload the manifest (optionally against a new path).

    Retained for backwards compatibility — the parser no longer caches,
    so this is now a thin alias for :func:`load_core_manifest`. Tests and
    long-running tooling that previously called it for cache-busting
    semantics will still get a fresh parse.
    """

    return load_core_manifest(manifest_path)


def is_constitution_path(path: str) -> bool:
    cm = load_core_manifest()
    normalized = _normalize_to_repo_relative(path)
    return any(matches_manifest_glob(pattern, normalized) for pattern in cm.constitution_paths)


def matches_manifest_glob(pattern: str, path: str) -> bool:
    normalized = _normalize_to_repo_relative(path)
    return _glob_matches(pattern, normalized)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _parse_manifest(manifest_path: Path) -> CoreManifest:
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    version = int(data.get("version", 1))

    constitution = data.get("constitution") or {}
    paths_raw = constitution.get("paths") or ()
    constitution_paths = tuple(str(p) for p in paths_raw)
    managed = data.get("managed") or {}
    managed_globs = tuple(str(p) for p in (managed.get("globs") or ()))

    ext_api = data.get("extension_api") or {}
    current = int(ext_api.get("current", 1))
    deprecation = ext_api.get("deprecation") or {}
    grace = int(deprecation.get("grace", 1))

    reload_section = data.get("reload") or {}
    tier_2 = tuple(str(a) for a in (reload_section.get("tier_2_atoms") or ()))

    return CoreManifest(
        version=version,
        constitution_paths=constitution_paths,
        managed_globs=managed_globs,
        extension_api_current=current,
        extension_api_grace=grace,
        tier_2_atoms=tier_2,
    )


def _normalize_to_repo_relative(path: str) -> str:
    candidate = Path(path)
    manifest_path = _MANIFEST_PATH_VAR.get()
    if manifest_path is None:
        # No repo root configured — best-effort relative normalization.
        return PurePosixPath(candidate).as_posix()
    repo_root = manifest_path.parent
    if candidate.is_absolute():
        try:
            rel = candidate.resolve().relative_to(repo_root)
        except ValueError:
            rel = candidate
        return PurePosixPath(rel).as_posix()
    return PurePosixPath(candidate).as_posix()


def _glob_matches(pattern: str, posix_path: str) -> bool:
    regex = _compile_glob(pattern)
    return regex.fullmatch(posix_path) is not None


@cache
def _compile_glob(pattern: str) -> re.Pattern[str]:
    # Hand-rolled instead of stdlib fnmatch/PurePath.match: neither handles
    # ``**`` recursively across path components in the way the manifest needs.
    # ``**`` = "zero or more path components"; ``*`` = "any chars within a
    # single component". Tokens are processed left to right; literal segments
    # are escaped.
    parts: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                # ``**`` (optionally followed by ``/``) collapses to "zero or
                # more path components" so ``a/**`` matches ``a`` itself,
                # ``a/b``, ``a/b/c``, etc.
                if i + 2 < n and pattern[i + 2] == "/":
                    parts.append("(?:.*/)?")
                    i += 3
                else:
                    parts.append(".*")
                    i += 2
            else:
                parts.append("[^/]*")
                i += 1
        elif ch == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(ch))
            i += 1
    return re.compile("".join(parts))


def configure_manifest_path(manifest_path: Path) -> Token[Path | None]:
    """Bind ``manifest_path`` to the current async/thread context.

    Returns the :class:`Token` so the caller can reset the binding on
    teardown. The runtime uses this at session construction and resets in
    ``AgentSession.shutdown``; tests prefer :func:`override_manifest_path`
    which wraps the same call in a context manager.
    """

    token = _MANIFEST_PATH_VAR.set(Path(manifest_path))
    return token


def reset_manifest_path(token: Token[Path | None]) -> None:
    """Undo a :func:`configure_manifest_path` call."""

    _MANIFEST_PATH_VAR.reset(token)


@contextmanager
def override_manifest_path(manifest_path: Path) -> Iterator[None]:
    """Scoped override for tests and short-lived tooling."""

    token = configure_manifest_path(manifest_path)
    try:
        yield
    finally:
        reset_manifest_path(token)


__all__ = [
    "CoreManifest",
    "CoreManifestPathUnsetError",
    "configure_manifest_path",
    "current_manifest_path",
    "is_constitution_path",
    "load_core_manifest",
    "matches_manifest_glob",
    "override_manifest_path",
    "reload_manifest",
    "reset_manifest_path",
]
