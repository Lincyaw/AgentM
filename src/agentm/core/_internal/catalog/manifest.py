"""Constitution boundary parser.

Loads ``core-manifest.yaml`` and exposes :func:`is_constitution_path` —
the predicate self-mod APIs route through to decide whether a path is on
the constitution side of the boundary.

Layer purity: this module imports only stdlib + ``yaml``. It does **not**
touch the filesystem at import time and does not reach into the
extensions / harness packages.

Path resolution policy lives in the harness. The harness sets
:data:`_MANIFEST_PATH` (typically to ``<cwd>/core-manifest.yaml``) before
the predicate is consulted. Tests monkeypatch the same module attribute
and call :func:`reload_manifest` to repoint the loader at a temp file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath

import yaml

# Test/harness seam — initially ``None`` so importing this module never
# touches the filesystem. Callers (typically the harness during session
# start-up, or tests via monkeypatch) assign a concrete path before any
# constitution-boundary check runs.
_MANIFEST_PATH: Path | None = None


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


def load_core_manifest(manifest_path: Path | None = None) -> CoreManifest:
    """Load the manifest from ``manifest_path`` or the configured seam.

    Passing ``manifest_path`` explicitly is preferred — the harness owns
    the policy of where ``core-manifest.yaml`` lives. The fall-back to
    :data:`_MANIFEST_PATH` exists so existing test seams (monkeypatching
    the module attribute) keep working without sprinkling explicit paths
    through every caller.
    """

    path = manifest_path if manifest_path is not None else _MANIFEST_PATH
    if path is None:
        raise CoreManifestPathUnsetError(
            "core-manifest.yaml path not configured: assign "
            "agentm.core._internal.catalog.manifest._MANIFEST_PATH or pass "
            "manifest_path explicitly."
        )
    return _load_cached(path)


def reload_manifest(manifest_path: Path | None = None) -> CoreManifest:
    """Drop the cache and reload (optionally against a new path)."""

    _load_cached.cache_clear()
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


@cache
def _load_cached(manifest_path: Path) -> CoreManifest:
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
    if _MANIFEST_PATH is None:
        # No repo root configured — best-effort relative normalization.
        return PurePosixPath(candidate).as_posix()
    repo_root = _MANIFEST_PATH.resolve().parent
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


def configure_manifest_path(manifest_path: Path) -> None:
    """Convenience setter for the harness/CLI startup path.

    Equivalent to assigning :data:`_MANIFEST_PATH` and calling
    :func:`reload_manifest`. Provided so harness code does not need to
    touch a private-named attribute.
    """

    global _MANIFEST_PATH
    _MANIFEST_PATH = Path(manifest_path)
    reload_manifest()


__all__ = [
    "CoreManifest",
    "CoreManifestPathUnsetError",
    "configure_manifest_path",
    "is_constitution_path",
    "load_core_manifest",
    "matches_manifest_glob",
    "reload_manifest",
]
