"""Auto-discovery of built-in extensions.

Walks ``agentm/extensions/builtin/`` and imports every ``<name>.py`` module
that is not a package (subpackages are forbidden by the §11 single-file
rule and are rejected by :func:`discover_builtin`). Used by:

- ``load_scenario`` — to validate that scenario YAML references resolve
  to real atoms with a clear error.
- ``validate.validate_builtin`` — to drive the §11 contract checks.
- Future agent tooling that needs "what atoms exist".

Discovery is **memoized at process scope** so repeated calls are cheap.
Tests that mutate the catalog (rare) call :func:`reset_cache`.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from agentm.extensions import ExtensionManifest

USER_ATOM_MODULE_PREFIX = "_agentm_user_atom__"
"""Synthetic module-name prefix shared with :class:`AtomReloader`. Atoms
auto-discovered from ``<cwd>/.agentm/atoms/`` are imported under
``_agentm_user_atom__<name>`` so a subsequent ``api.install_atom``,
``reload_atom``, or ``unload_atom`` call sees the same module identity it
would have synthesised itself."""


@dataclass(frozen=True, slots=True)
class BuiltinEntry:
    """One discovered built-in extension."""

    name: str
    """The ``MANIFEST.name`` value (must equal module stem)."""

    module_path: str
    """Full dotted import path, e.g. ``agentm.extensions.builtin.permission``."""

    module: ModuleType
    """The imported module object — convenient for tooling that wants to
    introspect ``install`` or other module attributes."""

    manifest: ExtensionManifest


_CACHE: dict[str, BuiltinEntry] | None = None


def reset_cache() -> None:
    """Drop the memoized discovery result. Tests that mutate the catalog
    (synthesize a fixture extension on the fly) call this between cases."""

    global _CACHE
    _CACHE = None


def discover_builtin() -> dict[str, BuiltinEntry]:
    """Return ``name → BuiltinEntry`` for every module under
    ``agentm/extensions/builtin/``.

    Skipped:
    - ``__init__`` (the package init itself, no manifest expected)
    - any name starting with ``_`` (test fixtures, private staging files)

    Errors:
    - ``ImportError`` propagates if a module fails to import (the validator
      reports this more gently; this function intentionally does not).
    - ``RuntimeError`` if a module declares no ``MANIFEST`` symbol or its
      name disagrees with the module stem.
    """

    global _CACHE
    if _CACHE is not None:
        return _CACHE

    pkg = importlib.import_module("agentm.extensions.builtin")
    pkg_path = Path(pkg.__file__).parent if pkg.__file__ else None
    if pkg_path is None:
        _CACHE = {}
        return _CACHE

    entries: dict[str, BuiltinEntry] = {}
    for info in pkgutil.iter_modules([str(pkg_path)]):
        if info.ispkg:
            # §11: subpackages are forbidden. We surface this as a
            # validator issue rather than swallow it; importing it would
            # also be load-bearing nonsense, so we just skip.
            continue
        if info.name.startswith("_"):
            continue

        module_path = f"agentm.extensions.builtin.{info.name}"
        module = importlib.import_module(module_path)
        manifest_obj: Any = getattr(module, "MANIFEST", None)
        if not isinstance(manifest_obj, ExtensionManifest):
            raise RuntimeError(
                f"extension {module_path!r} is missing a module-level "
                f"MANIFEST: ExtensionManifest constant"
            )
        if manifest_obj.name != info.name:
            raise RuntimeError(
                f"extension {module_path!r} has MANIFEST.name="
                f"{manifest_obj.name!r} which disagrees with module stem "
                f"{info.name!r}"
            )
        entries[info.name] = BuiltinEntry(
            name=info.name,
            module_path=module_path,
            module=module,
            manifest=manifest_obj,
        )

    _CACHE = entries
    return entries


def discover_user_atoms(cwd: Path) -> dict[str, BuiltinEntry]:
    """Return ``name → BuiltinEntry`` for every user-installed atom under
    ``<cwd>/.agentm/atoms/<name>.py``.

    These are atoms previously committed by ``api.install_atom`` (canonical
    landing zone documented on ``tool_catalog.install_atom``). On a fresh
    session they would otherwise be invisible — the framework loader only
    walks ``agentm/extensions/builtin/`` — which breaks the plug-and-play
    promise across process restarts. Auto-loading them here keeps the
    catalog and the running session in sync without forcing the user to
    re-execute ``install_atom`` every boot.

    Returned entries use a synthetic module path ``_agentm_user_atom__<name>``
    so they don't shadow real ``agentm.*`` packages and so reload/unload
    paths in :class:`AtomReloader` can address the same module identity.

    Errors:
    - File-level :class:`SyntaxError` / :class:`ImportError` propagate so the
      caller can decide whether to skip or fail loudly. The harness's
      auto-discovery branch catches these and downgrades to a diagnostic.
    - ``RuntimeError`` if the module declares no ``MANIFEST``, the manifest's
      ``name`` disagrees with the file stem, or the file declares
      ``tier >= 2`` (agent-installed atoms are tier-1 by contract — we
      refuse to silently auto-load anything privileged).
    """

    atoms_dir = cwd / ".agentm" / "atoms"
    if not atoms_dir.is_dir():
        return {}

    entries: dict[str, BuiltinEntry] = {}
    for path in sorted(atoms_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        stem = path.stem
        module_path = f"{USER_ATOM_MODULE_PREFIX}{stem}"
        spec = importlib.util.spec_from_file_location(module_path, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not build import spec for {path!s}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_path, None)
            raise

        manifest_obj: Any = getattr(module, "MANIFEST", None)
        if not isinstance(manifest_obj, ExtensionManifest):
            sys.modules.pop(module_path, None)
            raise RuntimeError(
                f"user atom {path!s} is missing a module-level "
                f"MANIFEST: ExtensionManifest constant"
            )
        if manifest_obj.name != stem:
            sys.modules.pop(module_path, None)
            raise RuntimeError(
                f"user atom {path!s} has MANIFEST.name="
                f"{manifest_obj.name!r} which disagrees with file stem "
                f"{stem!r}"
            )
        if manifest_obj.tier >= 2:
            sys.modules.pop(module_path, None)
            raise RuntimeError(
                f"user atom {path!s} declares tier={manifest_obj.tier}; "
                "auto-discovery refuses to load tier>=2 atoms — re-install "
                "them explicitly through a scenario manifest if intended"
            )
        entries[stem] = BuiltinEntry(
            name=stem,
            module_path=module_path,
            module=module,
            manifest=manifest_obj,
        )
    return entries


__all__ = [
    "BuiltinEntry",
    "USER_ATOM_MODULE_PREFIX",
    "discover_builtin",
    "discover_user_atoms",
    "reset_cache",
]
