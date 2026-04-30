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
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from agentm.extensions import ExtensionManifest


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


__all__ = [
    "BuiltinEntry",
    "discover_builtin",
    "reset_cache",
]
