"""Auto-discovery of built-in extensions.

Walks ``agentm/extensions/builtin/`` and imports every ``<name>.py`` module
that is not a package (subpackages are forbidden by the single-file
rule and are rejected by :func:`discover_builtin`). Used by:

- ``load_scenario`` — to validate that scenario YAML references resolve
  to real atoms with a clear error.
- ``validate.validate_builtin`` — to drive the contract checks.
- Future agent tooling that needs "what atoms exist".

Discovery is **memoized at process scope** so repeated calls are cheap.
Tests that mutate the catalog (rare) call :func:`reset_cache`.
"""

from __future__ import annotations

import importlib
import importlib.util
from loguru import logger
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from agentm.extensions import ExtensionManifest


CONTRIB_ATOM_MODULE_PREFIX = "_agentm_contrib__"
"""Synthetic module-name prefix for atoms discovered under
``<cwd>/contrib/extensions/<name>.py``. Distinct from the user-atom prefix
so reload paths can tell research-line extras apart from agent-installed
tier-1 atoms."""

HOME_ATOM_MODULE_PREFIX = "_agentm_home__"
"""Synthetic module-name prefix for atoms discovered under
``~/.agentm/contrib/extensions/<name>.py`` (or ``$AGENTM_HOME/contrib/
extensions/``). Distinct from the contrib prefix (source-checkout) and
the user-atom prefix (agent-installed) so reload paths can distinguish
home-installed extensions from both."""

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
_LAST_DISCOVERY_FAILURES: list[tuple[str, BaseException]] = []


def reset_cache() -> None:
    """Drop the memoized discovery result. Tests that mutate the catalog
    (synthesize a fixture extension on the fly) call this between cases."""

    global _CACHE
    _CACHE = None
    _LAST_DISCOVERY_FAILURES.clear()


def last_discovery_failures() -> list[tuple[str, BaseException]]:
    """Return ``(module_path, exception)`` for every atom that failed to
    import during the most recent :func:`discover_builtin` /
    :func:`discover_contrib_atoms` / :func:`discover_user_atoms` call.

    Callers (CLI startup banner, validator, tests) inspect this to surface
    per-atom failures without the whole catalog dying. The list is reset
    on each top-level discovery call so it always reflects the latest pass.
    """

    return list(_LAST_DISCOVERY_FAILURES)


def discover_builtin() -> dict[str, BuiltinEntry]:
    """Return ``name → BuiltinEntry`` for every module under
    ``agentm/extensions/builtin/``.

    Skipped:
    - ``__init__`` (the package init itself, no manifest expected)
    - any name starting with ``_`` (test fixtures, private staging files)

    Errors:
    - Per-atom :class:`ImportError`, :class:`RuntimeError`, and
      :class:`SyntaxError` are **caught and logged**, not propagated.
      A single broken atom must not deny the whole CLI: that violates the
      "loading replacements is the only unreplaceable substrate" axiom.
      Failures are accumulated and exposed via
      :func:`last_discovery_failures`.
    - A module that imports cleanly but lacks ``MANIFEST`` or whose
      ``MANIFEST.name`` disagrees with the module stem is recorded as a
      failure the same way.
    """

    global _CACHE
    if _CACHE is not None:
        return _CACHE

    _LAST_DISCOVERY_FAILURES.clear()

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
        try:
            module = importlib.import_module(module_path)
        except (ImportError, RuntimeError, SyntaxError) as exc:
            _LAST_DISCOVERY_FAILURES.append((module_path, exc))
            logger.warning(
                f"discover_builtin: failed to import {module_path} ({type(exc).__name__}: {exc})"
            )
            continue
        manifest_obj: Any = getattr(module, "MANIFEST", None)
        if not isinstance(manifest_obj, ExtensionManifest):
            failure: BaseException = RuntimeError(
                f"extension {module_path!r} is missing a module-level "
                f"MANIFEST: ExtensionManifest constant"
            )
            _LAST_DISCOVERY_FAILURES.append((module_path, failure))
            logger.warning(
                f"discover_builtin: skipping {module_path} ({type(failure).__name__}: {failure})"
            )
            continue
        if manifest_obj.name != info.name:
            failure = RuntimeError(
                f"extension {module_path!r} has MANIFEST.name="
                f"{manifest_obj.name!r} which disagrees with module stem "
                f"{info.name!r}"
            )
            _LAST_DISCOVERY_FAILURES.append((module_path, failure))
            logger.warning(
                f"discover_builtin: skipping {module_path} ({type(failure).__name__}: {failure})"
            )
            continue
        entries[info.name] = BuiltinEntry(
            name=info.name,
            module_path=module_path,
            module=module,
            manifest=manifest_obj,
        )

    _CACHE = entries
    return entries


def _discover_flat_atoms(
    atoms_dir: Path, *, module_prefix: str, label: str
) -> dict[str, BuiltinEntry]:
    """Walk ``atoms_dir`` for top-level ``<name>.py`` files and return
    ``name → BuiltinEntry``.

    Each file becomes a ``BuiltinEntry`` whose ``module_path`` is the
    synthetic ``f"{module_prefix}{stem}"``, registered into ``sys.modules``
    so reload paths can address it without re-executing.

    Errors:
    - Per-file :class:`SyntaxError`, :class:`ImportError`, and
      :class:`RuntimeError` are caught, logged, and recorded via
      :func:`last_discovery_failures`. A single broken atom must not deny
      every other contrib/user atom, mirroring the resilience contract of
      :func:`discover_builtin`.
    - Missing/mismatched ``MANIFEST`` and ``tier >= 2`` violations are
      also recorded as failures rather than raised — auto-discovery
      refuses to silently load tier>=2 atoms regardless.
    """

    if not atoms_dir.is_dir():
        return {}

    entries: dict[str, BuiltinEntry] = {}
    for path in sorted(atoms_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        stem = path.stem
        module_path = f"{module_prefix}{stem}"
        # Idempotent: a previous discovery (or explicit prime) may have
        # already registered the synthetic module. Re-executing every call
        # would reset module-level state and break callers that hold the
        # original import.
        existing = sys.modules.get(module_path)
        if existing is not None:
            module = existing
        else:
            try:
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
            except (ImportError, RuntimeError, SyntaxError) as exc:
                _LAST_DISCOVERY_FAILURES.append((module_path, exc))
                logger.warning(
                    f"{label}: failed to import {module_path} from {path} ({type(exc).__name__}: {exc})"
                )
                continue

        manifest_obj: Any = getattr(module, "MANIFEST", None)
        if not isinstance(manifest_obj, ExtensionManifest):
            sys.modules.pop(module_path, None)
            failure: BaseException = RuntimeError(
                f"{label} {path!s} is missing a module-level "
                f"MANIFEST: ExtensionManifest constant"
            )
            _LAST_DISCOVERY_FAILURES.append((module_path, failure))
            logger.warning(f"{label}: skipping {module_path} ({failure})")
            continue
        if manifest_obj.name != stem:
            sys.modules.pop(module_path, None)
            failure = RuntimeError(
                f"{label} {path!s} has MANIFEST.name="
                f"{manifest_obj.name!r} which disagrees with file stem "
                f"{stem!r}"
            )
            _LAST_DISCOVERY_FAILURES.append((module_path, failure))
            logger.warning(f"{label}: skipping {module_path} ({failure})")
            continue
        if manifest_obj.tier >= 2:
            sys.modules.pop(module_path, None)
            failure = RuntimeError(
                f"{label} {path!s} declares tier={manifest_obj.tier}; "
                "auto-discovery refuses to load tier>=2 atoms — re-install "
                "them explicitly through a scenario manifest if intended"
            )
            _LAST_DISCOVERY_FAILURES.append((module_path, failure))
            logger.warning(f"{label}: skipping {module_path} ({failure})")
            continue
        entries[stem] = BuiltinEntry(
            name=stem,
            module_path=module_path,
            module=module,
            manifest=manifest_obj,
        )
    return entries


def discover_by_role() -> dict[str, BuiltinEntry]:
    """Return ``role → BuiltinEntry`` over every builtin + contrib + home +
    entrypoint atom.

    Each atom may claim zero or more roles via ``MANIFEST.provides_role``;
    every role string maps to exactly one entry. A second atom claiming a
    role already taken raises :class:`RuntimeError` — the harness expects
    unambiguous resolution at session start, and a silent last-wins would
    let a contrib atom hijack a floor slot without anyone noticing.

    Home atoms (``~/.agentm/contrib/extensions/``) participate in role
    resolution alongside repo-contrib and entrypoint atoms — the user
    explicitly installed them, so they are trusted at the same level.

    User-discovered atoms (``.agentm/atoms/``) are intentionally excluded:
    a user atom hijacking the floor would mean session boot for that cwd
    silently disagrees with the SDK contract. Such conflicts surface as
    "role X has no provider" rather than a hijack.
    """

    entries: dict[str, BuiltinEntry] = {}
    sources: tuple[dict[str, BuiltinEntry], ...] = (
        discover_builtin(),
        discover_contrib_atoms(),
        discover_home_atoms(),
        discover_entrypoint_atoms(),
    )
    for source in sources:
        for entry in source.values():
            for role in entry.manifest.provides_role:
                existing = entries.get(role)
                if existing is not None and existing.name != entry.name:
                    raise RuntimeError(
                        f"role {role!r} is claimed by both "
                        f"{existing.module_path!r} and {entry.module_path!r}; "
                        "at most one atom may fulfil each role"
                    )
                entries[role] = entry
    return entries


def discover_user_atoms(cwd: Path) -> dict[str, BuiltinEntry]:
    """Atoms previously committed by ``api.install_atom`` to
    ``<cwd>/.agentm/atoms/<name>.py``. Auto-loaded so the catalog and the
    running session stay in sync across process restarts without forcing
    the user to re-execute ``install_atom`` every boot."""

    return _discover_flat_atoms(
        cwd / ".agentm" / "atoms",
        module_prefix=USER_ATOM_MODULE_PREFIX,
        label="user atom",
    )


def _agentm_repo_root() -> Path | None:
    """Return the AgentM source-checkout root, or ``None`` when running
    from a pip-installed wheel that does not ship ``contrib/``.

    Resolved from ``agentm.__file__`` (``<root>/src/agentm/__init__.py``)
    rather than from cwd because contrib atoms ship alongside the SDK,
    not alongside whatever directory the user invoked the CLI from.
    """

    import agentm  # local import: avoid a top-level cycle

    pkg_init = getattr(agentm, "__file__", None)
    if not pkg_init:
        return None
    candidate = Path(pkg_init).resolve().parent.parent.parent
    return candidate if (candidate / "contrib" / "extensions").is_dir() else None


def discover_contrib_atoms() -> dict[str, BuiltinEntry]:
    """Research-line / scenario-bound atoms shipped under
    ``<agentm-repo>/contrib/extensions/<name>.py``. Auto-discovered like
    builtins so the AgentM checkout's own contrib atoms are available
    without forcing every scenario manifest to list them, while keeping
    them physically out of the SDK ``src/agentm/extensions/builtin/``
    tree. Returns ``{}`` when running from a pip-installed wheel that
    does not include a ``contrib/`` directory."""

    repo_root = _agentm_repo_root()
    if repo_root is None:
        return {}
    return _discover_flat_atoms(
        repo_root / "contrib" / "extensions",
        module_prefix=CONTRIB_ATOM_MODULE_PREFIX,
        label="contrib atom",
    )


def discover_home_atoms() -> dict[str, BuiltinEntry]:
    """Extensions installed by the user into ``~/.agentm/contrib/extensions/``.

    Works like :func:`discover_contrib_atoms` but reads from the global home
    directory instead of the source checkout, so it works from pip-installed
    wheels. Users drop ``<name>.py`` files (each exporting ``MANIFEST`` +
    ``install``) into ``~/.agentm/contrib/extensions/`` and they auto-discover.
    """

    from agentm.core.lib import agentm_home_dir

    return _discover_flat_atoms(
        agentm_home_dir() / "contrib" / "extensions",
        module_prefix=HOME_ATOM_MODULE_PREFIX,
        label="home atom",
    )


def discover_entrypoint_atoms() -> dict[str, BuiltinEntry]:
    """Atoms published by any installed distribution via the ``agentm.atoms``
    entry-point group — the canonical "publish a plugin as a pip package" path
    (mirrors the ``agentm.scenarios`` group in ``loader.py``):

        [project.entry-points."agentm.atoms"]
        my_atom = "my_pkg.my_atom"   # importable module exposing MANIFEST + install

    Unlike builtin/contrib/user atoms, the import path is a real installed
    module (no synthetic name, no source checkout) — so it works from a wheel
    and a scenario can reference it by that same dotted ``module:`` path. The
    entry-point *name* must equal ``MANIFEST.name``. Per-atom failures are
    caught and recorded via :func:`last_discovery_failures`, never propagated.
    """

    try:
        from importlib.metadata import entry_points
    except Exception:  # noqa: BLE001 — defensive; importlib.metadata is stdlib
        return {}
    try:
        eps = entry_points(group="agentm.atoms")
    except Exception:  # noqa: BLE001 — a broken EP table must not deny the catalog
        return {}

    entries: dict[str, BuiltinEntry] = {}
    for ep in eps:
        module_path = ep.value
        try:
            module = importlib.import_module(module_path)
        except (ImportError, RuntimeError, SyntaxError) as exc:
            _LAST_DISCOVERY_FAILURES.append((module_path, exc))
            # An optional plugin whose own package/module is simply not installed
            # (e.g. the ``agent-env`` extra was not synced) raises
            # ModuleNotFoundError naming the entrypoint's own module/top package.
            # That is an expected, benign absence — log it at debug so it does
            # not scream on every session. A genuinely broken *installed* atom
            # (SyntaxError, RuntimeError, or an ImportError naming some other
            # transitive module) stays a warning.
            missing = getattr(exc, "name", None)
            top_pkg = module_path.split(".", 1)[0]
            not_installed = isinstance(exc, ModuleNotFoundError) and missing in (
                module_path,
                top_pkg,
            )
            log = logger.debug if not_installed else logger.warning
            log(
                f"entrypoint atom: failed to import {module_path} "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        manifest_obj: Any = getattr(module, "MANIFEST", None)
        if not isinstance(manifest_obj, ExtensionManifest):
            failure: BaseException = RuntimeError(
                f"entrypoint atom {module_path!r} is missing a module-level "
                f"MANIFEST: ExtensionManifest constant"
            )
            _LAST_DISCOVERY_FAILURES.append((module_path, failure))
            logger.warning(f"entrypoint atom: skipping {module_path} ({failure})")
            continue
        if manifest_obj.name != ep.name:
            failure = RuntimeError(
                f"entrypoint atom {module_path!r} has MANIFEST.name="
                f"{manifest_obj.name!r} which disagrees with entry-point name "
                f"{ep.name!r}"
            )
            _LAST_DISCOVERY_FAILURES.append((module_path, failure))
            logger.warning(f"entrypoint atom: skipping {module_path} ({failure})")
            continue
        entries[ep.name] = BuiltinEntry(
            name=ep.name,
            module_path=module_path,
            module=module,
            manifest=manifest_obj,
        )
    return entries


__all__ = [
    "BuiltinEntry",
    "CONTRIB_ATOM_MODULE_PREFIX",
    "HOME_ATOM_MODULE_PREFIX",
    "USER_ATOM_MODULE_PREFIX",
    "discover_builtin",
    "discover_by_role",
    "discover_contrib_atoms",
    "discover_entrypoint_atoms",
    "discover_home_atoms",
    "discover_user_atoms",
    "last_discovery_failures",
    "reset_cache",
]
