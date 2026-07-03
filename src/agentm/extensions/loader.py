"""Scenario loader.

A scenario is a directory under ``<cwd>/scenarios/<name>/`` containing a
``manifest.yaml`` and (optionally) one or more scenario-local atom modules.
The yaml lists extensions in declaration order; each entry is one of:

- ``module: <python.import.path>`` — references a builtin atom by its
  importable dotted path.
- ``local: <stem>`` — references ``<scenario_dir>/<stem>.py`` as a
  scenario-local atom; the loader registers it under the synthetic module
  name ``agentm._scenarios.<scenario>.<stem>`` so subsequent
  ``importlib.import_module`` calls resolve without re-execution.

Path resolution:

- An absolute path argument loads that file directly (or its
  ``manifest.yaml`` if a directory).
- A bare name is searched in this order:
    1. ``$AGENTM_PROJECT_ROOT/contrib/scenarios/<name>/manifest.yaml``
       when the env var is set (gives long-running daemons a stable
       anchor independent of process cwd).
    2. ``<cwd>/contrib/scenarios/<name>/manifest.yaml`` — the original
       behavior; works when ``agentm`` is invoked from a project root.
    3. ``~/.agentm/contrib/scenarios/<name>/manifest.yaml`` (or
       ``$AGENTM_HOME/contrib/scenarios/…``) — user-installed scenarios
       that work from pip-installed wheels without a source checkout.
    4. ``<agentm-package-root>/contrib/scenarios/<name>/manifest.yaml``
       — the worktree directory in editable installs, found by walking
       up from the ``agentm`` package source. Wheel installs without a
       sibling ``contrib/`` simply skip this candidate.
    5. ``agentm.scenarios.<name>/manifest.yaml`` packaged inside the wheel
       for portable built-in scenarios such as ``chatbot``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from agentm.extensions import ExtensionManifest
from agentm.core.abi import ExtensionLoadError

class ScenarioLoadError(ExtensionLoadError):
    """Raised when a scenario YAML cannot be resolved or validated."""


@dataclass(frozen=True, slots=True)
class ScenarioInfo:
    """Discoverable scenario summary for CLI/gateway presentation."""

    name: str
    manifest_path: str
    source: str
    description: str = ""


def _candidate_root_entries() -> list[tuple[str, Path]]:
    """Return search roots for ``contrib/scenarios/...`` resolution.

    Order matters: the first root that contains the requested manifest
    wins. ``AGENTM_PROJECT_ROOT`` lets long-running daemons (e.g. the
    Feishu gateway) anchor scenario lookup independent of process cwd;
    cwd is preserved as the canonical default so existing
    ``agentm --scenario X`` invocations from a project root keep
    working unchanged. The agentm-package-relative root rescues
    editable installs whose worktree contains a sibling ``contrib/``
    that the user didn't ``cd`` into — common when invoking via a
    console-script entry point.
    """

    roots: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def add(source: str, root: Path) -> None:
        text = str(root)
        if text in seen:
            return
        roots.append((source, root))
        seen.add(text)

    project_root_env = os.environ.get("AGENTM_PROJECT_ROOT")
    if project_root_env:
        add("project", Path(project_root_env))
    add("cwd", Path(os.getcwd()))
    # Home contrib directory: ~/.agentm/ (or $AGENTM_HOME/).
    # Scenarios installed here are resolved as
    # <home>/contrib/scenarios/<name>/manifest.yaml.
    try:
        from agentm.core.lib import agentm_home_dir

        home = agentm_home_dir()
        if (home / "contrib").is_dir():
            add("home", home)
    except Exception as exc:  # noqa: BLE001
        # Home-contrib discovery is a best-effort scenario root; skip it.
        logger.debug("scenario loader: home contrib root discovery failed: {}", exc)
    try:
        import agentm  # local import to dodge circular at module load time

        package_dir = Path(agentm.__file__).parent  # .../src/agentm
        # An editable install puts agentm under <worktree>/src/agentm,
        # so the worktree root is ``parent.parent`` (skip the ``src``).
        # We don't assume the layout — just walk up looking for a
        # ``contrib`` sibling, capping at ``_PACKAGE_WALK_DEPTH`` levels
        # to avoid escaping into ancestor directories on machines where
        # the package lives deeper than expected. ``AGENTM_PROJECT_ROOT``
        # is the canonical anchor for production deployments; this walk
        # is a development-time best-effort fallback.
        walker = package_dir
        for _ in range(_PACKAGE_WALK_DEPTH):
            if (walker / "contrib").is_dir():
                add("checkout", walker)
                break
            walker = walker.parent
    except Exception as exc:  # noqa: BLE001 — best-effort fallback
        # Package-relative contrib walk is a dev-time fallback; skip on failure.
        logger.debug("scenario loader: package-relative root walk failed: {}", exc)
    return roots


def _candidate_roots() -> list[Path]:
    return [root for _source, root in _candidate_root_entries()]

# Conservative cap: src/agentm → src → worktree-root covers the editable
# install case in two steps; doubled to absorb wheel layouts that nest
# the package one level deeper. Bump only when a real layout demands it.
_PACKAGE_WALK_DEPTH = 4

def _resolve_scenario_manifest(name_or_path: str, relative: Path) -> Path:
    """Find ``relative`` under any of :func:`_candidate_roots`.

    Raises :class:`ScenarioLoadError` listing every candidate that was
    tried so the operator knows exactly which paths were searched.
    """

    tried: list[Path] = []
    for root in _candidate_roots():
        candidate = root / relative
        if candidate.is_file():
            return candidate
        tried.append(candidate)
    raise ScenarioLoadError(
        name_or_path,
        FileNotFoundError(
            "scenario manifest not found. Tried: " + ", ".join(str(p) for p in tried)
        ),
    )

def _resolve_scenario_entrypoint(name: str) -> Path | None:
    """Resolve a scenario *name* to an on-disk ``manifest.yaml`` via an
    installed package's ``agentm.scenarios`` entry point.

    Convention (mirrors the ``agentm.atoms`` group):

        [project.entry-points."agentm.scenarios"]
        my_scenario = "my_pkg.scenarios.my_scenario"   # package holding manifest.yaml

    The entry-point *name* is the scenario name; the *value* is an importable
    package whose directory contains ``manifest.yaml`` (and whose final path
    component equals the scenario name, satisfying the dir-name == scenario-name
    contract). This lets a scenario ship as a pip package and resolve by name
    from a wheel — no source checkout, no ``AGENTM_PROJECT_ROOT``, no path.

    Returns the manifest ``Path`` (filesystem-backed; scenarios need a real
    directory for sibling ``local:`` atoms and import roots), or ``None`` when
    no entry point matches or it cannot be made concrete.
    """

    try:
        from importlib.metadata import entry_points
        from importlib.resources import files
    except Exception as exc:  # noqa: BLE001 — importlib always present on 3.12; defensive
        logger.debug("scenario EP lookup: importlib import failed: {}", exc)
        return None

    try:
        eps = entry_points(group="agentm.scenarios")
    except Exception as exc:  # noqa: BLE001 — never let discovery break loading
        logger.debug("scenario EP lookup: entry_points() call failed: {}", exc)
        return None

    for ep in eps:
        if ep.name != name:
            continue
        try:
            manifest = files(ep.value) / "manifest.yaml"
        except (ModuleNotFoundError, TypeError) as exc:
            logger.debug("scenario loader: entry-point {!r} not resolvable: {}", ep.name, exc)
            continue
        # Scenarios require a real on-disk directory (sibling ``local:`` atoms,
        # import roots), so demand a filesystem-backed resource: os.fspath()
        # succeeds for unpacked installs (the only kind AgentM supports) and
        # raises for zipped ones, which we skip rather than hand back an
        # unusable path. (No ``as_file`` — its temp extraction would be
        # deleted on context exit, and a single file can't satisfy a
        # scenario's sibling-file needs anyway.)
        try:
            # ``Traversable`` is not statically declared as ``os.PathLike``
            # but unpacked-install implementations (``PosixPath`` /
            # ``WindowsPath``) satisfy it at runtime — the ``except
            # TypeError`` above is the explicit fallback for the zipped
            # case where ``os.fspath`` legitimately refuses.
            concrete = Path(os.fspath(manifest))  # type: ignore[call-overload]
        except TypeError as exc:
            logger.debug("scenario loader: {!r} manifest not filesystem-backed: {}", ep.name, exc)
            continue
        if concrete.is_file():
            return concrete
    return None


def _resolve_packaged_scenario(name: str) -> Path | None:
    """Resolve a built-in portable scenario bundled inside the ``agentm`` wheel.

    These are fallbacks for wheel installs that do not have the source checkout's
    ``contrib/scenarios`` tree. User-installed scenarios under ``~/.agentm`` are
    resolved before this fallback, so local customization still wins.
    """

    if not name.isidentifier():
        return None
    try:
        from importlib.resources import files
    except Exception as exc:  # noqa: BLE001 — importlib.resources is stdlib; defensive
        logger.debug("packaged scenario lookup: importlib.resources import failed: {}", exc)
        return None
    try:
        manifest = files(f"agentm.scenarios.{name}") / "manifest.yaml"
    except (ModuleNotFoundError, TypeError) as exc:
        logger.debug("packaged scenario {!r}: module not found: {}", name, exc)
        return None
    try:
        concrete = Path(os.fspath(manifest))  # type: ignore[call-overload]
    except TypeError as exc:
        logger.debug("packaged scenario {!r}: not filesystem-backed: {}", name, exc)
        return None
    return concrete if concrete.is_file() else None


def validate_scenario(name_or_path: str) -> None:
    """Fail fast if ``name_or_path`` cannot be resolved and loaded."""

    load_scenario(name_or_path)


def list_scenarios() -> list[ScenarioInfo]:
    """List scenario manifests discoverable from the current process.

    Listing intentionally performs only cheap manifest summary reads. Full
    dependency/import validation remains the job of :func:`load_scenario` when a
    scenario is actually selected.
    """

    discovered: list[ScenarioInfo] = []
    seen: set[str] = set()

    def add(path: Path | None, *, source: str, name_hint: str) -> None:
        if path is None or not path.is_file():
            return
        name, description = _scenario_manifest_summary(path, name_hint)
        if name in seen:
            return
        discovered.append(
            ScenarioInfo(
                name=name,
                manifest_path=str(path),
                source=source,
                description=description,
            )
        )
        seen.add(name)

    for ep_name in sorted(_scenario_entrypoint_names()):
        add(
            _resolve_scenario_entrypoint(ep_name),
            source="entry_point",
            name_hint=ep_name,
        )

    for source, root in _candidate_root_entries():
        for manifest_path, name_hint in _iter_contrib_scenario_manifests(root):
            add(manifest_path, source=source, name_hint=name_hint)

    packaged_root = _packaged_scenario_root()
    if packaged_root is not None:
        for manifest_path, name_hint in _iter_packaged_scenario_manifests(packaged_root):
            add(manifest_path, source="packaged", name_hint=name_hint)

    return sorted(discovered, key=lambda item: item.name)


def _scenario_entrypoint_names() -> list[str]:
    try:
        from importlib.metadata import entry_points
    except Exception as exc:  # noqa: BLE001
        logger.debug("scenario list: importlib.metadata unavailable: {}", exc)
        return []
    try:
        return [ep.name for ep in entry_points(group="agentm.scenarios")]
    except Exception as exc:  # noqa: BLE001
        logger.debug("scenario list: entry point discovery failed: {}", exc)
        return []


def _iter_contrib_scenario_manifests(root: Path) -> list[tuple[Path, str]]:
    scenarios_root = root / "contrib" / "scenarios"
    if not scenarios_root.is_dir():
        return []
    manifests: list[tuple[Path, str]] = []
    for scenario_dir in sorted(scenarios_root.iterdir(), key=lambda item: item.name):
        if not scenario_dir.is_dir():
            continue
        manifests.append((scenario_dir / "manifest.yaml", scenario_dir.name))
        for variant_path in sorted(scenario_dir.glob("manifest.*.yaml")):
            variant = variant_path.name.removeprefix("manifest.").removesuffix(".yaml")
            if variant:
                manifests.append((variant_path, f"{scenario_dir.name}:{variant}"))
    return manifests


def _packaged_scenario_root() -> Path | None:
    try:
        import agentm
    except Exception as exc:  # noqa: BLE001
        logger.debug("scenario list: package import failed: {}", exc)
        return None
    root = Path(agentm.__file__).parent / "scenarios"
    return root if root.is_dir() else None


def _iter_packaged_scenario_manifests(root: Path) -> list[tuple[Path, str]]:
    manifests: list[tuple[Path, str]] = []
    for scenario_dir in sorted(root.iterdir(), key=lambda item: item.name):
        if scenario_dir.is_dir():
            manifests.append((scenario_dir / "manifest.yaml", scenario_dir.name))
    return manifests


def _scenario_manifest_summary(path: Path, name_hint: str) -> tuple[str, str]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("scenario list: failed to read {}: {}", path, exc)
        return name_hint, ""
    if not isinstance(payload, dict):
        return name_hint, ""
    raw_name = payload.get("name")
    name = raw_name if isinstance(raw_name, str) and raw_name else name_hint
    raw_description = payload.get("description")
    description = raw_description.strip() if isinstance(raw_description, str) else ""
    return name, description


def load_scenario(
    name_or_path: str,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    """Resolve and parse a scenario manifest.

    Returns ``(extensions, meta)`` where *extensions* is a list of
    ``(module_path, config)`` pairs in declaration order and *meta*
    carries top-level scenario metadata: ``scenario_dir``, ``task_class``,
    ``promotion``, etc.
    """

    candidate = Path(name_or_path)
    if candidate.is_absolute():
        manifest_path = (
            candidate / "manifest.yaml" if candidate.is_dir() else candidate
        )
        if not manifest_path.is_file():
            raise ScenarioLoadError(
                str(manifest_path),
                FileNotFoundError(
                    f"scenario manifest not found at {manifest_path}"
                ),
            )
    elif candidate.exists() or (candidate / "manifest.yaml").exists():
        manifest_path = (
            candidate / "manifest.yaml" if candidate.is_dir() else candidate
        )
        if not manifest_path.is_file():
            raise ScenarioLoadError(
                str(manifest_path),
                FileNotFoundError(
                    f"scenario manifest not found at {manifest_path}"
                ),
            )
    else:
        # Prefer an installed plugin's ``agentm.scenarios`` entry point (the
        # canonical "publish a scenario as a pip package" path — works from a
        # wheel with no source checkout). Variants (``name:variant``) remain
        # path-only. Fall back to the on-disk contrib/scenarios roots so
        # in-repo scenarios keep resolving unchanged.
        ep_manifest = (
            _resolve_scenario_entrypoint(name_or_path)
            if ":" not in name_or_path
            else None
        )
        if ep_manifest is not None:
            manifest_path = ep_manifest
        else:
            if ":" in name_or_path:
                base, _, variant = name_or_path.rpartition(":")
                if not base or not variant:
                    raise ScenarioLoadError(
                        name_or_path,
                        ValueError(
                            f"scenario variant must be '<name>:<variant>'; got {name_or_path!r}"
                        ),
                    )
                relative = Path("contrib") / "scenarios" / base / f"manifest.{variant}.yaml"
            else:
                relative = (
                    Path("contrib") / "scenarios" / name_or_path / "manifest.yaml"
                )
            try:
                manifest_path = _resolve_scenario_manifest(name_or_path, relative)
            except ScenarioLoadError:
                pkg_manifest = (
                    _resolve_packaged_scenario(name_or_path)
                    if ":" not in name_or_path
                    else None
                )
                if pkg_manifest is None:
                    raise
                manifest_path = pkg_manifest

    extensions = _load_from_path(manifest_path)
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("manifest YAML parse failed for {}: {}", manifest_path, exc)
        payload = None
    meta: dict[str, Any] = {"scenario_dir": str(manifest_path.parent)}
    if isinstance(payload, dict):
        for key in ("task_class", "promotion"):
            if key in payload:
                meta[key] = payload[key]
    return extensions, meta

def _load_from_path(path: Path) -> list[tuple[str, dict[str, Any]]]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ScenarioLoadError(str(path), exc) from exc

    scenario_dir = path.parent
    _ensure_scenario_import_roots(scenario_dir)
    scenario_name = _resolve_scenario_name(payload, scenario_dir, source=str(path))
    return _parse_extensions(
        payload,
        source=str(path),
        scenario_dir=scenario_dir,
        scenario_name=scenario_name,
    )

def _resolve_scenario_name(
    payload: Any,
    scenario_dir: Path,
    *,
    source: str,
) -> str:
    if not isinstance(payload, dict):
        raise ScenarioLoadError(source, ValueError("scenario must be a mapping"))

    declared = payload.get("name")
    dir_name = scenario_dir.name
    if declared is None:
        return dir_name
    if not isinstance(declared, str) or not declared:
        raise ScenarioLoadError(
            source, ValueError("scenario 'name' must be a non-empty string")
        )
    # Variant manifests carry a ``<dir>:<variant>`` name (e.g. ``rca:baseline``
    # in ``contrib/scenarios/rca/manifest.baseline.yaml``). Accept either the
    # bare directory name or that ``<dir>:<variant>`` form.
    base_name = declared.split(":", 1)[0] if ":" in declared else declared
    if base_name != dir_name:
        raise ScenarioLoadError(
            source,
            ValueError(
                f"scenario name {declared!r} does not match directory name "
                f"{dir_name!r}"
            ),
        )
    return declared

def _parse_extensions(
    payload: Any,
    *,
    source: str,
    scenario_dir: Path,
    scenario_name: str,
) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ScenarioLoadError(source, ValueError("scenario must be a mapping"))

    raw_extensions: list[Any] = []

    # --- includes: resolve overlay files and prepend their extensions ---
    includes = payload.get("includes")
    if includes is not None:
        if not isinstance(includes, list):
            raise ScenarioLoadError(
                source, ValueError("'includes' must be a list of relative paths")
            )
        for inc_index, inc_path in enumerate(includes):
            if not isinstance(inc_path, str) or not inc_path:
                raise ScenarioLoadError(
                    source,
                    ValueError(f"includes[{inc_index}]: must be a non-empty string"),
                )
            overlay_path = scenario_dir / inc_path
            if not overlay_path.is_file():
                raise ScenarioLoadError(
                    source,
                    FileNotFoundError(
                        f"includes[{inc_index}]: overlay not found at {overlay_path}"
                    ),
                )
            try:
                overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise ScenarioLoadError(
                    source,
                    ValueError(f"includes[{inc_index}]: failed to parse {overlay_path}: {exc}"),
                ) from exc
            if not isinstance(overlay, dict) or not isinstance(
                overlay.get("extensions"), list
            ):
                raise ScenarioLoadError(
                    source,
                    ValueError(
                        f"includes[{inc_index}]: overlay {overlay_path} must contain "
                        "an 'extensions' list"
                    ),
                )
            raw_extensions.extend(overlay["extensions"])

    declared = payload.get("extensions")
    if declared is not None:
        if not isinstance(declared, list):
            raise ScenarioLoadError(source, ValueError("'extensions' must be a list"))
        raw_extensions.extend(declared)

    if not raw_extensions:
        raise ScenarioLoadError(
            source, ValueError("scenario must declare extensions (via 'extensions' or 'includes')")
        )

    extensions: list[tuple[str, dict[str, Any]]] = []
    for index, item in enumerate(raw_extensions):
        if not isinstance(item, dict):
            raise ScenarioLoadError(
                source,
                ValueError(_entry_error(index, "must be a mapping")),
            )

        config = item.get("config", {})
        if not isinstance(config, dict):
            raise ScenarioLoadError(
                source,
                ValueError(_entry_error(index, "'config' must be a mapping")),
            )

        has_module = "module" in item
        has_local = "local" in item
        if has_module and has_local:
            raise ScenarioLoadError(
                source,
                ValueError(
                    _entry_error(
                        index,
                        "entry must declare exactly one of 'module' or 'local'",
                    )
                ),
            )

        if has_module:
            module = item["module"]
            if not isinstance(module, str) or not module:
                raise ScenarioLoadError(
                    source,
                    ValueError(_entry_error(index, "missing string 'module'")),
                )
            _validate_module(source, index, module)
            extensions.append((module, dict(config)))
        elif has_local:
            stem = item["local"]
            if not isinstance(stem, str) or not stem:
                raise ScenarioLoadError(
                    source,
                    ValueError(_entry_error(index, "missing string 'local'")),
                )
            synthetic = _register_local(
                source=source,
                index=index,
                scenario_dir=scenario_dir,
                scenario_name=scenario_name,
                stem=stem,
            )
            extensions.append((synthetic, dict(config)))
        else:
            raise ScenarioLoadError(
                source,
                ValueError(
                    _entry_error(
                        index,
                        "entry must declare exactly one of 'module' or 'local'",
                    )
                ),
            )

    return sort_extensions_by_requires(extensions, source=source)

# Atoms the session factory auto-mounts as floor atoms regardless of whether a
# scenario lists them (see ``session_factory.ensure_floor_atom``). An atom may
# ``requires`` one of these without the scenario manifest redundantly listing
# it — requires-validation runs here on the manifest list, before the factory
# injects the floor atoms, so a required floor atom would otherwise read as
# "not loaded". Names match the role fulfillers' ``MANIFEST.name``.
_FLOOR_ATOM_NAMES: frozenset[str] = frozenset(
    {"prompt_templates", "compaction_prompts", "slash_commands", "system_prompt",
     "retry_policy"}
)

def sort_extensions_by_requires(
    extensions: list[tuple[str, dict[str, Any]]],
    *,
    source: str = "<extensions>",
) -> list[tuple[str, dict[str, Any]]]:
    manifests: dict[str, ExtensionManifest] = {}
    name_by_module: dict[str, str] = {}
    entries_by_name: dict[str, tuple[str, dict[str, Any]]] = {}

    for module_path, config in extensions:
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            # A configured extension that fails to import is silently dropped
            # from the active set otherwise — warn so the omission is visible.
            logger.warning("scenario loader: extension {} failed to import, skipping: {}", module_path, exc)
            continue
        manifest = getattr(module, "MANIFEST", None)
        if not isinstance(manifest, ExtensionManifest):
            continue
        existing = entries_by_name.get(manifest.name)
        if existing is not None:
            raise ScenarioLoadError(
                source,
                ValueError(
                    f"extension {manifest.name!r} is loaded more than once; "
                    "duplicate extension entries are not supported"
                ),
            )
        manifests[manifest.name] = manifest
        name_by_module[module_path] = manifest.name
        entries_by_name[manifest.name] = (module_path, config)

    for name, manifest in manifests.items():
        for dep in manifest.requires:
            if dep not in entries_by_name and dep not in _FLOOR_ATOM_NAMES:
                raise ScenarioLoadError(
                    source,
                    ValueError(
                        f"extension {name!r} requires {dep!r}, but {dep!r} "
                        "is not loaded"
                    ),
                )

    sorted_entries: list[tuple[str, dict[str, Any]]] = []
    temporary: set[str] = set()
    permanent: set[str] = set()
    emitted_modules: set[str] = set()

    def visit(name: str) -> None:
        if name in permanent:
            return
        if name in temporary:
            raise ScenarioLoadError(
                source, ValueError(f"extension dependency cycle involving {name!r}")
            )
        temporary.add(name)
        for dep in manifests[name].requires:
            # Floor-atom deps (and any dep not in this list) are mounted and
            # ordered separately by the factory — skip them here.
            if dep in manifests:
                visit(dep)
        temporary.remove(name)
        permanent.add(name)
        entry = entries_by_name[name]
        if entry[0] not in emitted_modules:
            sorted_entries.append(entry)
            emitted_modules.add(entry[0])

    for entry in extensions:
        module_path = entry[0]
        module_name = name_by_module.get(module_path)
        if module_name is None:
            if module_path not in emitted_modules:
                sorted_entries.append(entry)
                emitted_modules.add(module_path)
            continue
        visit(module_name)

    return sorted_entries

def _validate_module(source: str, index: int, module: str) -> None:
    spec = importlib.util.find_spec(module)
    if spec is None:
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(index, f"module {module!r} is not importable")
            ),
        )

    # Prefer a cheap existence check here: scenario manifests should validate
    # even when optional third-party runtime dependencies (for example DuckDB
    # in the RCA package) are not installed in the current test environment.
    # If the module has already been imported elsewhere, keep the stronger
    # ``install()`` assertion.
    loaded = sys.modules.get(module)
    if loaded is not None and not callable(getattr(loaded, "install", None)):
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(index, f"module {module!r} does not export install()")
            ),
        )

def _find_project_root(start: Path) -> Path | None:
    """Locate the topmost project root above ``start``.

    A scenario directory may itself be a workspace member with its own
    ``pyproject.toml`` (e.g. ``contrib/scenarios/rca/pyproject.toml``),
    so the *first* marker hit when ascending is not the project root —
    it's the nested member. We want the outermost project so
    cross-package references like ``contrib.extensions.<name>`` resolve.

    Strategy: prefer the directory containing ``.git`` if any ancestor
    has one (the canonical monorepo boundary); otherwise return the
    highest ancestor that still carries a Python project marker.
    """

    project_markers = ("pyproject.toml", "setup.py", "setup.cfg")
    highest_with_marker: Path | None = None
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
        if any((candidate / m).is_file() for m in project_markers):
            highest_with_marker = candidate
    return highest_with_marker

def _prepend_sys_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

def _ensure_scenario_import_roots(scenario_dir: Path) -> None:
    """Make in-tree scenario packages importable during manifest load.

    Scenario manifests reference modules in two ways:

    1. ``<scenario_dir>/src/<pkg>`` — editable-style scenario packages
       (e.g. ``agentm_rca.tools.duckdb_sql`` under
       ``contrib/scenarios/rca/src/agentm_rca/``).
    2. ``<project_root>/<pkg>`` — peer packages from the same checkout
       (e.g. ``contrib.extensions.rcabench_contract`` under
       ``contrib/extensions/``).

    Entry-point scripts launch with ``sys.path[0]`` pointing at the
    venv bin dir rather than the project root, so neither path is on
    ``sys.path`` by default and ``import`` fails — silently, because
    the caller in ``session_factory`` swallows the manifest load error
    and the session ends up with zero tools. We surface both roots
    here so the imports resolve regardless of how the process was
    launched. Project root is located via the standard Python project
    markers rather than hard-coded directory names.
    """

    src_root = scenario_dir / "src"
    if src_root.is_dir():
        _prepend_sys_path(src_root)

    project_root = _find_project_root(scenario_dir)
    if project_root is not None:
        _prepend_sys_path(project_root)

def _register_local(
    *,
    source: str,
    index: int,
    scenario_dir: Path,
    scenario_name: str,
    stem: str,
) -> str:
    """Resolve ``<scenario_dir>/<stem>.py`` and register it under a
    synthetic module name. Returns the synthetic name."""

    synthetic = f"agentm._scenarios.{scenario_name}.{stem}"
    if synthetic in sys.modules:
        # Idempotent: a previous load already registered this atom.
        return synthetic

    file_path = scenario_dir / f"{stem}.py"
    if not file_path.is_file():
        raise ScenarioLoadError(
            source,
            FileNotFoundError(
                _entry_error(
                    index,
                    f"local atom file not found at {file_path}",
                )
            ),
        )

    spec = importlib.util.spec_from_file_location(synthetic, file_path)
    if spec is None or spec.loader is None:
        raise ScenarioLoadError(
            source,
            RuntimeError(
                _entry_error(index, f"could not build import spec for {file_path}")
            ),
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[synthetic] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        sys.modules.pop(synthetic, None)
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(index, f"failed to execute {file_path}: {exc}")
            ),
        ) from exc

    manifest_obj = getattr(module, "MANIFEST", None)
    if not isinstance(manifest_obj, ExtensionManifest):
        sys.modules.pop(synthetic, None)
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(
                    index,
                    f"local atom {stem!r} is missing a module-level "
                    "MANIFEST: ExtensionManifest constant",
                )
            ),
        )
    if manifest_obj.name != stem:
        sys.modules.pop(synthetic, None)
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(
                    index,
                    f"local atom {stem!r} has MANIFEST.name="
                    f"{manifest_obj.name!r}; must equal the file stem",
                )
            ),
        )
    if not callable(getattr(module, "install", None)):
        sys.modules.pop(synthetic, None)
        raise ScenarioLoadError(
            source,
            ValueError(
                _entry_error(
                    index,
                    f"local atom {stem!r} does not export install()",
                )
            ),
        )
    return synthetic

def _entry_error(index: int, detail: str) -> str:
    return f"extensions[{index}] {detail}"

__all__ = [
    "ScenarioInfo",
    "ScenarioLoadError",
    "list_scenarios",
    "load_scenario",
    "sort_extensions_by_requires",
    "validate_scenario",
]
