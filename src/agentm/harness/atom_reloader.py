"""Per-atom hot-reload state machine.

Pulled out of ``AgentSession.create`` (which had grown to ~640 lines around
this state) per the B1 refactor. The reloader owns:

* The loaded-atom registry (by module path and by manifest name).
* Per-owner handler unsubscribe lists and registration lists, so a reload
  can unwind everything an atom installed.
* The validation + atomic file-replace + rollback dance for swapping an
  atom's source.

It implements the :class:`_SessionGateway` shape and is plugged into
``_ExtensionAPIImpl`` as the ``gateway`` argument. Construction order in
``AgentSession.create``:

1. Build registries (``tools``, ``commands``, ...) and the bus.
2. Construct ``AtomReloader``, passing those registries by reference.
3. Define ``_make_api(owner)`` (which builds an ``_ExtensionAPIImpl`` with
   the reloader as its gateway).
4. Call ``reloader.set_api_factory(_make_api)`` so ``reload_atom`` can
   rebuild an API for the freshly-imported module.

Hard rule: this module imports only stdlib + ``agentm.core.*`` +
``agentm.harness.extension`` + ``agentm.extensions.*``. It never reaches
back into ``agentm.harness.session`` to avoid the circular imports the
old in-line implementation tolerated by being a closure.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from agentm.core._internal.catalog import (
    compute_atom_hash,
    freeze_current as freeze_atom_snapshot,
    is_constitution_path,
    source_path_for_hash,
)
from agentm.core.abi import EventBus, Tool
from agentm.extensions import ExtensionManifest
from agentm.extensions import discover as discover_mod
from agentm.extensions import validate as validate_mod
from agentm.harness.events import ApiRegisterEvent, ExtensionReloadEvent
from agentm.harness.extension import (
    AtomInfo,
    CommandSpec,
    ProviderConfig,
    ReloadResult,
    Renderer,
    _ExtensionAPIImpl,
    load_extension,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoadedAtom:
    """One installed atom's bookkeeping. Public so session.py can see it."""

    name: str
    module_path: str
    file_path: Path
    config: dict[str, Any]
    manifest: ExtensionManifest | None
    is_provider: bool = False


def _default_manifest(name: str) -> ExtensionManifest:
    return ExtensionManifest(
        name=name,
        description=f"Reload snapshot for {name}",
        registers=(),
    )


def _module_manifest(module: ModuleType) -> ExtensionManifest | None:
    manifest_obj = getattr(module, "MANIFEST", None)
    return manifest_obj if isinstance(manifest_obj, ExtensionManifest) else None


def _module_name(module_path: str, module: ModuleType) -> str:
    manifest_obj = _module_manifest(module)
    if manifest_obj is not None:
        return manifest_obj.name
    return module_path.rsplit(".", 1)[-1]


class AtomReloader:
    """State machine + ``_SessionGateway`` implementation."""

    def __init__(
        self,
        *,
        cwd: str,
        bus: EventBus,
        tools: list[Tool],
        commands: dict[str, CommandSpec],
        providers: dict[str, ProviderConfig],
        renderers: dict[str, Renderer],
        apis: dict[str, _ExtensionAPIImpl],
        on_provider_changed: Callable[[], None],
    ) -> None:
        self._cwd = cwd
        self._bus = bus
        self._tools = tools
        self._commands = commands
        self._providers = providers
        self._renderers = renderers
        self._apis = apis
        self._on_provider_changed = on_provider_changed

        self._loaded_by_module: dict[str, LoadedAtom] = {}
        self._loaded_by_name: dict[str, LoadedAtom] = {}
        self._handlers_by_atom: dict[str, list[Callable[[], None]]] = {}
        self._registrations_by_atom: dict[
            str, list[tuple[str, str, Any]]
        ] = {}
        self._command_owners: dict[str, str] = {}
        self._api_factory: Callable[[str], _ExtensionAPIImpl] | None = None

        bus.on("api_register", self._track_registration)

    # --- Lifecycle wiring --------------------------------------------------

    def set_api_factory(
        self, factory: Callable[[str], _ExtensionAPIImpl]
    ) -> None:
        """Inject the session's ``_make_api(owner)`` callable.

        Must be called before the first ``reload_atom`` call. We accept it
        post-construction because the API factory itself closes over this
        reloader (as the gateway), creating a chicken-and-egg cycle.
        """
        self._api_factory = factory

    # --- Public read accessors ---------------------------------------------

    @property
    def loaded_by_module(self) -> dict[str, LoadedAtom]:
        return self._loaded_by_module

    @property
    def loaded_by_name(self) -> dict[str, LoadedAtom]:
        return self._loaded_by_name

    @property
    def command_owners(self) -> dict[str, str]:
        return self._command_owners

    # --- Subscription tracking (called from ExtensionAPI.on / register) ----

    def wrap_api_on(self, api: _ExtensionAPIImpl, owner: str) -> None:
        """Wrap ``api.on`` so unsubscribes from ``owner`` are tracked.

        Each handler also gets a ``_agentm_obs_owner`` attribute so the
        ``observability`` atom can attribute event handlers to the
        installing extension.
        """
        original_on = api.on

        def tracked(channel: str, handler: Any) -> Any:
            try:
                setattr(handler, "_agentm_obs_owner", owner)
            except (AttributeError, TypeError):
                pass
            unsub = original_on(channel, handler)
            self._handlers_by_atom.setdefault(owner, []).append(unsub)
            return unsub

        api.on = tracked  # type: ignore[method-assign]

    def _track_registration(self, event: ApiRegisterEvent) -> None:
        self._registrations_by_atom.setdefault(event.extension, []).append(
            (event.kind, event.name, event.payload)
        )
        if event.kind == "command":
            self._command_owners[event.name] = event.extension

    # --- Atom registry mutation -------------------------------------------

    def record_loaded_atom(
        self,
        module_path: str,
        ext_cfg: dict[str, Any],
        *,
        is_provider: bool,
    ) -> None:
        module = importlib.import_module(module_path)
        module_file = getattr(module, "__file__", None)
        file_path = (
            Path(module_file).resolve() if module_file else Path(".")
        )
        manifest = _module_manifest(module)
        atom = LoadedAtom(
            name=_module_name(module_path, module),
            module_path=module_path,
            file_path=file_path,
            config=dict(ext_cfg),
            manifest=manifest,
            is_provider=is_provider,
        )
        self._loaded_by_module[module_path] = atom
        self._loaded_by_name[atom.name] = atom

    # --- Internal cleanup helpers -----------------------------------------

    def _remove_handlers(self, owner: str) -> None:
        for unsub in self._handlers_by_atom.pop(owner, []):
            unsub()

    def _remove_registrations(self, owner: str) -> None:
        for kind, name, payload in self._registrations_by_atom.pop(owner, []):
            if kind == "tool":
                self._tools[:] = [
                    tool for tool in self._tools if tool is not payload
                ]
            elif kind == "command":
                if self._commands.get(name) is payload:
                    self._commands.pop(name, None)
                if self._command_owners.get(name) == owner:
                    self._command_owners.pop(name, None)
            elif kind == "provider":
                if self._providers.get(name) is payload:
                    self._providers.pop(name, None)
                    self._on_provider_changed()
            elif kind == "renderer":
                if self._renderers.get(name) is payload:
                    self._renderers.pop(name, None)

    @staticmethod
    def _clear_module_bytecode(path: Path) -> None:
        cache_dir = path.parent / "__pycache__"
        if not cache_dir.exists():
            return
        for pyc in cache_dir.glob(f"{path.stem}*.pyc"):
            try:
                pyc.unlink()
            except OSError:
                pass

    def _validate_reload_source(
        self, name: str, module_path: str, new_source: str
    ) -> ExtensionManifest | None:
        with tempfile.TemporaryDirectory(
            prefix=f"agentm-reload-{name}-"
        ) as tmpdir:
            src_path = Path(tmpdir) / f"{name}.py"
            src_path.write_text(new_source, encoding="utf-8")
            spec = importlib.util.spec_from_file_location(
                f"_agentm_reload_validate_{name}_{uuid.uuid4().hex}",
                src_path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"could not build spec for {name!r}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            install = getattr(module, "install", None)
            if install is None or not callable(install):
                raise RuntimeError("missing callable 'install(api, config)'")
            sig = inspect.signature(install)
            positional = [
                p
                for p in sig.parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            if len(positional) < 2:
                raise RuntimeError(
                    f"'install' must accept (api, config); got {sig}"
                )

            manifest = _module_manifest(module)
            if manifest is None:
                return None
            if manifest.name != name:
                raise RuntimeError(
                    f"MANIFEST.name {manifest.name!r} does not match atom name {name!r}"
                )
            issues = validate_mod.validate_extension_contract(
                module_path=module_path,
                module=module,
                src_file=src_path,
                known_extension_names=set(self._loaded_by_name),
            )
            blocking = [issue for issue in issues if issue.severity == "error"]
            if blocking:
                raise RuntimeError(blocking[0].message)
            return manifest

    @staticmethod
    def _finish_install_sync(
        module_path: str,
        api: _ExtensionAPIImpl,
        ext_cfg: dict[str, Any],
    ) -> None:
        """Run ``install(api, config)`` synchronously, awaiting if needed.

        Reload happens from a synchronous call site (the agent's
        ``api.reload_atom`` doesn't ``await``), but ``install`` itself can
        be ``async``. We bounce into a fresh event loop on a worker thread
        rather than mixing loops on the calling thread.
        """
        result = load_extension(module_path, api, ext_cfg)
        if not inspect.isawaitable(result):
            return

        error: list[BaseException] = []

        def _runner() -> None:
            async def _await_result() -> None:
                await result

            try:
                asyncio.run(_await_result())
            except BaseException as exc:  # pragma: no cover - exercised in caller
                error.append(exc)

        thread = threading.Thread(
            target=_runner,
            name=f"agentm-reload-{module_path.rsplit('.', 1)[-1]}",
        )
        thread.start()
        thread.join()
        if error:
            raise error[0]

    # --- _SessionGateway protocol -----------------------------------------

    def reload_atom(
        self,
        name: str,
        new_source: str,
        *,
        agent_initiated: bool = True,
        rationale: str | None = None,
    ) -> ReloadResult:
        del rationale
        if self._api_factory is None:
            raise RuntimeError(
                "AtomReloader.set_api_factory must be called before reload_atom"
            )
        atom = self._loaded_by_name.get(name)
        if atom is None:
            return ReloadResult(
                ok=False,
                name=name,
                old_hash=None,
                new_hash=None,
                error=f"unknown atom {name!r}",
            )
        if is_constitution_path(str(atom.file_path)):
            return ReloadResult(
                ok=False,
                name=name,
                old_hash=None,
                new_hash=None,
                error=(
                    f"refusing to reload constitution layer path {atom.file_path}"
                ),
            )

        try:
            manifest = self._validate_reload_source(
                name, atom.module_path, new_source
            )
        except Exception as exc:  # noqa: BLE001
            return ReloadResult(
                ok=False,
                name=name,
                old_hash=None,
                new_hash=None,
                error=str(exc),
            )

        effective_manifest = (
            manifest or atom.manifest or _default_manifest(name)
        )
        if effective_manifest.tier == 2:
            logger.warning("tier-2 reload proceeds in MVP for %s", name)

        try:
            current_source = atom.file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return ReloadResult(
                ok=False,
                name=name,
                old_hash=None,
                new_hash=None,
                error=str(exc),
            )

        cwd_root = Path(self._cwd)
        old_hash = freeze_atom_snapshot(
            name,
            current_source,
            atom.manifest or _default_manifest(name),
            root=cwd_root,
        )
        snapshot_path = source_path_for_hash(name, old_hash, root=cwd_root)
        snapshot_source = snapshot_path.read_text(encoding="utf-8")
        new_hash = compute_atom_hash(new_source)

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".reload-{name}-",
            suffix=atom.file_path.suffix,
            dir=str(atom.file_path.parent),
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(new_source)

        previous_api = self._apis.get(atom.module_path)
        if previous_api is not None:
            previous_api.mark_stale()
        self._remove_handlers(atom.module_path)
        self._remove_registrations(atom.module_path)
        sys.modules.pop(atom.module_path, None)

        try:
            os.replace(tmp_name, atom.file_path)
            self._clear_module_bytecode(atom.file_path)
            importlib.invalidate_caches()
            self._finish_install_sync(
                atom.module_path,
                self._api_factory(atom.module_path),
                dict(atom.config),
            )
            self.record_loaded_atom(
                atom.module_path,
                atom.config,
                is_provider=atom.is_provider,
            )
            self._apis[atom.module_path]._owner_name = atom.module_path
            discover_mod.reset_cache()
            self._bus.emit_sync(
                "extension_reload",
                ExtensionReloadEvent(
                    name=name,
                    old_hash=old_hash,
                    new_hash=new_hash,
                    trigger="agent" if agent_initiated else "human",
                    tier=effective_manifest.tier,
                ),
            )
            return ReloadResult(
                ok=True,
                name=name,
                old_hash=old_hash,
                new_hash=new_hash,
            )
        except Exception as exc:  # noqa: BLE001
            try:
                rollback_fd, rollback_tmp_name = tempfile.mkstemp(
                    prefix=f".rollback-{name}-",
                    suffix=atom.file_path.suffix,
                    dir=str(atom.file_path.parent),
                )
                with os.fdopen(
                    rollback_fd, "w", encoding="utf-8"
                ) as rollback_handle:
                    rollback_handle.write(snapshot_source)
                os.replace(rollback_tmp_name, atom.file_path)
                sys.modules.pop(atom.module_path, None)
                self._clear_module_bytecode(atom.file_path)
                importlib.invalidate_caches()
                self._finish_install_sync(
                    atom.module_path,
                    self._api_factory(atom.module_path),
                    dict(atom.config),
                )
                self.record_loaded_atom(
                    atom.module_path,
                    atom.config,
                    is_provider=atom.is_provider,
                )
            except Exception as rollback_exc:  # noqa: BLE001
                self._loaded_by_module.pop(atom.module_path, None)
                self._loaded_by_name.pop(atom.name, None)
                self._apis.pop(atom.module_path, None)
                self._bus.emit_sync(
                    "extension_reload",
                    ExtensionReloadEvent(
                        name=name,
                        old_hash=old_hash,
                        new_hash=new_hash,
                        trigger="agent" if agent_initiated else "human",
                        tier=effective_manifest.tier,
                        error="rollback_failure",
                    ),
                )
                return ReloadResult(
                    ok=False,
                    name=name,
                    old_hash=old_hash,
                    new_hash=new_hash,
                    error=f"{exc}; rollback failed: {rollback_exc}",
                    rolled_back=True,
                )
            return ReloadResult(
                ok=False,
                name=name,
                old_hash=old_hash,
                new_hash=new_hash,
                error=str(exc),
                rolled_back=True,
            )

    def freeze_current(self, name: str) -> str:
        atom = self._loaded_by_name[name]
        source = atom.file_path.read_text(encoding="utf-8")
        return freeze_atom_snapshot(
            name,
            source,
            atom.manifest or _default_manifest(name),
            root=Path(self._cwd),
        )

    def list_atoms(self) -> list[AtomInfo]:
        out: list[AtomInfo] = []
        for atom in sorted(
            self._loaded_by_name.values(), key=lambda item: item.name
        ):
            current_hash = (
                compute_atom_hash(atom.file_path.read_text(encoding="utf-8"))
                if atom.file_path.exists()
                else None
            )
            manifest = atom.manifest or _default_manifest(atom.name)
            out.append(
                AtomInfo(
                    name=atom.name,
                    current_hash=current_hash,
                    tier=manifest.tier,
                    api_version=manifest.api_version,
                )
            )
        return out

    @staticmethod
    def is_constitution_path(path: str) -> bool:
        return is_constitution_path(path)


__all__ = ["AtomReloader", "LoadedAtom"]
