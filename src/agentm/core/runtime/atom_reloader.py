"""Per-atom hot-reload state machine and ``_SessionGateway`` implementation."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
from loguru import logger
import os
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Literal

from agentm.core._internal.catalog.hashing import compute_atom_hash
from agentm.core._internal.catalog.manifest import is_constitution_path
from agentm.core.runtime.catalog import _layout
from agentm.core.abi import BusPriority, EventBus, Tool
from agentm.core.abi.manifest import ExtensionManifest
from agentm.core.abi.events import (
    ApiRegisterEvent,
    BeforeInstallAtomEvent,
    BeforeUnloadAtomEvent,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
    ExtensionUnloadEvent,
)
from agentm.core.runtime.extension import (
    AtomInfo,
    CommandSpec,
    InstallAtomResult,
    ProviderConfig,
    ReloadResult,
    Renderer,
    UnloadAtomResult,
    _ExtensionAPIImpl,
    load_extension,
)
from agentm.core.runtime.resource_writer import ResourceWriter

_SCENARIO_MODULE_PREFIX = "agentm._scenarios."


@dataclass(slots=True)
class LoadedAtom:
    """One installed atom's bookkeeping. Public so session.py can see it."""

    name: str
    module_path: str
    file_path: Path
    config: dict[str, Any]
    manifest: ExtensionManifest | None
    is_provider: bool = False
    import_kind: Literal["module", "synthetic"] = "module"


@dataclass(slots=True, frozen=True)
class _ReloadSnapshot:
    """Pre-reload capture for transactional rollback."""

    loaded_by_module: dict[str, LoadedAtom]
    loaded_by_name: dict[str, LoadedAtom]
    handlers_by_atom: dict[str, list[Callable[[], None]]]
    registrations_by_atom: dict[str, list[tuple[str, str, Any]]]
    owners_by_kind: dict[str, dict[str, str]]
    tools: list[Tool]
    commands: dict[str, CommandSpec]
    providers: dict[str, ProviderConfig]
    renderers: dict[str, Renderer]
    apis: dict[str, _ExtensionAPIImpl]
    sys_module: ModuleType | None
    module_path: str
    bus_channels: dict[str, list[Any]]


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
        resource_writer: ResourceWriter,
        bus: EventBus,
        tools: list[Tool],
        commands: dict[str, CommandSpec],
        providers: dict[str, ProviderConfig],
        renderers: dict[str, Renderer],
        apis: dict[str, _ExtensionAPIImpl],
        on_provider_changed: Callable[[], None],
    ) -> None:
        self._cwd = cwd
        self._resource_writer = resource_writer
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
        self._registrations_by_atom: dict[str, list[tuple[str, str, Any]]] = {}
        self.owners_by_kind: dict[str, dict[str, str]] = {}
        self._subscription_unsubs: list[Callable[[], None]] = [
            bus.on(ApiRegisterEvent.CHANNEL, self._track_registration)
        ]
        self._api_factory: Callable[[str], _ExtensionAPIImpl] | None = None
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._sync_loop_thread: threading.Thread | None = None

    # --- Lifecycle wiring --------------------------------------------------

    def set_api_factory(self, factory: Callable[[str], _ExtensionAPIImpl]) -> None:
        """Must be called before the first ``reload_atom``."""
        self._api_factory = factory

    # --- Public read accessors ---------------------------------------------

    @property
    def loaded_by_module(self) -> dict[str, LoadedAtom]:
        return self._loaded_by_module

    @property
    def loaded_by_name(self) -> dict[str, LoadedAtom]:
        return self._loaded_by_name

    def shutdown(self) -> None:
        for unsub in self._subscription_unsubs:
            unsub()
        self._subscription_unsubs.clear()
        if self._sync_loop is not None:
            self._sync_loop.call_soon_threadsafe(self._sync_loop.stop)
        if self._sync_loop_thread is not None:
            self._sync_loop_thread.join(timeout=1)
        self._sync_loop = None
        self._sync_loop_thread = None

    # --- Subscription tracking (called from ExtensionAPI.on/add_observer) ---

    def wrap_api_on(self, api: _ExtensionAPIImpl, owner: str) -> None:
        original_on = api.on
        original_add_observer = api.add_observer

        def tracked(
            channel: str,
            handler: Any,
            *,
            priority: int = BusPriority.NORMAL,
        ) -> Any:
            try:
                setattr(handler, "_agentm_obs_owner", owner)
            except (AttributeError, TypeError):
                pass
            unsub = original_on(channel, handler, priority=priority)
            self._handlers_by_atom.setdefault(owner, []).append(unsub)
            return unsub

        def tracked_observer(callback: Any) -> Any:
            unsub = original_add_observer(callback)
            self._handlers_by_atom.setdefault(owner, []).append(unsub)
            return unsub

        api.on = tracked  # type: ignore[method-assign]
        api.add_observer = tracked_observer  # type: ignore[method-assign]

    def _track_registration(self, event: ApiRegisterEvent) -> None:
        self._registrations_by_atom.setdefault(event.extension, []).append(
            (event.kind, event.name, event.payload)
        )
        self.owners_by_kind.setdefault(event.kind, {})[event.name] = event.extension

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
        file_path = Path(module_file).resolve() if module_file else Path(".")
        manifest = _module_manifest(module)
        from agentm.extensions import discover as discover_mod

        import_kind: Literal["module", "synthetic"] = (
            "synthetic"
            if module_path.startswith(discover_mod.USER_ATOM_MODULE_PREFIX)
            or module_path.startswith(discover_mod.CONTRIB_ATOM_MODULE_PREFIX)
            or module_path.startswith(discover_mod.HOME_ATOM_MODULE_PREFIX)
            else "module"
        )
        atom = LoadedAtom(
            name=_module_name(module_path, module),
            module_path=module_path,
            file_path=file_path,
            config=dict(ext_cfg),
            manifest=manifest,
            is_provider=is_provider,
            import_kind=import_kind,
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
                self._tools[:] = [tool for tool in self._tools if tool is not payload]
            elif kind == "command":
                if self._commands.get(name) is payload:
                    self._commands.pop(name, None)
            elif kind == "provider":
                if self._providers.get(name) is payload:
                    self._providers.pop(name, None)
                    self._on_provider_changed()
            elif kind == "renderer":
                if self._renderers.get(name) is payload:
                    self._renderers.pop(name, None)
            if self.owners_by_kind.setdefault(kind, {}).get(name) == owner:
                self.owners_by_kind[kind].pop(name, None)

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
        manifest = self._validate_atom_source(
            name,
            module_path,
            new_source,
            tag="reload",
            require_manifest=True,
        )
        return manifest

    def _validate_atom_source(
        self,
        name: str,
        module_path: str,
        new_source: str,
        *,
        tag: str,
        require_manifest: bool,
    ) -> ExtensionManifest | None:
        """Validate ``new_source`` in a throwaway temp dir; return its manifest."""
        with tempfile.TemporaryDirectory(prefix=f"agentm-{tag}-{name}-") as tmpdir:
            src_path = Path(tmpdir) / f"{name}.py"
            src_path.write_text(new_source, encoding="utf-8")
            spec = importlib.util.spec_from_file_location(
                f"_agentm_{tag}_validate_{name}_{uuid.uuid4().hex}",
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
                raise RuntimeError(f"'install' must accept (api, config); got {sig}")

            manifest = _module_manifest(module)
            if manifest is None and require_manifest:
                return None
            if manifest is not None and manifest.name != name:
                raise RuntimeError(
                    f"MANIFEST.name {manifest.name!r} does not match atom name {name!r}"
                )
            from agentm.extensions import validate as validate_mod

            known = set(self._loaded_by_name)
            if not require_manifest:
                known = known | {name}
            issues = validate_mod.validate_extension_contract(
                module_path=module_path,
                module=module,
                src_file=src_path,
                known_extension_names=known,
            )
            blocking = [issue for issue in issues if issue.severity == "error"]
            if blocking:
                raise RuntimeError(blocking[0].message)
            return manifest

    @staticmethod
    async def _finish_install(
        module_path: str,
        api: _ExtensionAPIImpl,
        ext_cfg: dict[str, Any],
    ) -> None:
        # validate=False: the reloader validates atom source separately via
        # _validate_reload_source / _validate_install_source before reaching
        # here, so skip the load-time §11 check to avoid double work.
        result = load_extension(module_path, api, ext_cfg, validate=False)
        if inspect.isawaitable(result):
            await result

    def _run_api_boundary(self, coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        loop = self._ensure_sync_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def _ensure_sync_loop(self) -> asyncio.AbstractEventLoop:
        if self._sync_loop is not None and self._sync_loop.is_running():
            return self._sync_loop

        ready = threading.Event()

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._sync_loop = loop
            ready.set()
            loop.run_forever()
            loop.close()

        self._sync_loop_thread = threading.Thread(
            target=_runner,
            name="agentm-reloader-sync-loop",
            daemon=True,
        )
        self._sync_loop_thread.start()
        ready.wait()
        assert self._sync_loop is not None
        return self._sync_loop

    async def _activate_atom_install(self, atom: LoadedAtom) -> None:
        if self._api_factory is None:
            raise RuntimeError(
                "AtomReloader.set_api_factory must be called before reload_atom"
            )
        previous_api = self._apis.get(atom.module_path)
        # Capture handler positions BEFORE removal to restore after reload.
        positions = self._capture_handler_positions(atom.module_path)
        self._remove_handlers(atom.module_path)
        self._remove_registrations(atom.module_path)
        sys.modules.pop(atom.module_path, None)
        self._clear_module_bytecode(atom.file_path)
        importlib.invalidate_caches()
        # Synthetic atoms have no package finder; re-seed sys.modules from disk.
        if atom.import_kind == "synthetic" or atom.module_path.startswith(_SCENARIO_MODULE_PREFIX):
            # Scenario-local atoms are synthetic too.
            self._import_synthetic_module(atom.module_path, atom.file_path)
        await self._finish_install(
            atom.module_path,
            self._api_factory(atom.module_path),
            dict(atom.config),
        )
        if previous_api is not None:
            previous_api.mark_stale()
        self.record_loaded_atom(
            atom.module_path,
            atom.config,
            is_provider=atom.is_provider,
        )
        self._apis[atom.module_path]._owner_name = atom.module_path
        self._restore_handler_positions(atom.module_path, positions)
        from agentm.extensions import discover as _disc

        _disc.reset_cache()

    def _capture_handler_positions(self, owner: str) -> dict[str, int]:
        """Record per-channel handler positions for post-reload splicing."""
        positions: dict[str, int] = {}
        for channel in self._bus.channels():
            for idx, sub in enumerate(self._bus.subscriptions_for(channel)):
                if getattr(sub.handler, "_agentm_obs_owner", None) == owner:
                    positions[channel] = idx
                    break
        return positions

    def _restore_handler_positions(self, owner: str, positions: dict[str, int]) -> None:
        for channel, anchor_idx in positions.items():
            subs = self._bus.subscriptions_for(channel)
            if not subs:
                continue
            owner_subs = [
                sub
                for sub in subs
                if getattr(sub.handler, "_agentm_obs_owner", None) == owner
            ]
            if not owner_subs:
                continue
            non_owner = [
                sub
                for sub in subs
                if getattr(sub.handler, "_agentm_obs_owner", None) != owner
            ]
            clamp = min(anchor_idx, len(non_owner))
            self._bus.replace_subscriptions(
                channel,
                non_owner[:clamp] + owner_subs + non_owner[clamp:],
            )

    def _capture_snapshot(self, module_path: str) -> _ReloadSnapshot:
        return _ReloadSnapshot(
            loaded_by_module=dict(self._loaded_by_module),
            loaded_by_name=dict(self._loaded_by_name),
            handlers_by_atom={
                owner: list(unsubs)
                for owner, unsubs in self._handlers_by_atom.items()
            },
            registrations_by_atom={
                owner: list(regs) for owner, regs in self._registrations_by_atom.items()
            },
            owners_by_kind={
                kind: dict(name_map) for kind, name_map in self.owners_by_kind.items()
            },
            tools=list(self._tools),
            commands=dict(self._commands),
            providers=dict(self._providers),
            renderers=dict(self._renderers),
            apis=dict(self._apis),
            sys_module=sys.modules.get(module_path),
            module_path=module_path,
            bus_channels={
                channel: self._bus.subscriptions_for(channel)
                for channel in self._bus.channels()
            },
        )

    def _restore_from_snapshot(self, snapshot: _ReloadSnapshot) -> None:
        self._loaded_by_module.clear()
        self._loaded_by_module.update(snapshot.loaded_by_module)
        self._loaded_by_name.clear()
        self._loaded_by_name.update(snapshot.loaded_by_name)
        self._handlers_by_atom.clear()
        self._handlers_by_atom.update(
            {owner: list(unsubs) for owner, unsubs in snapshot.handlers_by_atom.items()}
        )
        self._registrations_by_atom.clear()
        self._registrations_by_atom.update(
            {owner: list(regs) for owner, regs in snapshot.registrations_by_atom.items()}
        )
        self.owners_by_kind.clear()
        self.owners_by_kind.update(
            {kind: dict(name_map) for kind, name_map in snapshot.owners_by_kind.items()}
        )
        self._tools[:] = list(snapshot.tools)
        self._commands.clear()
        self._commands.update(snapshot.commands)
        self._providers.clear()
        self._providers.update(snapshot.providers)
        self._renderers.clear()
        self._renderers.update(snapshot.renderers)
        self._apis.clear()
        self._apis.update(snapshot.apis)
        if snapshot.sys_module is not None:
            sys.modules[snapshot.module_path] = snapshot.sys_module
        else:
            sys.modules.pop(snapshot.module_path, None)
        # Reset the bus subscription map to the pre-reload shape.
        live_channels = set(self._bus.channels())
        for channel, subs in snapshot.bus_channels.items():
            self._bus.replace_subscriptions(channel, list(subs))
        for channel in live_channels - set(snapshot.bus_channels):
            self._bus.replace_subscriptions(channel, [])

    def _restore_git_path(self, atom: LoadedAtom, pre_sha: str) -> None:
        self._resource_writer.restore(atom.file_path, pre_sha)

    def _advisory_hash(self, source: str) -> str:
        # Legacy fallback for git-less/advisory environments only.
        return compute_atom_hash(source)

    # --- _SessionGateway protocol -----------------------------------------

    def reload_atom(
        self,
        name: str,
        new_source: str,
        *,
        agent_initiated: bool = True,
        rationale: str | None = None,
    ) -> ReloadResult:
        return self._run_api_boundary(
            self.reload_atom_async(
                name,
                new_source,
                agent_initiated=agent_initiated,
                rationale=rationale,
            )
        )

    async def reload_atom_async(
        self,
        name: str,
        new_source: str,
        *,
        agent_initiated: bool = True,
        rationale: str | None = None,
    ) -> ReloadResult:
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
                error=(f"refusing to reload constitution layer path {atom.file_path}"),
            )

        try:
            manifest = self._validate_reload_source(name, atom.module_path, new_source)
        except Exception as exc:  # noqa: BLE001
            return ReloadResult(
                ok=False,
                name=name,
                old_hash=None,
                new_hash=None,
                error=str(exc),
            )

        effective_manifest = manifest or atom.manifest or _default_manifest(name)
        if effective_manifest.tier == 2:
            logger.warning(f"tier-2 reload proceeds in MVP for {name}")

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
        write_result = await self._resource_writer.write(
            str(atom.file_path),
            new_source.encode("utf-8"),
            rationale=rationale or f"reload {name}",
            author="agent" if agent_initiated else "human",
        )
        if write_result.error is not None:
            return ReloadResult(
                ok=False,
                name=name,
                old_hash=write_result.commit_sha_before,
                new_hash=write_result.commit_sha_after,
                error=write_result.error,
            )

        advisory_mode = (
            write_result.path_class == "managed" and not write_result.committed
        ) or write_result.path_class == "unmanaged"
        old_hash = write_result.commit_sha_before
        new_hash = write_result.commit_sha_after
        if advisory_mode:
            old_hash = self._advisory_hash(current_source)
            new_hash = self._advisory_hash(new_source)
        event_new_hash = new_hash or self._advisory_hash(new_source)
        # Transactional snapshot for rollback.
        snapshot = self._capture_snapshot(atom.module_path)

        try:
            await self._activate_atom_install(atom)
            self._bus.emit_sync(
                ExtensionReloadEvent.CHANNEL,
                ExtensionReloadEvent(
                    name=name,
                    old_hash=old_hash,
                    new_hash=event_new_hash,
                    trigger="agent" if agent_initiated else "human",
                    tier=effective_manifest.tier,
                    is_self_modify=agent_initiated,
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
                if advisory_mode:
                    rollback_fd, rollback_tmp_name = tempfile.mkstemp(
                        prefix=f".rollback-{name}-",
                        suffix=atom.file_path.suffix,
                        dir=str(atom.file_path.parent),
                    )
                    with os.fdopen(
                        rollback_fd, "w", encoding="utf-8"
                    ) as rollback_handle:
                        rollback_handle.write(current_source)
                    os.replace(rollback_tmp_name, atom.file_path)
                elif write_result.commit_sha_before is not None:
                    self._restore_git_path(atom, write_result.commit_sha_before)
                await self._activate_atom_install(atom)
            except Exception as rollback_exc:  # noqa: BLE001
                # Rollback also failed; restore from immutable snapshot.
                # rather than something half-new / half-old.
                logger.exception(f"atom {name!r} rollback failed after apply failure; restoring from pre-reload snapshot")
                self._restore_from_snapshot(snapshot)
                self._bus.emit_sync(
                    ExtensionReloadEvent.CHANNEL,
                    ExtensionReloadEvent(
                        name=name,
                        old_hash=old_hash,
                        new_hash=event_new_hash,
                        trigger="agent" if agent_initiated else "human",
                        tier=effective_manifest.tier,
                        error="rollback_failure_state_preserved",
                        is_self_modify=agent_initiated,
                    ),
                )
                return ReloadResult(
                    ok=False,
                    name=name,
                    old_hash=old_hash,
                    new_hash=new_hash,
                    error=(
                        f"rollback_failure_state_preserved: {exc}; "
                        f"rollback failed: {rollback_exc}"
                    ),
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
        return self._run_api_boundary(self.freeze_current_async(name))

    async def freeze_current_async(self, name: str) -> str:
        atom = self._loaded_by_name[name]
        source = atom.file_path.read_text(encoding="utf-8")
        result = await self._resource_writer.write(
            str(atom.file_path),
            source.encode("utf-8"),
            rationale="freeze_current snapshot",
            author="indexer",
        )
        if result.error is not None:
            raise RuntimeError(result.error)
        version_key = (
            result.commit_sha_after
            or result.commit_sha_before
            or self._advisory_hash(source)
        )
        _layout.atom_runs_dir(name, version_key, root=Path(self._cwd)).mkdir(
            parents=True,
            exist_ok=True,
        )
        return version_key

    def list_atoms(self) -> list[AtomInfo]:
        out: list[AtomInfo] = []
        for atom in sorted(self._loaded_by_name.values(), key=lambda item: item.name):
            current_hash = self.current_version_for_path(str(atom.file_path))
            if current_hash is None and atom.file_path.is_file():
                current_hash = self._advisory_hash(
                    atom.file_path.read_text(encoding="utf-8")
                )
            manifest = atom.manifest or _default_manifest(atom.name)
            out.append(
                AtomInfo(
                    name=atom.name,
                    current_hash=current_hash,
                    tier=manifest.tier,
                    api_version=manifest.api_version,
                    source_path=str(atom.file_path),
                    config=dict(atom.config),
                )
            )
        return out

    @staticmethod
    def is_constitution_path(path: str) -> bool:
        return is_constitution_path(path)

    # --- Plug-and-play install / unload -----------------------------------
    #
    # ``reload_atom`` swaps an atom's source in place. ``install_atom`` adds
    # a brand-new one at runtime, ``unload_atom`` removes it. The mechanism
    # reuses the same building blocks: ResourceWriter for the on-disk write,
    # ``_validate_reload_source``-equivalent for the §11 contract, the
    # ``_remove_handlers``/``_remove_registrations``/``sys.modules.pop``
    # cleanup that ``_activate_atom_install`` already does mid-reload.
    #
    # The agent-installed module never goes on ``sys.path``: instead we
    # register it directly into ``sys.modules`` under a synthetic dotted
    # name. That isolates agent-written code from the rest of the package
    # (no accidental shadowing of builtin atoms) and lets us discard it
    # cleanly on unload.

    def install_atom(
        self,
        *,
        name: str,
        source: str,
        target_path: str | None,
        config: dict[str, Any] | None,
        rationale: str | None,
        agent_initiated: bool,
    ) -> InstallAtomResult:
        return self._run_api_boundary(
            self.install_atom_async(
                name=name,
                source=source,
                target_path=target_path,
                config=config,
                rationale=rationale,
                agent_initiated=agent_initiated,
            )
        )

    async def install_atom_async(
        self,
        *,
        name: str,
        source: str,
        target_path: str | None,
        config: dict[str, Any] | None,
        rationale: str | None,
        agent_initiated: bool,
    ) -> InstallAtomResult:
        if self._api_factory is None:
            raise RuntimeError(
                "AtomReloader.set_api_factory must be called before install_atom"
            )
        if not name or not name.isidentifier():
            return InstallAtomResult(
                ok=False,
                name=name,
                module_path=None,
                target_path=None,
                new_hash=None,
                file_created=False,
                error=f"invalid atom name {name!r}: must be a Python identifier",
            )
        if name in self._loaded_by_name:
            return InstallAtomResult(
                ok=False,
                name=name,
                module_path=None,
                target_path=None,
                new_hash=None,
                file_created=False,
                error=f"atom {name!r} is already loaded; use reload_atom to replace it",
            )

        from agentm.extensions import discover as discover_mod

        ext_cfg = dict(config or {})
        module_path = f"{discover_mod.USER_ATOM_MODULE_PREFIX}{name}"

        target_file = self._resolve_install_target(name, target_path)
        if is_constitution_path(str(target_file)):
            return InstallAtomResult(
                ok=False,
                name=name,
                module_path=None,
                target_path=str(target_file),
                new_hash=None,
                file_created=False,
                error=(
                    f"refusing to install at constitution path {target_file}; "
                    f"agent-installed atoms must live outside the constitution"
                ),
            )

        try:
            manifest = self._validate_install_source(name, module_path, source)
        except Exception as exc:  # noqa: BLE001
            return InstallAtomResult(
                ok=False,
                name=name,
                module_path=module_path,
                target_path=str(target_file),
                new_hash=None,
                file_created=False,
                error=str(exc),
            )

        if manifest is not None and manifest.tier >= 2:
            return InstallAtomResult(
                ok=False,
                name=name,
                module_path=module_path,
                target_path=str(target_file),
                new_hash=None,
                file_created=False,
                error=(
                    f"refusing to install tier-{manifest.tier} atom {name!r}: "
                    f"agent-installed atoms must be tier 1 (promotion needs human review)"
                ),
            )

        effective_tier = manifest.tier if manifest is not None else 1
        trigger: Any = "agent" if agent_initiated else "human"
        veto_reason = self._collect_block_veto(
            BeforeInstallAtomEvent.CHANNEL,
            BeforeInstallAtomEvent(
                name=name,
                module_path=module_path,
                target_path=str(target_file),
                source=source,
                config=ext_cfg,
                tier=effective_tier,
                trigger=trigger,
            ),
        )
        if veto_reason is not None:
            return InstallAtomResult(
                ok=False,
                name=name,
                module_path=module_path,
                target_path=str(target_file),
                new_hash=None,
                file_created=False,
                error=f"vetoed by policy: {veto_reason}",
            )

        file_existed = target_file.exists()
        target_file.parent.mkdir(parents=True, exist_ok=True)
        write_result = await self._resource_writer.write(
            str(target_file),
            source.encode("utf-8"),
            rationale=rationale or f"install atom {name}",
            author="agent" if agent_initiated else "human",
        )
        if write_result.error is not None:
            return InstallAtomResult(
                ok=False,
                name=name,
                module_path=module_path,
                target_path=str(target_file),
                new_hash=write_result.commit_sha_after,
                file_created=False,
                error=write_result.error,
            )

        new_hash = write_result.commit_sha_after or self._advisory_hash(source)
        effective_manifest = manifest or _default_manifest(name)

        # Register the module bytes synthetically so load_extension's
        # importlib.import_module call returns it.
        try:
            self._import_synthetic_module(module_path, target_file)
        except Exception as exc:  # noqa: BLE001
            self._cleanup_failed_install(target_file, file_existed, write_result)
            return InstallAtomResult(
                ok=False,
                name=name,
                module_path=module_path,
                target_path=str(target_file),
                new_hash=new_hash,
                file_created=not file_existed,
                error=f"import failed: {exc}",
            )

        self._bus.emit_sync(
            ExtensionInstallEvent.CHANNEL,
            ExtensionInstallEvent(
                module_path=module_path,
                config=dict(ext_cfg),
                phase="start",
                trigger=trigger,
            ),
        )

        try:
            api = self._api_factory(module_path)
            await self._finish_install(module_path, api, ext_cfg)
        except Exception as exc:  # noqa: BLE001
            self._remove_handlers(module_path)
            self._remove_registrations(module_path)
            sys.modules.pop(module_path, None)
            self._cleanup_failed_install(target_file, file_existed, write_result)
            self._bus.emit_sync(
                ExtensionInstallEvent.CHANNEL,
                ExtensionInstallEvent(
                    module_path=module_path,
                    config=dict(ext_cfg),
                    phase="error",
                    error=repr(exc),
                    trigger=trigger,
                ),
            )
            return InstallAtomResult(
                ok=False,
                name=name,
                module_path=module_path,
                target_path=str(target_file),
                new_hash=new_hash,
                file_created=not file_existed,
                error=f"install({module_path}) failed: {exc}",
            )

        # Bookkeeping: register as a loaded atom so reload_atom / unload_atom
        # / list_atoms see it. record_loaded_atom imports the module again,
        # but since it's already in sys.modules the import is a fast hit.
        self.record_loaded_atom(module_path, ext_cfg, is_provider=False)
        self._apis[module_path]._owner_name = module_path

        self._bus.emit_sync(
            ExtensionInstallEvent.CHANNEL,
            ExtensionInstallEvent(
                module_path=module_path,
                config=dict(ext_cfg),
                phase="end",
                trigger=trigger,
            ),
        )

        if effective_manifest.tier == 2:
            logger.warning(f"tier-2 install proceeds in MVP for {name} (no approval gate)")

        return InstallAtomResult(
            ok=True,
            name=name,
            module_path=module_path,
            target_path=str(target_file),
            new_hash=new_hash,
            file_created=not file_existed,
        )

    def unload_atom(
        self,
        name: str,
        *,
        agent_initiated: bool = True,
    ) -> UnloadAtomResult:
        atom = self._loaded_by_name.get(name)
        if atom is None:
            return UnloadAtomResult(
                ok=False,
                name=name,
                module_path=None,
                error=f"unknown atom {name!r}",
            )
        if atom.is_provider:
            return UnloadAtomResult(
                ok=False,
                name=name,
                module_path=atom.module_path,
                error=(
                    f"refusing to unload provider {name!r}: leaves the loop "
                    f"without a stream_fn"
                ),
            )
        if is_constitution_path(str(atom.file_path)):
            return UnloadAtomResult(
                ok=False,
                name=name,
                module_path=atom.module_path,
                error=(f"refusing to unload constitution-path atom {atom.file_path}"),
            )

        manifest = atom.manifest or _default_manifest(name)
        trigger: Any = "agent" if agent_initiated else "human"
        if manifest.tier == 2:
            logger.warning(f"tier-2 unload proceeds in MVP for {name} (no approval gate)")

        veto_reason = self._collect_block_veto(
            BeforeUnloadAtomEvent.CHANNEL,
            BeforeUnloadAtomEvent(
                name=name,
                module_path=atom.module_path,
                tier=manifest.tier,
                trigger=trigger,
            ),
        )
        if veto_reason is not None:
            return UnloadAtomResult(
                ok=False,
                name=name,
                module_path=atom.module_path,
                error=f"vetoed by policy: {veto_reason}",
            )

        previous_api = self._apis.get(atom.module_path)
        if previous_api is not None:
            previous_api.mark_stale()
        self._remove_handlers(atom.module_path)
        self._remove_registrations(atom.module_path)
        sys.modules.pop(atom.module_path, None)
        self._loaded_by_module.pop(atom.module_path, None)
        self._loaded_by_name.pop(atom.name, None)
        self._apis.pop(atom.module_path, None)

        self._bus.emit_sync(
            ExtensionUnloadEvent.CHANNEL,
            ExtensionUnloadEvent(
                name=name,
                module_path=atom.module_path,
                trigger=trigger,
                tier=manifest.tier,
            ),
        )

        return UnloadAtomResult(
            ok=True,
            name=name,
            module_path=atom.module_path,
        )

    def _collect_block_veto(self, channel: str, event: Any) -> str | None:
        """First-truthy ``{"block": True, "reason": "..."}`` from sync
        handlers on ``channel``, else ``None``. Mirrors the ``tool_call``
        block contract — first refusal wins; later handlers still run for
        observability but cannot un-veto.
        """
        for value in self._bus.emit_sync(channel, event):
            if isinstance(value, dict) and value.get("block"):
                return str(value.get("reason") or "blocked")
        return None

    def _resolve_install_target(self, name: str, target_path: str | None) -> Path:
        if target_path is None:
            return (Path(self._cwd) / ".agentm" / "atoms" / f"{name}.py").resolve()
        candidate = Path(target_path)
        if not candidate.is_absolute():
            candidate = Path(self._cwd) / candidate
        return candidate.resolve()

    def _validate_install_source(
        self, name: str, module_path: str, new_source: str
    ) -> ExtensionManifest | None:
        return self._validate_atom_source(
            name,
            module_path,
            new_source,
            tag="install",
            require_manifest=False,
        )

    @staticmethod
    def _import_synthetic_module(module_path: str, file_path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(module_path, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not build spec for {module_path!r}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_path, None)
            raise
        return module

    def _cleanup_failed_install(
        self,
        target_file: Path,
        file_existed: bool,
        write_result: Any,
    ) -> None:
        # Best-effort: if we wrote a brand-new file but the install raised,
        # roll the file back so the on-disk state matches the running session.
        # For pre-existing files we restore via git when possible, else leave
        # alone so the human can inspect.
        if not file_existed:
            try:
                target_file.unlink()
            except OSError:
                pass
            if write_result.committed and write_result.commit_sha_before is not None:
                try:
                    self._resource_writer.restore(
                        target_file, write_result.commit_sha_before
                    )
                except Exception:  # noqa: BLE001
                    pass

    def current_version_for_path(self, path: str) -> str | None:
        return self._resource_writer.current_version_for_path(path)


__all__ = ["AtomReloader", "LoadedAtom"]
