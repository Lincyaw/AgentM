"""AgentSession orchestrator: the fat-but-thin v2 façade.

Implements §4 (AgentSession) of ``.claude/designs/extension-as-scenario.md``.
The session holds references to every subsystem (event bus, session manager,
resource loader, registries) and wires events. It runs no business logic:
each "feature" is an extension that registers handlers on the bus.

Lifecycle (``AgentSession.create`` → ``prompt`` → ``shutdown``):

1. Build :class:`EventBus`, :class:`SessionManager`, :class:`ResourceLoader`,
   internal ``_ExtensionAPIImpl``.
2. Load every extension in order; await coroutine returns. The provider
   extension is loaded last so the picked-up provider reflects any earlier
   replacement attempts (last registration wins).
3. Append ``initial_messages`` (if any) into the session manager.

On ``prompt(text)``:

1. Build a :class:`UserMessage`, append it as a session entry.
2. Assemble the system prompt from context files + skill descriptions
   (placeholder; full skill-body expansion comes in a later phase).
3. Emit ``before_agent_start``; handlers may return ``{"system": "..."}``
   replacing the system prompt (last non-None wins, mirroring the kernel
   ``_collect_replacement`` convention).
4. Run ``AgentLoop.run``.
5. Append every new assistant + tool_result message as session entries.
6. Return the full updated message list.

Hard rule: this module imports only stdlib + ``agentm.core.kernel`` + the
three sibling v2 modules.
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
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from agentm.core.catalog import (
    compute_atom_hash,
    freeze_current as freeze_atom_snapshot,
    is_constitution_path,
    source_path_for_hash,
)
from agentm.core.kernel import (
    AgentEndEvent,
    AgentLoop,
    AgentMessage,
    EventBus,
    ImageContent,
    LoopConfig,
    Model,
    TextContent,
    Tool,
    UserMessage,
)

from agentm.harness.events import (
    ApiRegisterEvent,
    BeforeAgentStartEvent,
    ChildSessionEndEvent,
    ChildSessionStartEvent,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
)
from agentm.harness.extension import (
    AtomInfo,
    CommandSpec,
    ExtensionLoadError,
    ProviderConfig,
    ReadonlySession,
    ReloadResult,
    Renderer,
    _ExtensionAPIImpl,
    load_extension,
)
from agentm.extensions import ExtensionManifest
from agentm.extensions import discover as discover_mod
from agentm.extensions import validate as validate_mod
from agentm.harness.resource_loader import (
    DefaultResourceLoader,
    ResourceLoader,
)
from agentm.harness.session_manager import (
    InMemorySessionManager,
    SessionEntry,
    SessionManager,
)


logger = logging.getLogger(__name__)


# --- Config -----------------------------------------------------------------


@dataclass
class AgentSessionConfig:
    """Knobs handed to :func:`AgentSession.create`. Only ``cwd``, ``provider``
    are required; everything else has a sane default for embedded use."""

    cwd: str
    extensions: list[tuple[str, dict[str, Any]]]
    provider: tuple[str, dict[str, Any]]
    initial_messages: list[AgentMessage] = field(default_factory=list)
    session_manager: SessionManager | None = None
    resource_loader: ResourceLoader | None = None
    loop_config: LoopConfig | None = None
    # --- Child-session lifecycle (used by sub-agent extensions) ----------
    parent_bus: EventBus | None = None
    """If set, ``child_session_start`` / ``child_session_end`` are emitted on
    this bus when the session is created and shut down. Used by the
    ``sub_agent`` extension to roll up nested sessions on the parent."""

    parent_session_id: str | None = None
    """Caller-supplied id of the parent session. Surfaces verbatim in the
    child-lifecycle events. ``None`` becomes ``"unknown"`` in the payload."""

    purpose: str = "root"
    """Caller-defined purpose label, e.g. ``"subagent:worker"``;
    surfaces verbatim in :class:`ChildSessionStartEvent`."""


# --- Helpers ----------------------------------------------------------------


class _SessionView:
    """``ReadonlySession`` adapter over a ``SessionManager``.

    Exposes message reads plus the one mutation extensions are allowed:
    ``append_entry`` for persisting structured payloads (compaction summaries,
    hypothesis snapshots, plan submissions) into the entry tree. Everything
    else (fork / navigate) stays inside the harness.
    """

    def __init__(self, sm: SessionManager) -> None:
        self._sm = sm

    def get_messages(self) -> list[AgentMessage]:
        return self._sm.get_messages()

    def get_branch(self) -> list[SessionEntry]:
        return self._sm.get_active_branch()

    def get_leaf_id(self) -> str | None:
        return self._sm.get_leaf_id()

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        return self._sm.get_entry(entry_id)

    def append_entry(
        self,
        type: str,
        payload: Any,
        parent_id: str | None = None,
    ) -> str:
        if parent_id is None:
            branch = self._sm.get_active_branch()
            parent_id = branch[-1].id if branch else None
        entry = SessionEntry(
            type=type,
            id=uuid.uuid4().hex,
            parent_id=parent_id,
            timestamp=_now(),
            payload=payload,
        )
        self._sm.append(entry)
        return entry.id


def _now() -> float:
    return time.time()


def _collect_system_replacement(returns: list[Any]) -> str | None:
    """Pick the last non-None ``system`` replacement from handler returns.

    Mirrors ``loop._collect_replacement`` semantics: handlers may return a
    dict ``{"system": "..."}`` to override the assembled system prompt; the
    most recently registered authoritative voice wins.
    """

    chosen: str | None = None
    for value in returns:
        if isinstance(value, dict) and value.get("system") is not None:
            candidate = value["system"]
            if isinstance(candidate, str):
                chosen = candidate
    return chosen


@dataclass(slots=True)
class _LoadedAtom:
    name: str
    module_path: str
    file_path: Path
    config: dict[str, Any]
    manifest: ExtensionManifest | None
    is_provider: bool = False


# --- AgentSession -----------------------------------------------------------


class AgentSession:
    """Top-level v2 session façade. Construct via :meth:`create`."""

    def __init__(
        self,
        *,
        cwd: str,
        bus: EventBus,
        session_manager: SessionManager,
        resource_loader: ResourceLoader,
        loop: AgentLoop,
        active_provider_box: dict[str, ProviderConfig | None],
        tools: list[Tool],
        commands: dict[str, CommandSpec],
        providers: dict[str, ProviderConfig],
        renderers: dict[str, Renderer],
        apis: dict[str, _ExtensionAPIImpl],
        command_owners: dict[str, str],
        loaded_atoms_by_name: dict[str, _LoadedAtom],
        pending_user_messages: list[str | list[Any]],
        session_id: str,
        parent_bus: EventBus | None,
        parent_session_id: str | None,
        purpose: str,
    ) -> None:
        self._cwd = cwd
        self._bus = bus
        self._session_manager = session_manager
        self._resources = resource_loader
        self._loop = loop
        self._active_provider_box = active_provider_box
        self._tools = tools
        self._commands = commands
        self._providers = providers
        self._renderers = renderers
        self._apis = apis
        self._command_owners = command_owners
        self._loaded_atoms_by_name = loaded_atoms_by_name
        self._extension_api = next(iter(apis.values())) if apis else None
        self._pending_user_messages = pending_user_messages
        self._session_id = session_id
        self._parent_bus = parent_bus
        self._parent_session_id = parent_session_id
        self._purpose = purpose
        # Set by the cost_budget extension via the cost_budget_exceeded
        # channel; checked at the top of ``prompt`` so the next turn
        # short-circuits cleanly with stop_reason="budget".
        self._budget_exceeded: bool = False

    # --- Construction -----------------------------------------------------

    @classmethod
    async def create(cls, config: AgentSessionConfig) -> "AgentSession":
        """Build a session: assemble subsystems, load extensions, return."""

        bus = EventBus()
        session_manager: SessionManager = (
            config.session_manager
            if config.session_manager is not None
            else InMemorySessionManager(cwd=config.cwd)
        )
        resource_loader: ResourceLoader = (
            config.resource_loader
            if config.resource_loader is not None
            else DefaultResourceLoader(cwd=Path(config.cwd))
        )

        tools: list[Tool] = []
        commands: dict[str, CommandSpec] = {}
        providers: dict[str, ProviderConfig] = {}
        renderers: dict[str, Renderer] = {}
        pending_user_messages: list[str | list[Any]] = []
        apis: dict[str, _ExtensionAPIImpl] = {}
        handlers_by_atom: dict[str, list[Any]] = {}
        registrations_by_atom: dict[str, list[tuple[str, str, Any]]] = {}
        command_owners: dict[str, str] = {}
        loaded_atoms_by_module: dict[str, _LoadedAtom] = {}
        loaded_atoms_by_name: dict[str, _LoadedAtom] = {}

        # We need a forward reference to the picked-up active provider so the
        # api.model property reflects it once the provider extension runs.
        active_provider_box: dict[str, ProviderConfig | None] = {"value": None}
        loop_box: dict[str, AgentLoop | None] = {"value": None}

        def _model_getter() -> Model | None:
            cur = active_provider_box["value"]
            return cur.model if cur is not None else None

        def _provider_getter() -> ProviderConfig | None:
            return active_provider_box["value"]

        session_view: ReadonlySession = _SessionView(session_manager)

        def _refresh_active_provider() -> None:
            active_provider_box["value"] = (
                providers[next(reversed(providers))] if providers else None
            )
            loop = loop_box["value"]
            active = active_provider_box["value"]
            if loop is not None and active is not None:
                loop._stream_fn = active.stream_fn  # type: ignore[attr-defined]

        def _track_registration(event: ApiRegisterEvent) -> None:
            registrations_by_atom.setdefault(event.extension, []).append(
                (event.kind, event.name, event.payload)
            )
            if event.kind == "command":
                command_owners[event.name] = event.extension

        bus.on("api_register", _track_registration)

        def _wrap_on(api: _ExtensionAPIImpl, owner: str) -> None:
            original_on = api.on

            def tracked(channel: str, handler: Any) -> Any:
                try:
                    setattr(handler, "_agentm_obs_owner", owner)
                except (AttributeError, TypeError):
                    pass
                unsub = original_on(channel, handler)
                handlers_by_atom.setdefault(owner, []).append(unsub)
                return unsub

            api.on = tracked  # type: ignore[method-assign]

        def _default_manifest(name: str) -> ExtensionManifest:
            return ExtensionManifest(
                name=name,
                description=f"Reload snapshot for {name}",
                registers=(),
            )

        def _remove_handlers(owner: str) -> None:
            for unsub in handlers_by_atom.pop(owner, []):
                unsub()

        def _remove_registrations(owner: str) -> None:
            for kind, name, payload in registrations_by_atom.pop(owner, []):
                if kind == "tool":
                    tools[:] = [tool for tool in tools if tool is not payload]
                elif kind == "command":
                    if commands.get(name) is payload:
                        commands.pop(name, None)
                    if command_owners.get(name) == owner:
                        command_owners.pop(name, None)
                elif kind == "provider":
                    if providers.get(name) is payload:
                        providers.pop(name, None)
                        _refresh_active_provider()
                elif kind == "renderer":
                    if renderers.get(name) is payload:
                        renderers.pop(name, None)

        def _module_name(module_path: str, module: ModuleType) -> str:
            manifest_obj = getattr(module, "MANIFEST", None)
            if isinstance(manifest_obj, ExtensionManifest):
                return manifest_obj.name
            return module_path.rsplit(".", 1)[-1]

        def _module_manifest(module: ModuleType) -> ExtensionManifest | None:
            manifest_obj = getattr(module, "MANIFEST", None)
            return manifest_obj if isinstance(manifest_obj, ExtensionManifest) else None

        def _record_loaded_atom(
            module_path: str,
            ext_cfg: dict[str, Any],
            *,
            is_provider: bool,
        ) -> None:
            module = importlib.import_module(module_path)
            module_file = getattr(module, "__file__", None)
            file_path = Path(module_file).resolve() if module_file else Path(".")
            manifest = _module_manifest(module)
            atom = _LoadedAtom(
                name=_module_name(module_path, module),
                module_path=module_path,
                file_path=file_path,
                config=dict(ext_cfg),
                manifest=manifest,
                is_provider=is_provider,
            )
            loaded_atoms_by_module[module_path] = atom
            loaded_atoms_by_name[atom.name] = atom

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
            name: str,
            module_path: str,
            new_source: str,
        ) -> ExtensionManifest | None:
            with tempfile.TemporaryDirectory(prefix=f"agentm-reload-{name}-") as tmpdir:
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
                    raise RuntimeError(f"'install' must accept (api, config); got {sig}")

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
                    known_extension_names=set(loaded_atoms_by_name),
                )
                blocking = [issue for issue in issues if issue.severity == "error"]
                if blocking:
                    raise RuntimeError(blocking[0].message)
                return manifest

        def _finish_install_sync(
            module_path: str,
            api: _ExtensionAPIImpl,
            ext_cfg: dict[str, Any],
        ) -> None:
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

        class _Gateway:
            def reload_atom(
                self,
                name: str,
                new_source: str,
                *,
                agent_initiated: bool = True,
                rationale: str | None = None,
            ) -> ReloadResult:
                del rationale
                atom = loaded_atoms_by_name.get(name)
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
                        error=f"refusing to reload constitution layer path {atom.file_path}",
                    )

                try:
                    manifest = _validate_reload_source(name, atom.module_path, new_source)
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

                old_hash = freeze_atom_snapshot(
                    name,
                    current_source,
                    atom.manifest or _default_manifest(name),
                    root=Path(config.cwd),
                )
                snapshot_path = source_path_for_hash(name, old_hash, root=Path(config.cwd))
                snapshot_source = snapshot_path.read_text(encoding="utf-8")
                new_hash = compute_atom_hash(new_source)

                fd, tmp_name = tempfile.mkstemp(
                    prefix=f".reload-{name}-",
                    suffix=atom.file_path.suffix,
                    dir=str(atom.file_path.parent),
                )
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(new_source)

                previous_api = apis.get(atom.module_path)
                if previous_api is not None:
                    previous_api.mark_stale()
                _remove_handlers(atom.module_path)
                _remove_registrations(atom.module_path)
                sys.modules.pop(atom.module_path, None)

                try:
                    os.replace(tmp_name, atom.file_path)
                    _clear_module_bytecode(atom.file_path)
                    importlib.invalidate_caches()
                    _finish_install_sync(
                        atom.module_path,
                        _make_api(atom.module_path),
                        dict(atom.config),
                    )
                    _record_loaded_atom(
                        atom.module_path,
                        atom.config,
                        is_provider=atom.is_provider,
                    )
                    apis[atom.module_path]._owner_name = atom.module_path
                    discover_mod.reset_cache()
                    bus.emit_sync(
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
                        _clear_module_bytecode(atom.file_path)
                        importlib.invalidate_caches()
                        _finish_install_sync(
                            atom.module_path,
                            _make_api(atom.module_path),
                            dict(atom.config),
                        )
                        _record_loaded_atom(
                            atom.module_path,
                            atom.config,
                            is_provider=atom.is_provider,
                        )
                    except Exception as rollback_exc:  # noqa: BLE001
                        loaded_atoms_by_module.pop(atom.module_path, None)
                        loaded_atoms_by_name.pop(atom.name, None)
                        apis.pop(atom.module_path, None)
                        bus.emit_sync(
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
                atom = loaded_atoms_by_name[name]
                source = atom.file_path.read_text(encoding="utf-8")
                return freeze_atom_snapshot(
                    name,
                    source,
                    atom.manifest or _default_manifest(name),
                    root=Path(config.cwd),
                )

            def list_atoms(self) -> list[AtomInfo]:
                out: list[AtomInfo] = []
                for atom in sorted(loaded_atoms_by_name.values(), key=lambda item: item.name):
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

            def is_constitution_path(self, path: str) -> bool:
                return is_constitution_path(path)

        gateway = _Gateway()

        def _make_api(owner: str) -> _ExtensionAPIImpl:
            api = _ExtensionAPIImpl(
                bus=bus,
                cwd=config.cwd,
                session=session_view,
                tools=tools,
                commands=commands,
                providers=providers,
                renderers=renderers,
                pending_user_messages=pending_user_messages,
                model_getter=_model_getter,
                provider_getter=_provider_getter,
                gateway=gateway,
                owner_name=owner,
            )
            _wrap_on(api, owner)
            apis[owner] = api
            return api

        async def _install_with_events(
            module_path: str,
            ext_cfg: dict[str, Any],
            *,
            is_provider: bool = False,
        ) -> None:
            await bus.emit(
                "extension_install",
                ExtensionInstallEvent(
                    module_path=module_path, config=dict(ext_cfg), phase="start"
                ),
            )
            t0 = time.perf_counter_ns()
            try:
                result = load_extension(module_path, _make_api(module_path), ext_cfg)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                await bus.emit(
                    "extension_install",
                    ExtensionInstallEvent(
                        module_path=module_path,
                        config=dict(ext_cfg),
                        phase="error",
                        duration_ns=time.perf_counter_ns() - t0,
                        error=repr(exc),
                    ),
                )
                raise
            _record_loaded_atom(module_path, ext_cfg, is_provider=is_provider)
            await bus.emit(
                "extension_install",
                ExtensionInstallEvent(
                    module_path=module_path,
                    config=dict(ext_cfg),
                    phase="end",
                    duration_ns=time.perf_counter_ns() - t0,
                ),
            )

        # Load auxiliary extensions first.
        for module_path, ext_cfg in config.extensions:
            await _install_with_events(module_path, ext_cfg)

        # Load the provider extension. After it returns, we expect it to have
        # registered a ProviderConfig.
        provider_path, provider_cfg = config.provider
        await _install_with_events(provider_path, provider_cfg, is_provider=True)

        if not providers:
            raise ExtensionLoadError(
                provider_path,
                RuntimeError(
                    "provider extension did not call api.register_provider"
                ),
            )

        # Pick the most recently registered provider as active. dict insertion
        # order is preserved on Python 3.7+; the last-inserted entry is the
        # authoritative one.
        active_name = next(reversed(providers))
        active_provider = providers[active_name]
        active_provider_box["value"] = active_provider

        # Build the kernel loop now that we have a stream_fn.
        loop = AgentLoop(
            stream_fn=active_provider.stream_fn,
            bus=bus,
            config=config.loop_config or LoopConfig(),
        )
        loop_box["value"] = loop

        # Seed initial messages (if any) into the session manager.
        for msg in config.initial_messages:
            session_manager.append_message(msg)

        session_id = uuid.uuid4().hex
        instance = cls(
            cwd=config.cwd,
            bus=bus,
            session_manager=session_manager,
            resource_loader=resource_loader,
            loop=loop,
            active_provider_box=active_provider_box,
            tools=tools,
            commands=commands,
            providers=providers,
            renderers=renderers,
            apis=apis,
            command_owners=command_owners,
            loaded_atoms_by_name=loaded_atoms_by_name,
            pending_user_messages=pending_user_messages,
            session_id=session_id,
            parent_bus=config.parent_bus,
            parent_session_id=config.parent_session_id,
            purpose=config.purpose,
        )

        # Latch budget-exceeded once subscribed extensions emit it. The flag
        # is checked at the top of every ``prompt`` so the next turn
        # short-circuits with ``stop_reason='budget'``. Pure event-bus
        # signalling per §10b.8 — no exceptions cross handler boundaries.
        def _on_budget_exceeded(_: Any) -> None:
            instance._budget_exceeded = True

        bus.on("cost_budget_exceeded", _on_budget_exceeded)

        # Emit ``session_ready`` after every extension is loaded and the
        # active provider is picked. This is the only point where extensions
        # are guaranteed to see the final tool/command/model set; ``tool_filter``
        # and similar post-install scrubbers hook here.
        await bus.emit(
            "session_ready",
            SessionReadyEvent(
                cwd=config.cwd,
                session_id=session_id,
                tool_names=tuple(t.name for t in tools),
                command_names=tuple(commands.keys()),
                model=active_provider.model,
            ),
        )

        # Emit child-session lifecycle on the parent's bus (if any) so
        # ``sub_agent`` / ``trajectory`` extensions can roll up nested work.
        if config.parent_bus is not None:
            await config.parent_bus.emit(
                "child_session_start",
                ChildSessionStartEvent(
                    child_session_id=session_id,
                    parent_session_id=config.parent_session_id or "unknown",
                    purpose=config.purpose,
                ),
            )

        return instance

    # --- Public surface ---------------------------------------------------

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @property
    def resources(self) -> ResourceLoader:
        return self._resources

    @property
    def tools(self) -> list[Tool]:
        # v0: no allowlist filtering yet; that lands when tool_filter is
        # ported as a builtin extension in Phase 2.
        return list(self._tools)

    @property
    def model(self) -> Model | None:
        active = self._active_provider_box["value"]
        return active.model if active is not None else None

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def session_id(self) -> str:
        """Stable random id assigned at ``create``. Appears in
        :class:`ChildSessionStartEvent` / :class:`ChildSessionEndEvent`
        payloads when this session is a child of another."""
        return self._session_id

    # --- prompt -----------------------------------------------------------

    async def prompt(
        self,
        text: str,
        *,
        images: list[ImageContent] | None = None,
        signal: asyncio.Event | None = None,
    ) -> list[AgentMessage]:
        """Run one user-prompt → assistant-final-answer turn cycle.

        Drains queued ``send_user_message`` content, dispatches slash-commands,
        budget-gate-checks, drives the kernel loop, appends every new
        assistant + tool_result message, and returns the full active-branch
        message list. Stays a mechanical dispatcher per design §4.
        """

        # 0. Slash-command dispatch / input preprocessing. Code commands win;
        # otherwise ``input`` handlers may rewrite slash-prefixed text before
        # it falls through to the agent loop.
        text, slash_handled = await self._preprocess_input(text)
        if slash_handled is not None:
            return slash_handled

        # 1. Budget gate: if a previous turn tripped cost_budget_exceeded,
        # short-circuit with a stop_reason='budget' agent_end and persist
        # nothing. The flag stays latched until reset by an extension.
        if self._budget_exceeded:
            messages = self._session_manager.build_session_context().messages
            await self._bus.emit(
                "agent_end",
                AgentEndEvent(
                    messages=messages,
                    stop_reason="budget",
                ),
            )
            return messages

        # 2. Drain ``send_user_message`` queue (FIFO) into user-message
        # entries in the session. This is how ``sub_agent.inject_instruction``
        # and similar extensions push content into the next turn.
        await self._drain_pending_user_messages()

        # 3. Build the caller's user message, after any drained queue items
        # so they appear as turn-prefix context.
        text = text.replace("//", "/", 1) if text.lstrip().startswith("//") else text
        user_msg = self._build_user_message(text=text, images=images)
        entry = self._append_message(user_msg)

        # 4. Gather active-branch messages and run before_agent_start.
        messages = self._session_manager.build_session_context().messages
        system_prompt = self._build_system_prompt()
        before_returns = await self._bus.emit(
            "before_agent_start",
            BeforeAgentStartEvent(messages=messages, system=system_prompt),
        )
        replacement_system = _collect_system_replacement(before_returns)
        if replacement_system is not None:
            system_prompt = replacement_system

        # Snapshot object identities of pre-run messages. We can't slice by
        # index because per-turn extensions (e.g. micro_compact) may mutate
        # the list in place via ``messages[:] = compacted`` from a
        # ``before_send_to_llm`` handler — design ``extension-as-scenario``
        # §10b.2: SessionManager owns durable history; context is per-turn
        # ephemeral. Identity-based diff stays correct under any such
        # rewrite.
        pre_run_ids: set[int] = {id(m) for m in messages}
        budget_before_run = self._budget_exceeded

        # 5. Run the loop.
        final_messages = await self._loop.run(
            messages=messages,
            model=self._require_model(),
            tools=self._tools,
            system=system_prompt,
            signal=signal,
        )

        # 6. Append every new assistant / tool_result message — those whose
        # identities did not exist in the pre-run snapshot, in the order
        # they appear in the returned list.
        persisted_context = self._session_manager.build_session_context().messages
        cursor: str | None = self._session_manager.get_leaf_id() or entry.id
        for msg in final_messages:
            if id(msg) in pre_run_ids:
                continue
            if msg in persisted_context:
                continue
            if cursor is None:
                self._session_manager.reset_leaf()
            else:
                self._session_manager.branch(cursor)
            child = self._session_manager.append_message(msg)
            cursor = child.id

        if self._budget_exceeded and not budget_before_run:
            await self._bus.emit(
                "agent_end",
                AgentEndEvent(
                    messages=final_messages,
                    stop_reason="budget",
                ),
            )

        return final_messages

    # --- prompt helpers ---------------------------------------------------

    async def _preprocess_input(
        self, text: str
    ) -> tuple[str, list[AgentMessage] | None]:
        """Dispatch code commands, then let ``input`` handlers rewrite text.

        Unknown slash-prefixed text is left alone unless an ``input`` handler
        mutates it, so templates can expand and unmatched commands still reach
        the model verbatim.
        """

        stripped = text.lstrip()
        if not stripped.startswith("/") or stripped.startswith("//"):
            return text, None
        head, _, rest = stripped[1:].partition(" ")
        if not head:
            return text, None
        cmd = self._commands.get(head)
        if cmd is not None:
            owner = self._command_owners.get(head)
            api = self._apis[owner] if owner is not None else next(iter(self._apis.values()))
            result = cmd.handler(rest.strip(), api)
            if inspect.isawaitable(result):
                await result
            return text, self._session_manager.get_messages()

        event = {"text": text}
        await self._bus.emit("input", event)
        new_text = event.get("text")
        return (new_text if isinstance(new_text, str) else text), None

    async def _drain_pending_user_messages(self) -> None:
        """Pop every queued ``send_user_message`` payload and append it as a
        user-message entry. Called once per ``prompt`` before the caller's
        text is appended, so queued items act as turn-prefix context.
        """

        while self._pending_user_messages:
            queued = self._pending_user_messages.pop(0)
            content: list[TextContent | ImageContent]
            if isinstance(queued, str):
                content = [TextContent(type="text", text=queued)]
            else:
                content = list(queued)
            queued_msg = UserMessage(
                role="user", content=content, timestamp=_now()
            )
            self._append_message(queued_msg)

    def _build_user_message(
        self, *, text: str, images: list[ImageContent] | None
    ) -> UserMessage:
        content: list[TextContent | ImageContent] = []
        if text:
            content.append(TextContent(type="text", text=text))
        if images:
            content.extend(images)
        return UserMessage(role="user", content=content, timestamp=_now())

    def _append_message(self, msg: AgentMessage) -> Any:
        return self._session_manager.append_message(msg)

    def _require_model(self) -> Model:
        model = self.model
        if model is None:
            raise RuntimeError("no active provider model is available")
        return model

    # --- Lifecycle --------------------------------------------------------

    async def shutdown(self) -> None:
        """Signal extensions and clear handlers.

        Phase 1 emits a single ``session_shutdown`` event then drops every
        subscription. Extensions that need cleanup hook ``on('session_shutdown')``.
        """

        await self._bus.emit("session_shutdown", SessionShutdownEvent(cwd=self._cwd))

        # Notify the parent (if any) BEFORE clearing handlers so that an
        # extension subscribed on the parent bus can still observe the end
        # event with an accurate message count.
        if self._parent_bus is not None:
            await self._parent_bus.emit(
                "child_session_end",
                ChildSessionEndEvent(
                    child_session_id=self._session_id,
                    parent_session_id=self._parent_session_id or "unknown",
                    final_message_count=len(self._session_manager.get_messages()),
                    error=None,
                ),
            )

        self._bus.clear()

    # --- Helpers ----------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Concatenate context files + skill names/descriptions.

        Placeholder implementation (per design §4): full skill-body expansion
        (lazy injection on invocation) is deferred to a later phase. For v0
        we surface skill descriptions in the system prompt so the model knows
        they exist.
        """

        parts: list[str] = []
        for cf in self._resources.get_context_files():
            parts.append(cf.body.rstrip())

        skills = self._resources.get_skills()
        if skills:
            parts.append("# Available skills")
            for skill in skills:
                parts.append(f"- {skill.name}: {skill.description}")

        return "\n\n".join(parts)


__all__ = [
    "AgentSession",
    "AgentSessionConfig",
]
