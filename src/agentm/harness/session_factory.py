"""AgentSession construction wiring."""

from __future__ import annotations

import inspect
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from agentm.core.abi import AgentLoop, EventBus, LoopConfig, Model, Tool
from agentm.core.abi.events import DiagnosticEvent
from agentm.extensions import discover as discover_mod
from agentm.harness.atom_reloader import AtomReloader
from agentm.harness.command_dispatcher import HarnessCommandDispatcher
from agentm.harness.events import (
    ChildSessionStartEvent,
    ExtensionInstallEvent,
    SessionReadyEvent,
)
from agentm.harness.extension import (
    CommandSpec,
    ExtensionLoadError,
    ProviderConfig,
    ReadonlySession,
    Renderer,
    _ExtensionAPIImpl,
    load_extension,
)
from agentm.harness.provider_resolver import LastRegisteredWins
from agentm.harness.resource_loader import DefaultResourceLoader, ResourceLoader
from agentm.harness.resource_writer import GitBackedResourceWriter
from agentm.harness.session_config import (
    AgentSessionConfig,
    default_child_provider_factory,
)
from agentm.harness.session_helpers import (
    AtomSource,
    SessionView,
    collect_auto_discovered_atoms,
    ensure_floor_atom,
    resolve_provider_config,
)
from agentm.harness.session_manager import InMemorySessionManager, SessionManager
from agentm.harness.session_runtime import SessionRuntime

logger = logging.getLogger(__name__)


async def create_agent_session(
    session_cls: type[Any], config: AgentSessionConfig
) -> Any:
    bus = config.bus if config.bus is not None else EventBus()
    session_manager: SessionManager = (
        config.session_manager
        if config.session_manager is not None
        else InMemorySessionManager(cwd=config.cwd)
    )
    resource_loader: ResourceLoader = (
        config.resource_loader
        if config.resource_loader is not None
        else DefaultResourceLoader(
            cwd=Path(config.cwd),
            no_skills=config.no_skills,
            no_prompt_templates=config.no_prompt_templates,
        )
    )

    tools: list[Tool] = []
    commands: dict[str, CommandSpec] = {}
    providers: dict[str, ProviderConfig] = {}
    renderers: dict[str, Renderer] = {}
    pending_user_messages: list[str | list[Any]] = []
    apis: dict[str, _ExtensionAPIImpl] = {}
    services: dict[str, Any] = {}

    active_provider_box: dict[str, ProviderConfig | None] = {"value": None}
    loop_box: dict[str, AgentLoop | None] = {"value": None}
    configured_loop_config = config.loop_config or LoopConfig()
    provider_resolver = config.provider_resolver or LastRegisteredWins()
    child_provider_factory = (
        config.child_provider_factory or default_child_provider_factory
    )

    def _model_getter() -> Model | None:
        cur = active_provider_box["value"]
        return cur.model if cur is not None else None

    def _provider_getter() -> ProviderConfig | None:
        return active_provider_box["value"]

    session_id = uuid.uuid4().hex
    session_view: ReadonlySession = SessionView(
        session_manager,
        loop_config_getter=lambda: configured_loop_config,
    )
    resource_writer = GitBackedResourceWriter(
        cwd=config.cwd,
        session_id=session_id,
        bus=bus,
    )

    _configure_manifest(config.cwd)
    _migrate_catalog(config.cwd)

    def _refresh_active_provider() -> None:
        active_provider_box["value"] = (
            resolve_provider_config(
                providers, provider_resolver, provider_path="<resolver>"
            )
            if providers
            else None
        )
        loop = loop_box["value"]
        active = active_provider_box["value"]
        if loop is not None and active is not None:
            loop.set_stream_fn(active.stream_fn)

    reloader = AtomReloader(
        cwd=config.cwd,
        resource_writer=resource_writer,
        bus=bus,
        tools=tools,
        commands=commands,
        providers=providers,
        renderers=renderers,
        apis=apis,
        on_provider_changed=_refresh_active_provider,
    )

    async def _spawn_child_session(child_config: Any) -> Any:
        if isinstance(child_config, dict):
            child_config = AgentSessionConfig(**child_config)
        if not isinstance(child_config, AgentSessionConfig):
            raise TypeError(
                "spawn_child_session expects an AgentSessionConfig or kwargs dict; "
                f"got {type(child_config).__name__}"
            )
        spec = AgentSessionConfig(**{**child_config.__dict__})
        spec.parent_bus = bus
        spec.parent_session_id = session_id
        spec.root_session_id = config.root_session_id or session_id
        if spec.provider is None:
            parent_provider = _provider_getter()
            if parent_provider is None:
                raise RuntimeError(
                    "spawn_child_session: AgentSessionConfig.provider is None but "
                    "the parent session has no active provider to inherit."
                )
            spec.provider = child_provider_factory(parent_provider)
        return await session_cls.create(spec)

    def _make_api(owner: str) -> _ExtensionAPIImpl:
        api = _ExtensionAPIImpl(
            bus=bus,
            cwd=config.cwd,
            session_id=session_id,
            session=session_view,
            tools=tools,
            commands=commands,
            providers=providers,
            renderers=renderers,
            pending_user_messages=pending_user_messages,
            model_getter=_model_getter,
            provider_getter=_provider_getter,
            gateway=reloader,
            owner_name=owner,
            child_session_factory=_spawn_child_session,
            resource_writer=resource_writer,
            service_registry=services,
        )
        reloader.wrap_api_on(api, owner)
        apis[owner] = api
        return api

    reloader.set_api_factory(_make_api)
    services["slash_commands"] = HarnessCommandDispatcher(
        commands=commands,
        owners_by_kind=reloader.owners_by_kind,
        apis=apis,
        fallback_owner="agentm.extensions.builtin.slash_commands",
    )

    async def _install_with_events(
        module_path: str,
        ext_cfg: dict[str, Any],
        *,
        is_provider: bool = False,
    ) -> None:
        await bus.emit(
            ExtensionInstallEvent.CHANNEL,
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
                ExtensionInstallEvent.CHANNEL,
                ExtensionInstallEvent(
                    module_path=module_path,
                    config=dict(ext_cfg),
                    phase="error",
                    duration_ns=time.perf_counter_ns() - t0,
                    error=repr(exc),
                ),
            )
            raise
        reloader.record_loaded_atom(module_path, ext_cfg, is_provider=is_provider)
        await bus.emit(
            ExtensionInstallEvent.CHANNEL,
            ExtensionInstallEvent(
                module_path=module_path,
                config=dict(ext_cfg),
                phase="end",
                duration_ns=time.perf_counter_ns() - t0,
            ),
        )

    await _prime_contrib_discovery(config, bus)
    to_load = await _resolve_extensions(config, bus)

    for module_path, ext_cfg in to_load:
        try:
            await _install_with_events(module_path, ext_cfg)
        except Exception as exc:  # noqa: BLE001
            await bus.emit(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(
                    level="error",
                    source="extension_loader",
                    message=f"{module_path}: {exc}",
                ),
            )

    if config.provider is None:
        raise ExtensionLoadError(
            "<provider>",
            RuntimeError(
                "AgentSessionConfig.provider is None. Root sessions must specify "
                "a provider explicitly; only spawn_child_session auto-fills None "
                "with the inherit_provider builtin."
            ),
        )
    provider_path, provider_cfg = config.provider
    await _install_with_events(provider_path, provider_cfg, is_provider=True)

    if not providers:
        raise ExtensionLoadError(
            provider_path,
            RuntimeError("provider extension did not call api.register_provider"),
        )

    if config.tool_allowlist is not None:
        tools[:] = [t for t in tools if t.name in config.tool_allowlist]

    active_provider = resolve_provider_config(
        providers, provider_resolver, provider_path=provider_path
    )
    active_provider_box["value"] = active_provider

    loop = AgentLoop(
        stream_fn=active_provider.stream_fn,
        bus=bus,
        config=configured_loop_config,
    )
    loop_box["value"] = loop

    for msg in config.initial_messages:
        session_manager.append_message(msg)

    runtime = SessionRuntime(
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
        services=services,
        reloader=reloader,
        pending_user_messages=pending_user_messages,
    )
    instance = session_cls(
        cwd=config.cwd,
        runtime=runtime,
        session_id=session_id,
        parent_bus=config.parent_bus,
        parent_session_id=config.parent_session_id,
        purpose=config.purpose,
    )

    await bus.emit(
        SessionReadyEvent.CHANNEL,
        SessionReadyEvent(
            cwd=config.cwd,
            session_id=session_id,
            tool_names=tuple(t.name for t in tools),
            command_names=tuple(commands.keys()),
            extension_module_paths=tuple(module_path for module_path, _ in to_load),
            model=active_provider.model,
            root_session_id=config.root_session_id or session_id,
            task_id=config.task_id,
            persona=config.persona,
        ),
    )

    if config.parent_bus is not None:
        await config.parent_bus.emit(
            ChildSessionStartEvent.CHANNEL,
            ChildSessionStartEvent(
                child_session_id=session_id,
                parent_session_id=config.parent_session_id or "unknown",
                purpose=config.purpose,
            ),
        )

    return instance


def _configure_manifest(cwd: str) -> None:
    try:
        from agentm.core._internal.catalog import manifest as _manifest_mod

        if _manifest_mod._MANIFEST_PATH is None:
            manifest_path = Path(cwd) / "core-manifest.yaml"
            if manifest_path.exists():
                _manifest_mod.configure_manifest_path(manifest_path)
    except Exception as exc:
        logger.warning(
            "agentm core-manifest configuration failed during startup: %r", exc
        )


def _migrate_catalog(cwd: str) -> None:
    try:
        from agentm.harness.catalog.migrate import migrate_catalog_v2

        migrate_catalog_v2(root=Path(cwd))
    except Exception as exc:
        logger.warning("agentm catalog migration failed during startup: %r", exc)


async def _prime_contrib_discovery(config: AgentSessionConfig, bus: EventBus) -> None:
    if config.no_extensions:
        return
    try:
        discover_mod.discover_contrib_atoms()
    except Exception as exc:  # noqa: BLE001
        await bus.emit(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="error",
                source="auto_discovery",
                message=f"contrib atom discovery failed: {exc}",
            ),
        )


async def _resolve_extensions(
    config: AgentSessionConfig, bus: EventBus
) -> list[tuple[str, dict[str, Any]]]:
    floor_atoms = discover_mod.discover_builtin()
    slash_commands_module = floor_atoms["slash_commands"].module_path
    compaction_prompts_module = floor_atoms["compaction_prompts"].module_path
    system_prompt_module = floor_atoms["system_prompt"].module_path
    if config.no_extensions:
        to_load: list[tuple[str, dict[str, Any]]] = []
    elif config.extensions:
        to_load = list(config.extensions)
        ensure_floor_atom(to_load, compaction_prompts_module)
        ensure_floor_atom(to_load, slash_commands_module)
    elif config.scenario is not None:
        from agentm.extensions.loader import ScenarioLoadError, load_scenario

        try:
            to_load = load_scenario(config.scenario)
        except (ScenarioLoadError, Exception) as exc:  # noqa: BLE001
            await bus.emit(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(
                    level="error",
                    source="scenario_loader",
                    message=str(exc),
                ),
            )
            to_load = []
        ensure_floor_atom(to_load, compaction_prompts_module)
        ensure_floor_atom(to_load, slash_commands_module)
    else:
        to_load = await collect_auto_discovered_atoms(
            bus=bus,
            sources=(
                AtomSource(
                    label="builtin",
                    discover=discover_mod.discover_builtin,
                ),
                AtomSource(
                    label="contrib",
                    discover=discover_mod.discover_contrib_atoms,
                    skip_label="contrib atom ",
                ),
                AtomSource(
                    label="user",
                    discover=lambda: discover_mod.discover_user_atoms(
                        Path(config.cwd)
                    ),
                    skip_label="user atom ",
                ),
            ),
        )
        ensure_floor_atom(to_load, system_prompt_module)
        for index, (module_path, _cfg) in enumerate(to_load):
            if module_path == system_prompt_module:
                to_load[index] = (module_path, {"prompt": ""})
                break

    if not config.no_extensions and not config.extensions and config.extra_extensions:
        to_load.extend(config.extra_extensions)
    if not config.no_extensions:
        loaded_modules = {module_path for module_path, _cfg in to_load}
        sub_agent_module = floor_atoms["sub_agent"].module_path
        if sub_agent_module in loaded_modules and system_prompt_module not in loaded_modules:
            to_load.insert(0, (system_prompt_module, {"prompt": ""}))

        from agentm.extensions.loader import sort_extensions_by_requires

        to_load = sort_extensions_by_requires(to_load, source="session extensions")
    return to_load


__all__ = ["create_agent_session"]
