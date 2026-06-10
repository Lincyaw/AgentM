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
from agentm.core.abi.roles import (
    COMMAND_PARSER,
    COMPACTION_PROMPTS,
    LOOP_BUDGET_SERVICE,
    MODEL_RESOLVER_SERVICE,
    PROMPT_REGISTRY,
    SESSION_STORE_SERVICE,
    SLASH_COMMAND_DISPATCHER_SERVICE,
    SUB_AGENT_RUNTIME,
    SYSTEM_PROMPT_PROVIDER,
)
from agentm.core.runtime.atom_reloader import AtomReloader
from agentm.core.runtime.atom_sandbox import apply_atom_source_overrides
from agentm.core.runtime.command_dispatcher import HarnessCommandDispatcher
from agentm.core.abi.events import (
    ChildSessionExtendingEvent,
    ChildSessionStartEvent,
    ExtensionInstallEvent,
    SessionReadyEvent,
)
from agentm.core.runtime.extension import (
    CommandSpec,
    ExtensionLoadError,
    ProviderConfig,
    ReadonlySession,
    Renderer,
    _ExtensionAPIImpl,
    build_extension_api_scope,
    load_extension,
)
from agentm.core.runtime.provider_resolver import LastRegisteredWins
from agentm.core.runtime.resource_loader import InMemoryResourceLoader, ResourceLoader
from agentm.core.runtime.resource_writer import GitBackedResourceWriter
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session_helpers import (
    AtomSource,
    SessionView,
    collect_auto_discovered_atoms,
    ensure_floor_atom,
    resolve_provider_config,
)
from agentm.core.runtime.session_inbox import SessionInbox
from agentm.core.runtime.session_manager import InMemorySessionManager, SessionManager
from agentm.core.runtime.session_runtime import SessionRuntime

logger = logging.getLogger(__name__)


def default_child_provider_factory(parent_provider: Any) -> tuple[str, dict[str, Any]]:
    """Return the spec for whichever atom claims the ``PROVIDER_INHERITOR``
    role. Looking up by role rather than by atom name lets a scenario ship
    a customised provider-inheritor without editing the runtime.

    Moved here from ``core.abi.session_config`` to avoid an ``extensions``
    import in the ABI layer (recovery-floor violation).
    """

    from agentm.core.abi.roles import PARENT_PROVIDER_CONFIG_KEY, PROVIDER_INHERITOR
    from agentm.extensions import discover as discover_mod

    entry = discover_mod.discover_by_role().get(PROVIDER_INHERITOR)
    if entry is None:
        raise RuntimeError(
            f"no atom claims the {PROVIDER_INHERITOR!r} role; cannot build "
            "a child-session provider spec"
        )
    return (entry.module_path, {PARENT_PROVIDER_CONFIG_KEY: parent_provider})


def apply_child_session_contributions(
    base_extensions: list[tuple[str, dict[str, Any]]],
    handler_returns: list[Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Concatenate handler-contributed extension entries onto a child's load
    order, dedupe by ``module_path``.

    See :class:`agentm.core.abi.events.ChildSessionExtendingEvent`. Each
    element of ``handler_returns`` is either ``None`` (no opinion) or an
    iterable of ``(module_path, config)`` tuples. Order is preserved:
    the operator-supplied entries on ``base_extensions`` come first,
    then handler returns in registration order. The first occurrence of
    each module wins; later duplicates are dropped silently so handlers
    don't have to dedupe themselves.

    Exposed at module top-level (not nested inside the factory) so the
    fail-stop test in ``tests/unit/core/test_child_session_extending_event.py``
    can drive the dedupe logic directly without standing up a real
    session.
    """
    result: list[tuple[str, dict[str, Any]]] = list(base_extensions)
    seen: set[str] = {entry[0] for entry in result if isinstance(entry, tuple) and entry}
    for ret in handler_returns:
        if ret is None:
            continue
        # Permissive: accept any iterable of (module, config) — handlers
        # may return list, tuple, or generator.
        try:
            iterator = list(ret)
        except TypeError:
            continue
        for entry in iterator:
            if not (isinstance(entry, tuple) and len(entry) == 2):
                continue
            module, cfg = entry
            if not isinstance(module, str) or module in seen:
                continue
            if not isinstance(cfg, dict):
                continue
            result.append((module, cfg))
            seen.add(module)
    return result


async def create_agent_session(
    session_cls: type[Any], config: AgentSessionConfig
) -> Any:
    from agentm.extensions import discover as discover_mod

    bus = config.bus if config.bus is not None else EventBus()
    session_manager: SessionManager = (
        config.session_manager
        if config.session_manager is not None
        else InMemorySessionManager(cwd=config.cwd)
    )
    # SDK default is empty — no filesystem walks, no implicit context.
    # Callers that want disk-resident skills / prompt templates /
    # AGENTS.md / CLAUDE.md must construct ``DefaultResourceLoader(cwd=...)``
    # explicitly. The CLI does this; embedded SDK and child sessions
    # get a clean slate so their LLM context isn't contaminated by
    # whatever happens to be in the parent project's ``CLAUDE.md``.
    resource_loader: ResourceLoader = (
        config.resource_loader
        if config.resource_loader is not None
        else InMemoryResourceLoader()
    )

    tools: list[Tool] = []
    commands: dict[str, CommandSpec] = {}
    providers: dict[str, ProviderConfig] = {}
    renderers: dict[str, Renderer] = {}
    inbox = SessionInbox()
    apis: dict[str, _ExtensionAPIImpl] = {}
    services: dict[str, Any] = {}
    # Seed caller-supplied services BEFORE atoms install.
    # See AgentSessionConfig.initial_services.
    services.update(config.initial_services)
    if SESSION_STORE_SERVICE not in services:
        from agentm.core.runtime.session_bootstrap import make_default_session_store
        services[SESSION_STORE_SERVICE] = make_default_session_store(config.cwd)

    if MODEL_RESOLVER_SERVICE not in services:
        def _resolve_model(model_name: str) -> tuple[str, dict[str, Any]] | None:
            from agentm.ai import DEFAULT_PROVIDER_REGISTRY
            from agentm.core.lib.user_config import resolve_model_profile

            profile = resolve_model_profile(model_name)
            if profile is None:
                return None
            build_config = profile.to_build_config()
            provider_id = profile.provider
            try:
                return DEFAULT_PROVIDER_REGISTRY.build(provider_id, build_config)
            except KeyError:
                return None

        services[MODEL_RESOLVER_SERVICE] = _resolve_model

    active_provider_box: dict[str, ProviderConfig | None] = {"value": None}
    loop_box: dict[str, AgentLoop | None] = {"value": None}
    # Effective loop budget is resolved after atoms install: the ``loop_budget``
    # atom (when a scenario lists it) registers a ``LoopConfig`` under
    # ``LOOP_BUDGET_SERVICE``; the substrate reads it just before building the
    # loop. Precedence: explicit caller override (CLI ``--max-turns`` / SDK
    # ``loop_config=``) > loop_budget atom > ``LoopConfig()`` default (no cap).
    # Seed with the explicit override or the default; the atom layer is applied
    # below once the service registry is populated.
    loop_config_box: dict[str, LoopConfig] = {
        "value": config.loop_config or LoopConfig()
    }
    provider_resolver = config.provider_resolver or LastRegisteredWins()
    child_provider_factory = (
        config.child_provider_factory or default_child_provider_factory
    )

    def _model_getter() -> Model | None:
        cur = active_provider_box["value"]
        return cur.model if cur is not None else None

    def _provider_getter() -> ProviderConfig | None:
        return active_provider_box["value"]

    # OTel-native identity:
    #   * ``session_id`` is this session's *span_id* — 8 bytes / 16 hex.
    #   * ``root_session_id`` is the *trace_id* shared across the whole
    #     agent tree — 16 bytes / 32 hex. For a fresh root session we
    #     generate a new trace_id; child sessions inherit their parent's
    #     (the spawn factory below threads it onto ``spec.root_session_id``).
    #   * ``parent_session_id`` is the parent's span_id and only matters
    #     for children (root sessions get ``None``).
    # Honour caller-supplied ids so external systems can reuse their own
    # trace / span identifiers verbatim — the JSONL filename becomes
    # ``<session_id>.jsonl`` and the OTel ``trace_id`` field on every
    # event is the caller's trace_id. Falls back to fresh uuid hex of the
    # appropriate length when no id is supplied.
    # Single-event-log merge: observability writes the merged event log to
    # ``<cwd>/.agentm/observability/<session_id>.jsonl`` and SessionManager
    # is the authoritative source of the session header id, so the filename
    # lines up with what ``continue_recent`` looks for. Pull from the
    # manager when no caller-supplied id was provided.
    session_id = (
        config.session_id
        or session_manager.get_session_id()
        or uuid.uuid4().hex[:16]
    )
    root_session_id = config.root_session_id or uuid.uuid4().hex
    session_view: ReadonlySession = SessionView(
        session_manager,
        loop_config_getter=lambda: loop_config_box["value"],
        bus=bus,
    )
    from agentm.core.runtime.resource_writer import DEFAULT_PROTECTED_BRANCHES

    resource_writer = config.resource_writer or GitBackedResourceWriter(
        cwd=config.cwd,
        session_id=session_id,
        bus=bus,
        auto_commit=config.auto_commit,
        protected_branches=(
            config.protected_branches
            if config.protected_branches is not None
            else DEFAULT_PROTECTED_BRANCHES
        ),
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
        # Child inherits the *parent's* trace_id (root_session_id) so the
        # whole agent tree shares one OTel trace. Never fall back to the
        # parent's session_id — that would mint a fresh trace per child
        # and undo the unification.
        spec.root_session_id = root_session_id
        if spec.provider is None:
            parent_provider = _provider_getter()
            if parent_provider is None:
                raise RuntimeError(
                    "spawn_child_session: AgentSessionConfig.provider is None but "
                    "the parent session has no active provider to inherit."
                )
            spec.provider = child_provider_factory(parent_provider)

        # Give extensions on the parent bus a chance to contribute atoms
        # to the child's load order. ``emit_sync`` because we need the
        # contributions resolved BEFORE ``session_cls.create`` runs; the
        # event is delivered on the parent bus only — child sessions don't
        # observe their own extending pass.
        #
        # We clone the extensions list before emit so a misbehaving
        # handler mutating ``ev.child_config.extensions`` in place cannot
        # affect what the factory sees: only the substrate-controlled
        # ``spec.extensions`` is honoured below.
        returns = bus.emit_sync(
            ChildSessionExtendingEvent.CHANNEL,
            ChildSessionExtendingEvent(
                parent_session_id=session_id,
                child_config=spec,
            ),
        )
        spec.extensions = apply_child_session_contributions(
            list(spec.extensions), returns
        )

        return await session_cls.create(spec)

    scope = build_extension_api_scope(
        bus=bus,
        cwd=config.cwd,
        session_id=session_id,
        root_session_id=root_session_id,
        parent_session_id=config.parent_session_id,
        purpose=config.purpose,
        scenario=config.scenario,
        session=session_view,
        tools=tools,
        commands=commands,
        providers=providers,
        renderers=renderers,
        inbox=inbox,
        model_getter=_model_getter,
        provider_getter=_provider_getter,
        gateway=reloader,
        child_session_factory=_spawn_child_session,
        resource_writer=resource_writer,
        service_registry=services,
    )

    def _make_api(owner: str) -> _ExtensionAPIImpl:
        api = _ExtensionAPIImpl(scope, owner_name=owner)
        reloader.wrap_api_on(api, owner)
        apis[owner] = api
        return api

    reloader.set_api_factory(_make_api)
    command_parser_entry = discover_mod.discover_by_role().get(COMMAND_PARSER)
    services[SLASH_COMMAND_DISPATCHER_SERVICE] = HarnessCommandDispatcher(
        commands=commands,
        owners_by_kind=reloader.owners_by_kind,
        apis=apis,
        fallback_owner=(
            command_parser_entry.module_path
            if command_parser_entry is not None
            else "<no-command-parser>"
        ),
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

    if scope.operations.bundle is None:
        raise ExtensionLoadError(
            "<operations>",
            RuntimeError(
                "no atom registered Operations; the active scenario manifest "
                "must list an atom that calls api.register_operations(...) "
                "(default: agentm.extensions.builtin.operations)"
            ),
        )

    if config.tool_allowlist is not None:
        tools[:] = [t for t in tools if t.name in config.tool_allowlist]

    active_provider = resolve_provider_config(
        providers, provider_resolver, provider_path=provider_path
    )
    active_provider_box["value"] = active_provider

    # Apply the loop_budget atom's registration, if any. An explicit caller
    # override (CLI flag / SDK ``loop_config=``) wins wholesale and is left
    # untouched; otherwise the atom-supplied budget replaces the default.
    if config.loop_config is None:
        registered_loop = services.get(LOOP_BUDGET_SERVICE)
        if isinstance(registered_loop, LoopConfig):
            loop_config_box["value"] = registered_loop

    loop = AgentLoop(
        stream_fn=active_provider.stream_fn,
        bus=bus,
        config=loop_config_box["value"],
    )
    loop_box["value"] = loop

    # Single-event-log merge (.claude/designs/single-event-log.md):
    # SessionManager no longer writes its own JSONL. Attaching it to the
    # bus here — after the extension-load loop installed observability —
    # flushes any SessionHeaderEmittedEvent buffered during
    # SessionManager.__init__ before any message rows, and routes every
    # subsequent ``append_message`` through the merged log.
    session_manager.attach_bus(bus)

    for msg in config.initial_messages:
        session_manager.append_message(msg)

    # Apply ``atom_source_overrides`` BEFORE building the AgentSession
    # façade so the running atoms reflect the overridden source by the
    # time the loop is invoked. The helper redirects each atom's
    # ``file_path`` to ``.agentm/eval-sandbox/<id>/`` (classified as
    # ``unmanaged`` by ResourceWriter — no git mutation). Cleaned up by
    # ``AgentSession.shutdown``.
    eval_sandbox = await apply_atom_source_overrides(
        reloader=reloader,
        bus=bus,
        resource_writer=resource_writer,
        cwd=config.cwd,
        session_id=session_id,
        overrides=config.atom_source_overrides or {},
    )

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
        inbox=inbox,
    )
    instance = session_cls(
        cwd=config.cwd,
        runtime=runtime,
        session_id=session_id,
        parent_bus=config.parent_bus,
        parent_session_id=config.parent_session_id,
        eval_sandbox=eval_sandbox,
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
            # Use the canonical ``root_session_id`` computed at line 119
            # (uuid4().hex when not supplied) — falling back to
            # ``session_id`` here would emit a 16-hex trace_id on the
            # bus while the observability sink writes the 32-hex one,
            # silently breaking any consumer that joins both.
            root_session_id=root_session_id,
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
    # Each session binds its own path on the ContextVar — asyncio tasks
    # copy the context at creation, so concurrent sessions in different
    # cwds do not race. Sequential ``create_agent_session`` calls inside
    # the same task overwrite the binding, which matches the desired
    # "current session's cwd wins" semantics.
    try:
        from agentm.core._internal.catalog import manifest as _manifest_mod

        manifest_path = Path(cwd) / "core-manifest.yaml"
        if manifest_path.exists():
            _manifest_mod.configure_manifest_path(manifest_path)
    except Exception as exc:
        logger.warning(
            "agentm core-manifest configuration failed during startup: %r", exc
        )


def _migrate_catalog(cwd: str) -> None:
    try:
        from agentm.core.runtime.catalog.migrate import migrate_catalog_v2

        migrate_catalog_v2(root=Path(cwd))
    except Exception as exc:
        logger.warning("agentm catalog migration failed during startup: %r", exc)


async def _prime_contrib_discovery(config: AgentSessionConfig, bus: EventBus) -> None:
    if config.no_extensions:
        return
    from agentm.extensions import discover as discover_mod

    for label, discover_fn in [
        ("contrib", discover_mod.discover_contrib_atoms),
        ("home", discover_mod.discover_home_atoms),
    ]:
        try:
            discover_fn()
        except Exception as exc:  # noqa: BLE001
            await bus.emit(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(
                    level="error",
                    source="auto_discovery",
                    message=f"{label} atom discovery failed: {exc}",
                ),
            )


async def _resolve_extensions(
    config: AgentSessionConfig, bus: EventBus
) -> list[tuple[str, dict[str, Any]]]:
    from agentm.extensions import discover as discover_mod

    roles = discover_mod.discover_by_role()

    def _role_module(role: str) -> str:
        entry = roles.get(role)
        if entry is None:
            raise RuntimeError(
                f"floor role {role!r} has no atom — no builtin/contrib atom "
                "declares this role in MANIFEST.provides_role"
            )
        return entry.module_path

    command_parser_module = _role_module(COMMAND_PARSER)
    compaction_prompts_module = _role_module(COMPACTION_PROMPTS)
    prompt_registry_module = _role_module(PROMPT_REGISTRY)
    system_prompt_module = _role_module(SYSTEM_PROMPT_PROVIDER)
    sub_agent_runtime_entry = roles.get(SUB_AGENT_RUNTIME)
    if config.no_extensions:
        to_load: list[tuple[str, dict[str, Any]]] = []
    elif config.extensions:
        to_load = list(config.extensions)
        ensure_floor_atom(to_load, prompt_registry_module)
        ensure_floor_atom(to_load, compaction_prompts_module)
        ensure_floor_atom(to_load, command_parser_module)
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
        ensure_floor_atom(to_load, prompt_registry_module)
        ensure_floor_atom(to_load, compaction_prompts_module)
        ensure_floor_atom(to_load, command_parser_module)
        # Layer ``<cwd>/.agentm/atoms/`` agent-installed atoms on top of the
        # scenario. Without this merge, ``api.install_atom`` calls would only
        # take effect for the lifetime of one session — the next process
        # start would load the scenario, see no user atoms, and forget
        # everything the agent installed previously, even though the source
        # files are still on disk. Skip duplicates so a scenario that
        # explicitly lists a user-atom module wins over the auto-discovered
        # entry (preserves config).
        user_atoms = await collect_auto_discovered_atoms(
            bus=bus,
            sources=(
                AtomSource(
                    label="home",
                    discover=discover_mod.discover_home_atoms,
                    skip_label="home atom ",
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
        existing_modules = {module for module, _ in to_load}
        for module_path, atom_config in user_atoms:
            if module_path not in existing_modules:
                to_load.append((module_path, atom_config))
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
                    label="home",
                    discover=discover_mod.discover_home_atoms,
                    skip_label="home atom ",
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
        existing_modules = {m for m, _ in to_load}
        for module_path, ext_cfg in config.extra_extensions:
            if module_path not in existing_modules:
                to_load.append((module_path, ext_cfg))
                existing_modules.add(module_path)
    if not config.no_extensions:
        loaded_modules = {module_path for module_path, _cfg in to_load}
        # The sub-agent runtime injects inherited prompt text via the
        # system-prompt hook, so any session that loads it needs an atom
        # filling the SYSTEM_PROMPT_PROVIDER role even if the scenario
        # author forgot to list one.
        if (
            sub_agent_runtime_entry is not None
            and sub_agent_runtime_entry.module_path in loaded_modules
            and system_prompt_module not in loaded_modules
        ):
            to_load.insert(0, (system_prompt_module, {"prompt": ""}))

        from agentm.core.lib.atom_config import (
            AtomConfigError,
            resolve_atom_configs,
        )
        from agentm.extensions.loader import sort_extensions_by_requires

        # Bind env (AGENTM_<ATOM>_<KEY>) and --set overrides on top of the
        # manifest-supplied config before install. Done once here so every
        # presenter (CLI, channels, embedded SDK) gets identical semantics.
        # A malformed value degrades like a scenario-load failure: emit a
        # diagnostic and fall back to the manifest configs rather than
        # aborting the whole session build.
        try:
            to_load = resolve_atom_configs(
                to_load, overrides=config.atom_config_overrides
            )
        except AtomConfigError as exc:
            await bus.emit(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(
                    level="error",
                    source="atom_config",
                    message=str(exc),
                ),
            )
        to_load = sort_extensions_by_requires(to_load, source="session extensions")
    return to_load


__all__ = ["create_agent_session"]
