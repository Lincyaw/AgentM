"""Build SDK sessions from explicit extension specs.

The factory owns runtime construction only. It does not search source trees,
home directories, or package contrib locations. Callers either pass
``extensions`` directly or provide a ``ScenarioLoader`` that resolves a
scenario name into extension specs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from agentm.core.abi.bus import EventBus
from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.cancel import CompositeCancelSignal
from agentm.core.abi.codec import CodecRegistry
from agentm.core.abi.context import ContextPolicy
from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    AtomActivation,
    AtomCatalog,
    AtomCatalogQuery,
    CatalogActiveSetInput,
    ResourceVersion,
    VersionedResourceStore,
)
from agentm.core.abi.errors import ExtensionLoadError
from agentm.core.abi.lifecycle import EffectScope, EnvironmentRestoreFailureHandler
from agentm.core.abi.messages import freeze_json
from agentm.core.abi.operations import EnvironmentOperations
from agentm.core.abi.manifest import (
    AtomInstallPriority,
    ExtensionManifest,
    provided_capability_keys,
    requirement_key,
)
from agentm.core.abi.permission import PermissionPolicy
from agentm.core.abi.provider import ProviderResolver
from agentm.core.abi.provider import ProviderSessionIdentity
from agentm.core.abi.resource import ResourceReader, ResourceStore, ResourceWriter
from agentm.core.abi.tool_executor import ToolExecutor
from agentm.core.abi.tool_orchestration import ToolOrchestrator
from agentm.core.abi.roles import (
    ACTIVE_SET_FINGERPRINT_SERVICE,
    ATOM_CATALOG_SERVICE,
    CATALOG_QUERY_SERVICE,
    ENVIRONMENT_RESTORE_FAILURE_HANDLER_SERVICE,
    LOOP_BUDGET_SERVICE,
    PROVIDER_RESOLVER_SERVICE,
    RESOLVED_SESSION_SPEC_SERVICE,
    SCENARIO_LOADER_SERVICE,
    SESSION_SPEC_RESOLVER_SERVICE,
    TRAJECTORY_QUERY_STORE_SERVICE,
    VERSIONED_RESOURCE_STORE_SERVICE,
)
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.session_api import (
    ExtensionInput,
    ExtensionSpec,
    ResolvedSessionSpec,
    ScenarioLoader,
    ScenarioSpec,
    SessionContext,
    normalize_extension_spec,
)
from agentm.core.abi.store import SessionMeta, TrajectoryNodeStore, TrajectoryStore
from agentm.core.abi.stream import Model, StreamFn
from agentm.core.abi.tool import Tool
from agentm.core.abi.trajectory import Turn
from agentm.core.abi.trajectory import TurnRef
from agentm.core.abi.tree import SessionGraphProtocol
from agentm.core.abi.trigger import TriggerRenderer
from agentm.core.runtime.catalog import (
    InMemoryAtomCatalog,
    InMemoryVersionedResourceStore,
    build_atom_identity_payload,
    normalize_atom_config,
)
from agentm.core.runtime.driver import ThinkingLevel
from agentm.core.runtime.extension import (
    install_extension,
    load_extension_module,
)
from agentm.core.runtime.session import Session, SessionRuntimeConfig
from agentm.core.runtime.session_meta import session_meta_config
from agentm.core.runtime.stores.query import TrajectoryStoreQueryAdapter
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.lib.store_resolve import resolve_trajectory_store_or_create

if TYPE_CHECKING:
    from agentm.core.abi.session_api import AgentSessionConfig


@dataclass(frozen=True, slots=True)
class _ExtensionPlanItem:
    spec: ExtensionSpec
    module_path: str
    config: dict[str, Any]
    index: int
    name: str
    manifest: ExtensionManifest | None
    requires: tuple[str, ...]
    registers: tuple[str, ...]
    priority: int
    provides: tuple[str, ...]


def _copy_extension_specs(specs: Sequence[ExtensionInput]) -> list[ExtensionSpec]:
    return [normalize_extension_spec(spec) for spec in specs]


def _normalize_scenario_result(
    result: ScenarioSpec | Sequence[ExtensionInput],
) -> tuple[list[ExtensionSpec], str | None]:
    if isinstance(result, ScenarioSpec):
        return _copy_extension_specs(result.extensions), result.base_dir
    return _copy_extension_specs(result), None


def _resolve_scenario(
    scenario: str,
    loader: ScenarioLoader | None,
) -> tuple[list[ExtensionSpec], str | None]:
    if loader is not None:
        return _normalize_scenario_result(loader(scenario))
    raise ValueError(
        f"cannot resolve scenario {scenario!r}; pass extensions directly or provide "
        "AgentSessionConfig.scenario_loader"
    )


def _resolve_extensions(
    *,
    scenario: str | None,
    extensions: Sequence[ExtensionInput] | None,
    extra_extensions: Sequence[ExtensionInput],
    atom_configs: dict[str, dict[str, Any]] | None,
    scenario_loader: ScenarioLoader | None,
) -> tuple[list[ExtensionSpec], str | None, str | None]:
    scenario_name = scenario
    resolved: list[ExtensionSpec]
    if extensions is None:
        if scenario_name is None:
            resolved, base_dir = [], None
        else:
            resolved, base_dir = _resolve_scenario(scenario_name, scenario_loader)
    else:
        resolved = _copy_extension_specs(extensions)
        base_dir = None

    resolved.extend(_copy_extension_specs(extra_extensions))
    if atom_configs:
        configured: list[ExtensionSpec] = []
        for spec in resolved:
            manifest = _load_manifest(spec)
            atom_name = manifest.name if manifest is not None else spec.module_path
            configured.append(
                spec.with_config(
                    {
                        **spec.config,
                        **atom_configs.get(atom_name, {}),
                    }
                )
            )
        resolved = configured
    return resolved, base_dir, scenario_name


def _load_manifest(spec: ExtensionSpec) -> ExtensionManifest | None:
    module_path = spec.module_path
    module = load_extension_module(spec)
    manifest = module.__dict__.get("MANIFEST")
    if manifest is None:
        return None
    if not isinstance(manifest, ExtensionManifest):
        raise ExtensionLoadError(
            module_path,
            TypeError("MANIFEST must be an ExtensionManifest"),
        )
    return manifest


def _service_capabilities(services: ServiceRegistry | None) -> set[str]:
    if services is None:
        return set()
    return {f"service:{name}" for name in services.names()}


def _extension_plan(
    specs: Sequence[ExtensionSpec],
    *,
    available_capabilities: set[str] | None = None,
) -> list[_ExtensionPlanItem]:
    items: list[_ExtensionPlanItem] = []
    by_name: dict[str, _ExtensionPlanItem] = {}
    for index, spec in enumerate(specs):
        module_path = spec.module_path
        manifest = _load_manifest(spec)
        name = manifest.name if manifest is not None else module_path
        item = _ExtensionPlanItem(
            spec=spec,
            module_path=module_path,
            config=dict(spec.config),
            index=index,
            name=name,
            manifest=manifest,
            requires=manifest.requires if manifest is not None else (),
            registers=manifest.registers if manifest is not None else (),
            priority=(
                manifest.priority
                if manifest is not None
                else AtomInstallPriority.NORMAL
            ),
            provides=provided_capability_keys(
                atom_name=name,
                registers=manifest.registers if manifest is not None else (),
            ),
        )
        if name in by_name:
            previous = by_name[name]
            raise ValueError(
                f"duplicate atom {name!r}: {previous.module_path!r} and {module_path!r}"
            )
        items.append(item)
        by_name[name] = item

    available = set(available_capabilities or ())
    plan_capabilities = {capability for item in items for capability in item.provides}
    missing: dict[str, list[str]] = {}
    for item in items:
        for requirement in item.requires:
            key = requirement_key(requirement)
            if key not in available and key not in plan_capabilities:
                missing.setdefault(item.name, []).append(requirement)
    if missing:
        detail = "; ".join(
            f"{name} requires {', '.join(deps)}"
            for name, deps in sorted(missing.items())
        )
        raise ValueError(f"unsatisfied atom dependencies: {detail}")

    remaining = dict(by_name)
    ordered: list[_ExtensionPlanItem] = []
    provided = set(available)
    while remaining:
        ready = [
            item
            for item in remaining.values()
            if all(
                requirement_key(requirement) in provided
                for requirement in item.requires
            )
        ]
        if not ready:
            cycle = ", ".join(sorted(remaining))
            raise ValueError(f"cyclic atom dependencies: {cycle}")
        ready.sort(key=lambda item: (item.priority, item.index))
        chosen = ready[0]
        ordered.append(chosen)
        provided.update(chosen.provides)
        del remaining[chosen.name]
    return ordered


def _get_scenario_loader(services: ServiceRegistry | None) -> ScenarioLoader | None:
    if services is None:
        return None
    candidate = services.get(SCENARIO_LOADER_SERVICE)
    return candidate if callable(candidate) else None


async def _ensure_store_session(
    store: TrajectoryStore | None,
    *,
    meta: SessionMeta,
    initial_turns: Sequence[Turn],
) -> None:
    if store is None:
        return
    exists = await asyncio.to_thread(store.session_exists, meta.id)
    if not exists:
        await asyncio.to_thread(store.create_session_with_turns, meta, initial_turns)


def _register_default_catalog_services(services: ServiceRegistry) -> None:
    if not services.has(VERSIONED_RESOURCE_STORE_SERVICE):
        services.register(
            VERSIONED_RESOURCE_STORE_SERVICE,
            InMemoryVersionedResourceStore(),
            VersionedResourceStore,
            scope="host",
        )
    if not services.has(ATOM_CATALOG_SERVICE):
        catalog = InMemoryAtomCatalog()
        services.register(
            ATOM_CATALOG_SERVICE,
            catalog,
            AtomCatalog,
            scope="host",
        )
        services.register(
            CATALOG_QUERY_SERVICE,
            catalog,
            AtomCatalogQuery,
            scope="host",
        )


def _register_default_query_store(
    services: ServiceRegistry,
    store: TrajectoryStore | None,
) -> None:
    if store is None or services.has(TRAJECTORY_QUERY_STORE_SERVICE):
        return
    services.register(
        TRAJECTORY_QUERY_STORE_SERVICE,
        TrajectoryStoreQueryAdapter(store),
        scope="resource",
    )


def _resolve_session_spec(config: "AgentSessionConfig") -> ResolvedSessionSpec | None:
    resolver = config.spec_resolver
    if resolver is None:
        return None
    return resolver.resolve(config)


def _resolved_atom_config(
    spec: ResolvedSessionSpec | None,
    fallback: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if spec is None:
        return fallback
    return {module: dict(config) for module, config in spec.atom_config.items()}


def _config_fingerprint(config: Any) -> str | None:
    if not config:
        return None
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


async def _freeze_atom_version(
    store: VersionedResourceStore,
    item: _ExtensionPlanItem,
) -> ResourceVersion:
    content, metadata = build_atom_identity_payload(
        source=item.spec.source,
        manifest=item.manifest,
        config=item.config,
    )
    version = await store.put(
        resource_id=f"atom:{item.name}",
        content=content,
        media_type="application/vnd.agentm.atom-identity+json",
        metadata=metadata,
    )
    await store.alias(f"atom:{item.name}:active", version)
    return version


def _atom_activation(
    item: _ExtensionPlanItem,
    *,
    version: ResourceVersion | None = None,
) -> AtomActivation:
    return AtomActivation(
        name=item.name,
        module_path=item.module_path,
        version=version,
        priority=item.priority,
        requires=tuple(item.requires),
        registers=tuple(item.registers),
        required_capabilities=tuple(
            requirement_key(requirement) for requirement in item.requires
        ),
        provided_capabilities=tuple(item.provides),
        config_fingerprint=_config_fingerprint(
            normalize_atom_config(item.manifest, item.config)
        ),
    )


async def _record_active_set(
    session: Session,
    plan: Sequence[_ExtensionPlanItem],
    *,
    created_at: float,
) -> ActiveSetFingerprint | None:
    catalog = session.get_atom_catalog()
    if catalog is None:
        return None
    version_store = session.get_versioned_resource_store()
    activations: list[AtomActivation] = []
    for item in plan:
        version = (
            await _freeze_atom_version(version_store, item)
            if version_store is not None
            else None
        )
        activations.append(_atom_activation(item, version=version))
    provider_identity = session.provider_session_identity()
    fingerprint = await catalog.record_active_set(
        CatalogActiveSetInput(
            session_id=session.id,
            root_session_id=session.ctx.root_session_id,
            parent_session_id=session.ctx.parent_session_id,
            scenario=session.ctx.scenario,
            provider=provider_identity.name if provider_identity is not None else None,
            created_at=created_at,
            atoms=tuple(activations),
        )
    )
    session.services.register(
        ACTIVE_SET_FINGERPRINT_SERVICE,
        fingerprint,
        ActiveSetFingerprint,
        scope="session",
    )
    return fingerprint


@dataclass(slots=True)
class SessionBuildConfig:
    """Normalized composition and runtime inputs for the low-level factory."""

    scenario: str | None = None
    stream_fn: StreamFn | None = None
    model: Model | None = None
    system: str | None = None
    cwd: str = ""
    purpose: str = "root"
    store: TrajectoryStore | None = None
    graph: SessionGraphProtocol | None = None
    session_context: SessionContext | None = None
    session_id: str | None = None
    root_session_id: str | None = None
    parent_session_id: str | None = None
    bus: EventBus | None = None
    initial_turns: list[Turn] | None = None
    fork_point: TurnRef | None = None
    tools: list[Tool] | None = None
    context_policies: list[ContextPolicy] | None = None
    trigger_renderers: dict[str, TriggerRenderer] | None = None
    codec: CodecRegistry | None = None
    extensions: Sequence[ExtensionInput] | None = None
    extra_extensions: Sequence[ExtensionInput] = ()
    provider: ExtensionInput | None = None
    provider_resolver: ProviderResolver | None = None
    provider_identity: ProviderSessionIdentity | None = None
    resource_reader: ResourceReader | None = None
    resource_store: ResourceStore | None = None
    resource_writer: ResourceWriter | None = None
    tool_executor: ToolExecutor | None = None
    tool_orchestrator: ToolOrchestrator | None = None
    permission_policy: PermissionPolicy | None = None
    trajectory_node_store: TrajectoryNodeStore | None = None
    effect_scope: EffectScope | None = None
    environment_operations: EnvironmentOperations | None = None
    environment_restore_failure_handler: EnvironmentRestoreFailureHandler | None = None
    versioned_resource_store: VersionedResourceStore | None = None
    atom_catalog: AtomCatalog | None = None
    atom_configs: dict[str, dict[str, Any]] | None = None
    scenario_loader: ScenarioLoader | None = None
    services: ServiceRegistry | None = None
    resolved_spec: ResolvedSessionSpec | None = None
    max_turns: int | None = None
    max_tool_calls: int | None = None
    tool_allowlist: list[str] | None = None
    thinking: ThinkingLevel = "off"
    cancel_signal: CancelSignal | None = None


async def create_session(config: SessionBuildConfig) -> Session:
    """Create a root SDK session."""

    scenario = config.scenario
    stream_fn = config.stream_fn
    model = config.model
    system = config.system
    cwd = config.cwd
    purpose = config.purpose
    store = config.store
    graph = config.graph
    session_context = config.session_context
    session_id = config.session_id
    root_session_id = config.root_session_id
    parent_session_id = config.parent_session_id
    bus = config.bus
    initial_turns = config.initial_turns
    fork_point = config.fork_point
    tools = config.tools
    context_policies = config.context_policies
    trigger_renderers = config.trigger_renderers
    codec = config.codec
    extensions = config.extensions
    extra_extensions = config.extra_extensions
    provider = config.provider
    provider_resolver = config.provider_resolver
    provider_identity = config.provider_identity
    resource_reader = config.resource_reader
    resource_store = config.resource_store
    resource_writer = config.resource_writer
    tool_executor = config.tool_executor
    tool_orchestrator = config.tool_orchestrator
    permission_policy = config.permission_policy
    trajectory_node_store = config.trajectory_node_store
    effect_scope = config.effect_scope
    environment_operations = config.environment_operations
    environment_restore_failure_handler = config.environment_restore_failure_handler
    versioned_resource_store = config.versioned_resource_store
    atom_catalog = config.atom_catalog
    atom_configs = config.atom_configs
    scenario_loader = config.scenario_loader
    services = config.services
    resolved_spec = config.resolved_spec
    max_turns = config.max_turns
    max_tool_calls = config.max_tool_calls
    tool_allowlist = config.tool_allowlist
    thinking = config.thinking
    cancel_signal = config.cancel_signal

    resolved_services = services or ServiceRegistry()
    effective_loader = scenario_loader or _get_scenario_loader(resolved_services)
    extension_specs, scenario_dir, scenario_name = _resolve_extensions(
        scenario=scenario,
        extensions=extensions,
        extra_extensions=extra_extensions,
        atom_configs=atom_configs,
        scenario_loader=effective_loader,
    )
    if effective_loader is not None:
        resolved_services.register(
            SCENARIO_LOADER_SERVICE,
            effective_loader,
            scope="host",
        )
    if provider_resolver is not None:
        resolved_services.register(
            PROVIDER_RESOLVER_SERVICE,
            provider_resolver,
            scope="host",
        )
    _register_default_catalog_services(resolved_services)
    _register_default_query_store(resolved_services, store)

    if session_context is None:
        resolved_session_id = session_id or uuid.uuid4().hex[:16]
        resolved_root_id = root_session_id or resolved_session_id
        ctx = SessionContext(
            session_id=resolved_session_id,
            root_session_id=resolved_root_id,
            parent_session_id=parent_session_id,
            cwd=cwd or "",
            purpose=purpose,
            scenario=scenario_name,
            scenario_dir=scenario_dir,
        )
    else:
        ctx = session_context

    session = Session(
        SessionRuntimeConfig(
            ctx=ctx,
            trajectory=Trajectory(turns=initial_turns),
            bus=bus,
            stream_fn=stream_fn,
            model=model,
            system=system,
            store=store,
            graph=graph,
            tools=list(tools or ()),
            context_policies=list(context_policies or ()),
            trigger_renderers=dict(trigger_renderers or {}),
            codec=codec,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            tool_allowlist=tool_allowlist,
            thinking=thinking,
            cancel_signal=cancel_signal,
            tool_executor=tool_executor,
            tool_orchestrator=tool_orchestrator,
            permission_policy=permission_policy,
            resource_reader=resource_reader,
            resource_store=resource_store,
            trajectory_node_store=trajectory_node_store,
            versioned_resource_store=versioned_resource_store,
            environment_restore_failure_handler=environment_restore_failure_handler,
            provider_identity=provider_identity,
            services=resolved_services,
            cwd=cwd,
            purpose=purpose,
        )
    )
    if resource_writer is not None:
        session.register_resource_writer(resource_writer, replace=True)
    if effect_scope is not None:
        session.register_effect_scope(effect_scope, replace=True)
    if environment_operations is not None:
        session.register_operations(
            environment=environment_operations,
            bash=environment_operations.bash,
        )
    if atom_catalog is not None:
        session.register_atom_catalog(atom_catalog, replace=True)

    plan_specs = list(extension_specs)
    if provider is not None:
        plan_specs.append(normalize_extension_spec(provider))
    plan = _extension_plan(
        plan_specs,
        available_capabilities=_service_capabilities(resolved_services),
    )
    for item in plan:
        await install_extension(session, item.spec)
    created_at = time.time()
    active_set = await _record_active_set(session, plan, created_at=created_at)
    await _ensure_store_session(
        store,
        meta=SessionMeta(
            id=session.id,
            parent_id=ctx.parent_session_id,
            fork_point=fork_point,
            purpose=ctx.purpose,
            cwd=ctx.cwd,
            created_at=created_at,
            config=session_meta_config(
                ctx,
                resolved_spec=resolved_spec,
                active_set=active_set,
                provider_identity=session.provider_session_identity(),
            ),
        ),
        initial_turns=initial_turns or (),
    )

    return session


async def create_from_config(
    config: "AgentSessionConfig",
    *,
    restored_context: SessionContext | None = None,
    restored_provider_identity: ProviderSessionIdentity | None = None,
) -> Session:
    """Create a root session from the public SDK config dataclass."""

    max_turns = config.loop_config.max_turns if config.loop_config else None
    max_tool_calls = config.loop_config.max_tool_calls if config.loop_config else None
    services = ServiceRegistry()
    if config.experiment is not None:
        experiment = freeze_json(config.experiment)
        if not isinstance(experiment, Mapping):
            raise TypeError("experiment config must be a JSON object")
        services.register("experiment", experiment, scope="tree")
    if config.loop_config is not None:
        services.register(LOOP_BUDGET_SERVICE, config.loop_config, scope="session")
    if config.scenario_loader is not None:
        services.register(SCENARIO_LOADER_SERVICE, config.scenario_loader, scope="host")
    if config.spec_resolver is not None:
        services.register(
            SESSION_SPEC_RESOLVER_SERVICE,
            config.spec_resolver,
            scope="host",
        )
    if config.environment_restore_failure_handler is not None:
        services.register(
            ENVIRONMENT_RESTORE_FAILURE_HANDLER_SERVICE,
            config.environment_restore_failure_handler,
            EnvironmentRestoreFailureHandler,
            scope="host",
        )
    resolved_spec = _resolve_session_spec(config)
    if resolved_spec is not None and restored_provider_identity is not None:
        resolved_spec = replace(
            resolved_spec,
            provider_identity=restored_provider_identity,
        )
    if resolved_spec is not None:
        services.register(
            RESOLVED_SESSION_SPEC_SERVICE,
            resolved_spec,
            ResolvedSessionSpec,
            scope="session",
        )
    session = await create_session(
        SessionBuildConfig(
            scenario=(
                resolved_spec.scenario if resolved_spec is not None else config.scenario
            ),
            extensions=(
                list(resolved_spec.extensions)
                if resolved_spec is not None
                else config.extensions
            ),
            extra_extensions=config.extra_extensions,
            provider=resolved_spec.provider
            if resolved_spec is not None
            else config.provider,
            provider_resolver=config.provider_resolver,
            provider_identity=(
                restored_provider_identity
                if restored_provider_identity is not None
                else (
                    resolved_spec.provider_identity
                    if resolved_spec is not None
                    else None
                )
            ),
            stream_fn=config.stream_fn,
            model=config.model,
            system=config.system,
            resource_reader=config.resource_reader,
            resource_store=config.resource_store,
            resource_writer=config.resource_writer,
            tool_executor=config.tool_executor,
            tool_orchestrator=config.tool_orchestrator,
            permission_policy=config.permission_policy,
            trajectory_node_store=config.trajectory_node_store,
            effect_scope=config.effect_scope,
            environment_restore_failure_handler=(
                config.environment_restore_failure_handler
            ),
            versioned_resource_store=config.versioned_resource_store,
            atom_catalog=config.atom_catalog,
            atom_configs=_resolved_atom_config(
                resolved_spec, config.atom_config_overrides
            ),
            scenario_loader=config.scenario_loader,
            cwd=config.cwd,
            purpose=config.purpose,
            store=config.store if config.store is not None else resolve_trajectory_store_or_create(config.cwd or None),
            session_context=restored_context,
            session_id=config.session_id,
            root_session_id=config.root_session_id,
            parent_session_id=config.parent_session_id,
            bus=config.bus,
            initial_turns=config.initial_turns,
            services=services,
            resolved_spec=resolved_spec,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            tool_allowlist=config.tool_allowlist,
            cancel_signal=config.cancel_signal,
        )
    )

    for tool in config.extra_tools:
        session.register_tool(tool)
    return session


async def create_child_session(
    *,
    parent: Session,
    config: "AgentSessionConfig",
) -> Session:
    """Create a child session through the same SDK factory pipeline."""

    child_services = ServiceRegistry()
    child_services.inherit_from(parent.services)
    _register_default_catalog_services(child_services)
    if config.experiment is not None:
        experiment = freeze_json(config.experiment)
        if not isinstance(experiment, Mapping):
            raise TypeError("experiment config must be a JSON object")
        child_services.register("experiment", experiment, scope="tree")
    if config.loop_config is not None:
        child_services.register(
            LOOP_BUDGET_SERVICE, config.loop_config, scope="session"
        )
    if config.scenario_loader is not None:
        child_services.register(
            SCENARIO_LOADER_SERVICE,
            config.scenario_loader,
            scope="host",
        )
    if config.spec_resolver is not None:
        child_services.register(
            SESSION_SPEC_RESOLVER_SERVICE,
            config.spec_resolver,
            scope="host",
        )
    if config.provider_resolver is not None:
        child_services.register(
            PROVIDER_RESOLVER_SERVICE,
            config.provider_resolver,
            scope="host",
        )
    if config.environment_restore_failure_handler is not None:
        child_services.register(
            ENVIRONMENT_RESTORE_FAILURE_HANDLER_SERVICE,
            config.environment_restore_failure_handler,
            EnvironmentRestoreFailureHandler,
            scope="host",
        )
    resolved_spec = _resolve_session_spec(config)
    if resolved_spec is not None:
        child_services.register(
            RESOLVED_SESSION_SPEC_SERVICE,
            resolved_spec,
            ResolvedSessionSpec,
            scope="session",
        )
    provider_spec = (
        resolved_spec.provider if resolved_spec is not None else config.provider
    )
    if provider_spec is None and config.stream_fn is None and config.model is None:
        inherited_provider = parent.get_provider()
        if inherited_provider is not None:
            child_services.register(
                f"provider:{inherited_provider.name}",
                inherited_provider,
                scope="session",
            )

    scenario_loader = config.scenario_loader or _get_scenario_loader(child_services)
    inherit_parent_composition = (
        resolved_spec is None and config.scenario is None and config.extensions is None
    )
    if resolved_spec is not None:
        scenario = resolved_spec.scenario
    elif config.scenario is not None:
        scenario = config.scenario
    else:
        scenario = parent.ctx.scenario
    requested_extensions = (
        list(resolved_spec.extensions)
        if resolved_spec is not None
        else parent._composition_extensions(
            include_provider_atoms=(
                config.provider is None
                and config.stream_fn is None
                and config.model is None
            )
        )
        if inherit_parent_composition
        else config.extensions
    )
    extensions, scenario_dir, scenario_name = _resolve_extensions(
        scenario=scenario,
        extensions=requested_extensions,
        extra_extensions=() if resolved_spec is not None else config.extra_extensions,
        atom_configs=_resolved_atom_config(resolved_spec, config.atom_config_overrides),
        scenario_loader=scenario_loader,
    )
    if inherit_parent_composition:
        scenario_dir = parent.ctx.scenario_dir

    child_id = config.session_id or uuid.uuid4().hex[:16]
    child_store = config.store if config.store is not None else parent.store
    if child_store is not parent.store:
        child_services.unregister(TRAJECTORY_QUERY_STORE_SERVICE)
        _register_default_query_store(child_services, child_store)
    child_ctx = parent.ctx.child(
        session_id=child_id,
        purpose=config.purpose,
        cwd=config.cwd or None,
        scenario=scenario_name,
        scenario_dir=scenario_dir,
    )

    max_turns = config.loop_config.max_turns if config.loop_config else None
    max_tool_calls = config.loop_config.max_tool_calls if config.loop_config else None

    if config.parent_cancellation not in {"inherit", "independent"}:
        raise ValueError("parent_cancellation must be 'inherit' or 'independent'")
    parent_signal = (
        CompositeCancelSignal(
            parent._interrupt,
            parent._shutdown,
            parent._parent_cancel_signal,
        )
        if config.parent_cancellation == "inherit"
        else None
    )
    child_cancel_signal = (
        CompositeCancelSignal(parent_signal, config.cancel_signal)
        if parent_signal is not None and config.cancel_signal is not None
        else config.cancel_signal
        if config.cancel_signal is not None
        else parent_signal
    )

    child = Session(
        SessionRuntimeConfig(
            ctx=child_ctx,
            trajectory=Trajectory(turns=config.initial_turns),
            bus=config.bus,
            stream_fn=config.stream_fn or parent._stream_fn,
            model=config.model or parent._model,
            system=config.system if config.system is not None else parent.system,
            store=child_store,
            graph=parent.graph,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            tool_allowlist=config.tool_allowlist,
            cancel_signal=child_cancel_signal,
            environment_restore_failure_handler=(
                config.environment_restore_failure_handler
            ),
            provider_identity=(
                resolved_spec.provider_identity if resolved_spec is not None else None
            ),
            services=child_services,
            cwd=config.cwd or parent.ctx.cwd,
            purpose=config.purpose,
        )
    )
    if config.resource_writer is not None:
        child.register_resource_writer(config.resource_writer, replace=True)
    if config.resource_reader is not None:
        child.register_resource_reader(config.resource_reader, replace=True)
    if config.resource_store is not None:
        child.register_resource_store(config.resource_store, replace=True)
    if config.tool_executor is not None:
        child.register_tool_executor(config.tool_executor, replace=True)
    if config.tool_orchestrator is not None:
        child.register_tool_orchestrator(config.tool_orchestrator, replace=True)
    if config.permission_policy is not None:
        child.register_permission_policy(config.permission_policy, replace=True)
    if config.trajectory_node_store is not None:
        child.register_trajectory_node_store(config.trajectory_node_store, replace=True)
    if config.effect_scope is not None:
        child.register_effect_scope(config.effect_scope, replace=True)
    if config.versioned_resource_store is not None:
        child.register_versioned_resource_store(
            config.versioned_resource_store,
            replace=True,
        )
    if config.atom_catalog is not None:
        child.register_atom_catalog(config.atom_catalog, replace=True)

    plan_specs = list(extensions)
    if provider_spec is not None:
        plan_specs.append(normalize_extension_spec(provider_spec))
    plan = _extension_plan(
        plan_specs,
        available_capabilities=_service_capabilities(child_services),
    )
    for item in plan:
        await install_extension(
            child,
            item.spec,
            trigger="child_session_start",
        )

    for tool in config.extra_tools:
        child.register_tool(tool)

    created_at = time.time()
    active_set = await _record_active_set(child, plan, created_at=created_at)
    await _ensure_store_session(
        child_store,
        meta=SessionMeta(
            id=child.id,
            parent_id=parent.id,
            purpose=config.purpose,
            cwd=child_ctx.cwd,
            created_at=created_at,
            config=session_meta_config(
                child_ctx,
                resolved_spec=resolved_spec,
                active_set=active_set,
                provider_identity=child.provider_session_identity(),
            ),
        ),
        initial_turns=config.initial_turns,
    )

    return child


__all__ = [
    "SessionBuildConfig",
    "create_child_session",
    "create_from_config",
    "create_session",
]
