# code-health: ignore-file[AM025] -- runtime composes plugin, service, and trajectory boundary values
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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from agentm.core.abi.bus import EventBus
from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.cancel import CompositeCancelSignal
from agentm.core.abi.codec import CodecRegistry
from agentm.core.abi.context import ContextPolicy
from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    AtomActivation,
    AtomCatalog,
    CatalogActiveSetInput,
    ResourceVersion,
    VersionedResourceStore,
)
from agentm.core.abi.errors import ExtensionLoadError
from agentm.core.abi.lifecycle import EffectScope, EnvironmentRestoreFailureHandler
from agentm.core.abi.messages import JsonValue, freeze_json
from agentm.core.abi.operations import EnvironmentOperations
from agentm.core.abi.manifest import (
    AtomInstallPriority,
    ExtensionManifest,
    parse_capability_ref,
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
    ACTIVE_SET_FINGERPRINT_ROLE,
    ATOM_CATALOG_ROLE,
    ATOM_CATALOG_SERVICE,
    EFFECT_SCOPE_ROLE,
    ENVIRONMENT_RESTORE_FAILURE_HANDLER,
    EXPERIMENT_SERVICE,
    LOOP_BUDGET_SERVICE,
    PROVIDER_RESOLVER_SERVICE,
    RESOLVED_SESSION_SPEC_SERVICE,
    RESOURCE_WRITER,
    SCENARIO_LOADER_SERVICE,
    TRAJECTORY_QUERY_STORE,
    TRAJECTORY_QUERY_STORE_SERVICE,
    TRAJECTORY_STORE_SERVICE,
    VERSIONED_RESOURCE_STORE_ROLE,
    VERSIONED_RESOURCE_STORE_SERVICE,
    bind_atom_catalog,
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
from agentm.core.abi.store import SessionMeta, TrajectoryStore
from agentm.core.abi.stream import Model, StreamFn, ThinkingLevel
from agentm.core.abi.tool import Tool
from agentm.core.abi.trajectory import (
    DEFAULT_TRAJECTORY_BRANCH_ID,
    DEFAULT_TRAJECTORY_HEAD_ID,
    TrajectoryHead,
    Turn,
    TurnRef,
)
from agentm.core.abi.tree import SessionGraphProtocol
from agentm.core.abi.trigger import TriggerRenderer
from agentm.core.runtime.catalog import (
    InMemoryAtomCatalog,
    InMemoryVersionedResourceStore,
    build_atom_identity_payload,
    normalize_atom_config,
)
from agentm.core.runtime.extension import (
    install_extension,
    load_extension_module,
)
from agentm.core.runtime.session import Session
from agentm.core.runtime.session_core import SessionRuntimeConfig
from agentm.core.runtime.session_meta import session_meta_config
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.lib.async_cancel import await_known_outcome
from agentm.core.lib.trajectory_query import TrajectoryStoreQueryAdapter
from agentm.core.lib.trajectory_nodes import turns_to_nodes

if TYPE_CHECKING:
    from agentm.core.abi.session_api import AgentSessionConfig


@dataclass(frozen=True, slots=True)
class _ExtensionPlanItem:
    spec: ExtensionSpec
    module_path: str
    config: dict[str, JsonValue]
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
    atom_configs: dict[str, dict[str, JsonValue]] | None,
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


def _verify_registered_capabilities(
    session: Session,
    item: _ExtensionPlanItem,
    *,
    plan_required_keys: frozenset[str],
) -> None:
    """Check manifest ``registers`` declarations against reality after install.

    Dependency solving trusts ``registers``, so a declared-but-missing
    capability that some plan member required is a composition lie and fails
    the install. A missing declaration nobody required only logs a warning —
    provisions may legitimately depend on config (e.g. an enabled-tools
    subset), and runtime consumers still fail loudly at their own lookup.
    Kinds without an addressable registry (``event``, ``context_policy``)
    are skipped; presence is what is verified, so a pre-existing binding the
    atom deliberately preserved (host override) passes.
    """

    missing: list[str] = []
    for declared in item.registers:
        ref = parse_capability_ref(declared)
        if ref.kind in {"service", "operations"}:
            name = ref.name if ref.kind == "service" else f"operations:{ref.name}"
            present = session.services.has(name)
        elif ref.kind == "tool":
            present = any(tool.name == ref.name for tool in session.tools)
        elif ref.kind == "provider":
            present = session.has_provider(ref.name)
        elif ref.kind == "trigger_renderer":
            present = ref.name in session.trigger_renderers
        else:
            continue
        if not present:
            missing.append(declared)
    required_missing = [
        declared
        for declared in missing
        if requirement_key(declared) in plan_required_keys
    ]
    if required_missing:
        raise ExtensionLoadError(
            item.module_path,
            RuntimeError(
                "manifest declares capabilities that other atoms require but "
                f"install() did not provide: {', '.join(required_missing)}"
            ),
        )
    for declared in missing:
        logger.warning(
            "atom {} declares {} in registers but did not provide it",
            item.name,
            declared,
        )


def _get_scenario_loader(services: ServiceRegistry | None) -> ScenarioLoader | None:
    if services is None:
        return None
    candidate = services.get(SCENARIO_LOADER_SERVICE)
    if isinstance(candidate, ScenarioLoader):
        return candidate
    return None


async def _ensure_store_session(
    store: TrajectoryStore | None,
    *,
    meta: SessionMeta,
    initial_turns: Sequence[Turn],
    initial_head: TrajectoryHead | None,
    root_session_id: str,
    parent_session_id: str | None,
    trigger_renderers: dict[str, TriggerRenderer],
) -> None:
    if store is None:
        return
    exists = await asyncio.to_thread(store.session_exists, meta.id)
    if not exists:
        nodes = (
            []
            if initial_head is not None
            else turns_to_nodes(
                initial_turns,
                session_id=meta.id,
                root_session_id=root_session_id,
                parent_session_id=parent_session_id,
                renderers=trigger_renderers,
            )
        )
        last = nodes[-1] if nodes else None
        head = initial_head or TrajectoryHead(
            session_id=meta.id,
            head_id=DEFAULT_TRAJECTORY_HEAD_ID,
            branch_id=DEFAULT_TRAJECTORY_BRANCH_ID,
            node_id=last.id if last is not None else None,
            seq=last.seq if last is not None else None,
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
            status="active",
            updated_at=time.time(),
        )
        await await_known_outcome(
            asyncio.to_thread(
                store.create_session,
                meta,
                turns=initial_turns,
                nodes=nodes,
                head=head,
            )
        )


def _register_default_catalog_services(services: ServiceRegistry) -> None:
    if not services.has(VERSIONED_RESOURCE_STORE_SERVICE):
        services.bind(
            VERSIONED_RESOURCE_STORE_ROLE,
            InMemoryVersionedResourceStore(),
        )
    if not services.has(ATOM_CATALOG_SERVICE):
        bind_atom_catalog(services, InMemoryAtomCatalog())


def _register_default_query_store(
    services: ServiceRegistry,
    store: TrajectoryStore | None,
) -> None:
    if store is None or services.has(TRAJECTORY_QUERY_STORE_SERVICE):
        return
    services.bind(
        TRAJECTORY_QUERY_STORE,
        TrajectoryStoreQueryAdapter(store),
    )


def _resolve_session_spec(config: "AgentSessionConfig") -> ResolvedSessionSpec | None:
    resolver = config.spec_resolver
    if resolver is None:
        return None
    return resolver.resolve(config)


def _compose_config_services(
    services: ServiceRegistry,
    config: "AgentSessionConfig",
) -> ResolvedSessionSpec | None:
    """Register the AgentSessionConfig-driven services shared by root and child.

    Single source for the experiment / loop-budget / scenario-loader /
    provider-resolver / restore-handler / resolved-spec composition; the root
    and child pipelines must not diverge on these.
    """

    if config.experiment is not None:
        experiment = freeze_json(config.experiment)
        if not isinstance(experiment, Mapping):
            raise TypeError("experiment config must be a JSON object")
        services.register(EXPERIMENT_SERVICE, experiment, scope="tree")
    if config.loop_config is not None:
        services.register(LOOP_BUDGET_SERVICE, config.loop_config, scope="session")
    if config.scenario_loader is not None:
        services.register(
            SCENARIO_LOADER_SERVICE,
            config.scenario_loader,
            scope="tree",
        )
    if config.provider_resolver is not None:
        services.register(
            PROVIDER_RESOLVER_SERVICE,
            config.provider_resolver,
            scope="tree",
        )
    if config.environment_restore_failure_handler is not None:
        services.bind(
            ENVIRONMENT_RESTORE_FAILURE_HANDLER,
            config.environment_restore_failure_handler,
            replace=True,
        )
    resolved_spec = _resolve_session_spec(config)
    if resolved_spec is not None:
        services.register(
            RESOLVED_SESSION_SPEC_SERVICE,
            resolved_spec,
            ResolvedSessionSpec,
            scope="session",
        )
    return resolved_spec


def _bind_boundary_overrides(
    session: Session,
    *,
    resource_writer: ResourceWriter | None,
    effect_scope: EffectScope | None,
    environment_operations: EnvironmentOperations | None,
    atom_catalog: AtomCatalog | None,
) -> None:
    """Bind post-construction boundary overrides shared by root and child."""

    if resource_writer is not None:
        session.services.bind(RESOURCE_WRITER, resource_writer, replace=True)
    if effect_scope is not None:
        session.services.bind(EFFECT_SCOPE_ROLE, effect_scope, replace=True)
    if environment_operations is not None:
        session.register_operations(
            environment=environment_operations,
            bash=environment_operations.bash,
            replace=True,
            service_scope="tree",
        )
    if atom_catalog is not None:
        bind_atom_catalog(session.services, atom_catalog, replace=True)


def _resolved_atom_config(
    spec: ResolvedSessionSpec | None,
    overrides: dict[str, dict[str, JsonValue]],
) -> dict[str, dict[str, JsonValue]]:
    source = overrides if spec is None else spec.atom_config
    return {module: dict(config) for module, config in source.items()}


def _config_fingerprint(config: object) -> str | None:
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
    catalog = session.services.get_role(ATOM_CATALOG_ROLE)
    if catalog is None:
        return None
    version_store = session.services.get_role(VERSIONED_RESOURCE_STORE_ROLE)
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
    session.services.bind(ACTIVE_SET_FINGERPRINT_ROLE, fingerprint, replace=True)
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
    initial_head: TrajectoryHead | None = None
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
    effect_scope: EffectScope | None = None
    environment_operations: EnvironmentOperations | None = None
    environment_restore_failure_handler: EnvironmentRestoreFailureHandler | None = None
    versioned_resource_store: VersionedResourceStore | None = None
    atom_catalog: AtomCatalog | None = None
    atom_configs: dict[str, dict[str, JsonValue]] | None = None
    scenario_loader: ScenarioLoader | None = None
    services: ServiceRegistry | None = None
    resolved_spec: ResolvedSessionSpec | None = None
    max_turns: int | None = None
    max_tool_calls: int | None = None
    tool_allowlist: list[str] | None = None
    thinking: ThinkingLevel = "off"
    cancel_signal: CancelSignal | None = None


async def _cleanup_failed_session(
    session: Session,
    creation_error: BaseException,
) -> None:
    try:
        await session.shutdown()
    except BaseException as cleanup_error:
        raise BaseExceptionGroup(
            "session creation and cleanup failed",
            (creation_error, cleanup_error),
        ) from creation_error


async def create_session(
    config: SessionBuildConfig,
    *,
    session_type: type[Session] = Session,
) -> Session:
    """Create a root SDK session."""

    resolved_services = (
        ServiceRegistry() if config.services is None else config.services
    )
    effective_loader = (
        _get_scenario_loader(resolved_services)
        if config.scenario_loader is None
        else config.scenario_loader
    )
    extension_specs, scenario_dir, scenario_name = _resolve_extensions(
        scenario=config.scenario,
        extensions=config.extensions,
        extra_extensions=config.extra_extensions,
        atom_configs=config.atom_configs,
        scenario_loader=effective_loader,
    )
    if effective_loader is not None:
        resolved_services.register(
            SCENARIO_LOADER_SERVICE,
            effective_loader,
            scope="tree",
        )
    if config.provider_resolver is not None:
        resolved_services.register(
            PROVIDER_RESOLVER_SERVICE,
            config.provider_resolver,
            scope="tree",
        )
    _register_default_catalog_services(resolved_services)
    _register_default_query_store(resolved_services, config.store)

    if config.session_context is None:
        resolved_session_id = config.session_id or uuid.uuid4().hex[:16]
        resolved_root_id = config.root_session_id or resolved_session_id
        ctx = SessionContext(
            session_id=resolved_session_id,
            root_session_id=resolved_root_id,
            parent_session_id=config.parent_session_id,
            cwd=config.cwd or "",
            purpose=config.purpose,
            scenario=scenario_name,
            scenario_dir=scenario_dir,
        )
    else:
        ctx = config.session_context

    session = session_type(
        SessionRuntimeConfig(
            ctx=ctx,
            trajectory=Trajectory(turns=config.initial_turns),
            bus=config.bus,
            stream_fn=config.stream_fn,
            model=config.model,
            system=config.system,
            store=config.store,
            graph=config.graph,
            tools=list(config.tools or ()),
            context_policies=list(config.context_policies or ()),
            trigger_renderers=dict(config.trigger_renderers or {}),
            codec=config.codec,
            max_turns=config.max_turns,
            max_tool_calls=config.max_tool_calls,
            tool_allowlist=config.tool_allowlist,
            thinking=config.thinking,
            cancel_signal=config.cancel_signal,
            tool_executor=config.tool_executor,
            tool_orchestrator=config.tool_orchestrator,
            permission_policy=config.permission_policy,
            resource_reader=config.resource_reader,
            resource_store=config.resource_store,
            versioned_resource_store=config.versioned_resource_store,
            environment_restore_failure_handler=(
                config.environment_restore_failure_handler
            ),
            provider_identity=config.provider_identity,
            services=resolved_services,
            cwd=config.cwd,
            purpose=config.purpose,
        )
    )
    try:
        _bind_boundary_overrides(
            session,
            resource_writer=config.resource_writer,
            effect_scope=config.effect_scope,
            environment_operations=config.environment_operations,
            atom_catalog=config.atom_catalog,
        )

        plan_specs = list(extension_specs)
        if config.provider is not None:
            plan_specs.append(normalize_extension_spec(config.provider))
        plan = _extension_plan(
            plan_specs,
            available_capabilities=_service_capabilities(resolved_services),
        )
        plan_required_keys = frozenset(
            requirement_key(requirement)
            for planned in plan
            for requirement in planned.requires
        )
        for item in plan:
            await install_extension(session, item.spec)
            _verify_registered_capabilities(
                session,
                item,
                plan_required_keys=plan_required_keys,
            )
        created_at = time.time()
        active_set = await _record_active_set(session, plan, created_at=created_at)
        await _ensure_store_session(
            config.store,
            meta=SessionMeta(
                id=session.id,
                parent_id=ctx.parent_session_id,
                fork_point=config.fork_point,
                purpose=ctx.purpose,
                cwd=ctx.cwd,
                created_at=created_at,
                config=session_meta_config(
                    ctx,
                    resolved_spec=config.resolved_spec,
                    active_set=active_set,
                    provider_identity=session.provider_session_identity(),
                ),
            ),
            initial_turns=config.initial_turns or (),
            initial_head=config.initial_head,
            root_session_id=ctx.root_session_id,
            parent_session_id=ctx.parent_session_id,
            trigger_renderers=session.trigger_renderers,
        )
    except BaseException as creation_error:
        await _cleanup_failed_session(session, creation_error)
        raise

    return session


async def create_from_config(
    config: "AgentSessionConfig",
    *,
    restored_context: SessionContext | None = None,
    restored_provider_identity: ProviderSessionIdentity | None = None,
    session_type: type[Session] = Session,
    host_services: ServiceRegistry | None = None,
) -> Session:
    """Create a root session from the public SDK config dataclass."""

    max_turns = config.loop_config.max_turns if config.loop_config else None
    max_tool_calls = config.loop_config.max_tool_calls if config.loop_config else None
    services = ServiceRegistry()
    if host_services is not None:
        services.update_from(host_services)
    resolved_spec = _compose_config_services(services, config)
    trajectory_store = config.trajectory_store
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
            effect_scope=config.effect_scope,
            environment_operations=config.environment_operations,
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
            store=trajectory_store,
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
        ),
        session_type=session_type,
    )

    try:
        for tool in config.extra_tools:
            session.register_tool(tool)
    except BaseException as creation_error:
        await _cleanup_failed_session(session, creation_error)
        raise
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
    resolved_spec = _compose_config_services(child_services, config)
    provider_spec = (
        resolved_spec.provider if resolved_spec is not None else config.provider
    )
    inherit_provider = (
        provider_spec is None and config.stream_fn is None and config.model is None
    )
    if inherit_provider:
        inherited_provider = parent.get_provider()
        if inherited_provider is not None:
            child_services.register(
                f"provider:{inherited_provider.name}",
                inherited_provider,
                scope="session",
            )

    snapshot = parent.composition_snapshot(
        include_provider_atoms=inherit_provider,
    )
    scenario_loader = (
        _get_scenario_loader(child_services)
        if config.scenario_loader is None
        else config.scenario_loader
    )
    inherit_parent_composition = (
        resolved_spec is None and config.scenario is None and config.extensions is None
    )
    if resolved_spec is not None:
        scenario = resolved_spec.scenario
    elif config.scenario is not None:
        scenario = config.scenario
    else:
        scenario = parent.ctx.scenario
    requested_extensions: Sequence[ExtensionInput] | None
    if resolved_spec is not None:
        requested_extensions = list(resolved_spec.extensions)
    elif inherit_parent_composition:
        requested_extensions = list(snapshot.extensions)
    else:
        requested_extensions = config.extensions
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
    child_store = (
        parent.store if config.trajectory_store is None else config.trajectory_store
    )
    if child_store is not parent.store:
        child_services.unregister(TRAJECTORY_QUERY_STORE_SERVICE)
        child_services.unregister(TRAJECTORY_STORE_SERVICE)
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
        snapshot.lineage_cancel if config.parent_cancellation == "inherit" else None
    )
    child_cancel_signal = (
        CompositeCancelSignal(parent_signal, config.cancel_signal)
        if parent_signal is not None and config.cancel_signal is not None
        else config.cancel_signal
        if config.cancel_signal is not None
        else parent_signal
    )

    child = type(parent)(
        SessionRuntimeConfig(
            ctx=child_ctx,
            trajectory=Trajectory(turns=config.initial_turns),
            bus=config.bus,
            stream_fn=(
                snapshot.stream_fn if config.stream_fn is None else config.stream_fn
            ),
            model=snapshot.model if config.model is None else config.model,
            system=config.system if config.system is not None else snapshot.system,
            store=child_store,
            graph=parent.graph,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            tool_allowlist=config.tool_allowlist,
            cancel_signal=child_cancel_signal,
            tool_executor=config.tool_executor,
            tool_orchestrator=config.tool_orchestrator,
            permission_policy=config.permission_policy,
            resource_reader=config.resource_reader,
            resource_store=config.resource_store,
            versioned_resource_store=config.versioned_resource_store,
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
    try:
        _bind_boundary_overrides(
            child,
            resource_writer=config.resource_writer,
            effect_scope=config.effect_scope,
            environment_operations=config.environment_operations,
            atom_catalog=config.atom_catalog,
        )

        plan_specs = list(extensions)
        if provider_spec is not None:
            plan_specs.append(normalize_extension_spec(provider_spec))
        plan = _extension_plan(
            plan_specs,
            available_capabilities=_service_capabilities(child_services),
        )
        plan_required_keys = frozenset(
            requirement_key(requirement)
            for planned in plan
            for requirement in planned.requires
        )
        for item in plan:
            await install_extension(
                child,
                item.spec,
                trigger="child_session_start",
            )
            _verify_registered_capabilities(
                child,
                item,
                plan_required_keys=plan_required_keys,
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
            initial_head=None,
            root_session_id=child_ctx.root_session_id,
            parent_session_id=child_ctx.parent_session_id,
            trigger_renderers=child.trigger_renderers,
        )
    except BaseException as creation_error:
        await _cleanup_failed_session(child, creation_error)
        raise

    return child


__all__ = [
    "SessionBuildConfig",
    "create_child_session",
    "create_from_config",
    "create_session",
]
