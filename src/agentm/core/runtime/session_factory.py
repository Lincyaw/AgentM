"""Session factory -- create a v2 Session from a scenario manifest.

The factory reads the scenario's manifest.yaml, resolves extensions,
creates a v2 Session, then installs each atom by calling
module.install(session, config).

Atoms receive the Session directly (no adapter).
"""

from __future__ import annotations

import inspect
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentm.core.abi.session_api import AgentSessionConfig

import yaml
from loguru import logger

from agentm.core.abi.bus import EventBus
from agentm.core.abi.events import ExtensionInstallEvent
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.session_api import SessionContext
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.stream import Model, StreamFn
from agentm.core.abi.trajectory import Turn
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.runtime.session import Session
from agentm.core.runtime.extension import load_extension


# --- Scenario loading -------------------------------------------------------

_SCENARIO_SEARCH_PATHS: list[Path] = []


def _find_scenario_dir(scenario: str, scenario_dir: str | None = None) -> Path | None:
    """Locate the scenario directory by name."""
    if scenario_dir is not None:
        p = Path(scenario_dir)
        if p.is_dir():
            return p

    # Search in standard locations
    search = list(_SCENARIO_SEARCH_PATHS)

    # Source checkout contrib/scenarios/
    pkg_root = Path(__file__).parents[4]
    search.append(pkg_root / "contrib" / "scenarios")

    # Home contrib
    try:
        from agentm.core.lib.user_config import agentm_home_dir
        search.append(agentm_home_dir() / "contrib" / "scenarios")
    except Exception:  # noqa: S110
        pass

    for base in search:
        candidate = base / scenario
        if (candidate / "manifest.yaml").is_file():
            return candidate
    return None


def _load_scenario_extensions(
    scenario: str,
    scenario_dir: str | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], str | None]:
    """Load extensions list from scenario manifest.

    Returns (extensions, resolved_scenario_dir).
    """
    sdir = _find_scenario_dir(scenario, scenario_dir)
    if sdir is None:
        logger.warning("scenario {!r} not found; using empty extension list", scenario)
        return [], None

    manifest_path = sdir / "manifest.yaml"
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("failed to read scenario manifest {}: {}", manifest_path, exc)
        return [], str(sdir)

    if not isinstance(data, dict):
        return [], str(sdir)

    extensions: list[tuple[str, dict[str, Any]]] = []
    for entry in data.get("extensions", []):
        if isinstance(entry, str):
            extensions.append((entry, {}))
        elif isinstance(entry, dict):
            module = entry.get("module", "")
            config = entry.get("config", {})
            if not isinstance(config, dict):
                config = {}
            extensions.append((module, config))

    return extensions, str(sdir)


# --- Factory ----------------------------------------------------------------


async def create_session(
    *,
    scenario: str = "chatbot",
    scenario_dir: str | None = None,
    stream_fn: StreamFn | None = None,
    model: Model | None = None,
    system: str | None = None,
    cwd: str = "",
    purpose: str = "root",
    store: TrajectoryStore | None = None,
    session_id: str | None = None,
    root_session_id: str | None = None,
    parent_session_id: str | None = None,
    bus: EventBus | None = None,
    initial_turns: list[Turn] | None = None,
    extra_extensions: list[str] | None = None,
    extensions: list[tuple[str, dict[str, Any]]] | None = None,
    provider: tuple[str, dict[str, Any]] | None = None,
    atom_configs: dict[str, dict[str, Any]] | None = None,
    services: ServiceRegistry | None = None,
    max_turns: int | None = None,
    tool_allowlist: list[str] | None = None,
) -> Session:
    """Create a v2 Session from a scenario manifest.

    Loads extensions from the scenario, creates the Session, then installs
    each atom by passing the Session directly.
    """

    if extensions is not None:
        extension_specs = list(extensions)
        resolved_scenario_dir = scenario_dir
    else:
        extension_specs, resolved_scenario_dir = _load_scenario_extensions(
            scenario, scenario_dir
        )

    if extra_extensions:
        for ext in extra_extensions:
            extension_specs.append((ext, {}))

    # Apply per-atom config overrides
    if atom_configs:
        extension_specs = [
            (mod, {**cfg, **atom_configs.get(mod, {})})
            for mod, cfg in extension_specs
        ]

    resolved_session_id = session_id or uuid.uuid4().hex[:16]
    resolved_root_id = root_session_id or resolved_session_id
    ctx = SessionContext(
        session_id=resolved_session_id,
        root_session_id=resolved_root_id,
        parent_session_id=parent_session_id,
        cwd=cwd or "",
        purpose=purpose,
        scenario=scenario,
        scenario_dir=resolved_scenario_dir,
    )

    session = Session(
        ctx=ctx,
        trajectory=Trajectory(turns=initial_turns),
        bus=bus,
        stream_fn=stream_fn,
        model=model,
        system=system,
        store=store,
        max_turns=max_turns,
        services=services,
        cwd=cwd,
        purpose=purpose,
    )

    if tool_allowlist is not None:
        session.services.register("tool_allowlist", tool_allowlist)

    if provider is not None:
        await _install_extension(session, provider[0], provider[1])

    # Install atoms -- pass Session directly
    for module_path, config in extension_specs:
        if not module_path:
            continue
        await _install_extension(session, module_path, config)

    return session


async def create_from_config(config: "AgentSessionConfig") -> Session:
    """Create a root session from the public SDK config dataclass."""

    max_turns = config.loop_config.max_turns if config.loop_config else None
    services = ServiceRegistry()
    if config.experiment is not None:
        services.register("experiment", config.experiment)
    if config.lineage:
        services.register("lineage", config.lineage)
    if config.task_id is not None:
        services.register("task_id", config.task_id)
    if config.persona is not None:
        services.register("persona", config.persona)
    if config.trace_label is not None:
        services.register("trace_label", config.trace_label)
    if config.loop_config is not None:
        from agentm.core.abi import LOOP_BUDGET_SERVICE

        services.register(LOOP_BUDGET_SERVICE, config.loop_config)

    extra_extensions = [module for module, _ in config.extra_extensions]
    atom_configs = dict(config.atom_config_overrides)
    for module, cfg in config.extra_extensions:
        if cfg:
            atom_configs[module] = {**atom_configs.get(module, {}), **cfg}

    session = await create_session(
        scenario=config.scenario or "chatbot",
        extensions=config.extensions,
        extra_extensions=extra_extensions,
        provider=config.provider,
        atom_configs=atom_configs,
        cwd=config.cwd,
        purpose=config.purpose,
        store=config.store,
        session_id=config.session_id,
        root_session_id=config.root_session_id,
        parent_session_id=config.parent_session_id,
        bus=config.bus,
        initial_turns=config.initial_turns,
        services=services,
        max_turns=max_turns,
        tool_allowlist=config.tool_allowlist,
    )

    for tool in config.extra_tools:
        session.register_tool(tool)
    return session


async def _install_extension(
    session: Session,
    module_path: str,
    config: dict[str, Any],
    *,
    trigger: str = "session_start",
) -> None:
    started_ns = time.perf_counter_ns()
    name = module_path.rsplit(".", 1)[-1]
    await session.bus.emit(
        ExtensionInstallEvent.CHANNEL,
        ExtensionInstallEvent(
            name=name,
            module_path=module_path,
            phase="start",
            config=dict(config),
            trigger=trigger,
        ),
    )
    error: str | None = None
    try:
        result = load_extension(module_path, session, config)
        if inspect.isawaitable(result):
            await result
        session.installed_extensions.append(module_path)
        logger.debug("installed atom: {}", module_path)
    except Exception as exc:
        error = str(exc)
        logger.exception("failed to install atom: {}", module_path)
        raise
    finally:
        await session.bus.emit(
            ExtensionInstallEvent.CHANNEL,
            ExtensionInstallEvent(
                name=name,
                module_path=module_path,
                phase="error" if error else "end",
                config=dict(config),
                duration_ns=time.perf_counter_ns() - started_ns,
                trigger=trigger,
                error=error,
            ),
        )


async def create_child_session(
    *,
    parent: Session,
    config: "AgentSessionConfig",
) -> Session:
    """Create a child session from an ``AgentSessionConfig``.

    Resolves extensions either from ``config.extensions`` (explicit full
    list) or from the scenario + ``config.extra_extensions``.  Inherits
    the parent's store, graph, stream_fn, and model unless overridden.
    """
    scenario = config.scenario or parent.ctx.scenario or "chatbot"

    if config.extensions is not None:
        extensions = list(config.extensions)
    else:
        extensions, _ = _load_scenario_extensions(
            scenario, parent.ctx.scenario_dir
        )
        for mod, cfg in config.extra_extensions:
            extensions.append((mod, cfg))

    if config.atom_config_overrides:
        extensions = [
            (mod, {**cfg, **config.atom_config_overrides.get(mod, {})})
            for mod, cfg in extensions
        ]

    child_id = config.session_id or uuid.uuid4().hex[:16]
    child_ctx = parent.ctx.child(
        session_id=child_id,
        purpose=config.purpose,
        cwd=config.cwd or None,
        scenario=scenario,
    )

    child_services = ServiceRegistry()
    child_services.update_from(parent.services)

    if config.experiment is not None:
        child_services.register("experiment", config.experiment)
    if config.lineage:
        child_services.register("lineage", config.lineage)
    if config.task_id is not None:
        child_services.register("task_id", config.task_id)
    if config.persona is not None:
        child_services.register("persona", config.persona)
    if config.trace_label is not None:
        child_services.register("trace_label", config.trace_label)
    if config.loop_config is not None:
        from agentm.core.abi import LOOP_BUDGET_SERVICE
        child_services.register(LOOP_BUDGET_SERVICE, config.loop_config)

    max_turns = config.loop_config.max_turns if config.loop_config else None

    child = Session(
        ctx=child_ctx,
        stream_fn=parent._stream_fn,
        model=parent._model,
        store=parent.store,
        graph=parent.graph,
        max_turns=max_turns,
        services=child_services,
        cwd=config.cwd or parent.ctx.cwd,
        purpose=config.purpose,
    )

    if config.tool_allowlist is not None:
        child.services.register("tool_allowlist", config.tool_allowlist)

    # Provider wiring:
    # - provider=None: child inherits parent's stream_fn/model via constructor
    #   AND parent's provider:<name> service via update_from.  No extra atom.
    # - provider=(module, cfg): explicit provider extension — install after
    #   the main extension list so it overrides the inherited one.
    provider_ext: tuple[str, dict[str, Any]] | None = None
    if config.provider is not None:
        provider_ext = config.provider

    for module_path, ext_config in extensions:
        if not module_path:
            continue
        try:
            result = load_extension(module_path, child, ext_config)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("failed to install atom in child: {}", module_path)

    if provider_ext is not None:
        mod, cfg = provider_ext
        try:
            result = load_extension(mod, child, cfg)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("failed to install provider in child: {}", mod)

    for tool in config.extra_tools:
        child.tools.append(tool)

    return child


__all__ = [
    "create_child_session",
    "create_session",
]
