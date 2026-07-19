"""Session factory -- create a v2 Session from a scenario manifest.

The factory reads the scenario's manifest.yaml, resolves extensions,
creates a v2 Session, then installs each atom by calling
module.install(session, config).

Atoms receive the Session directly (no adapter).
"""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentm.core.abi.session_api import AgentSessionConfig

import yaml
from loguru import logger

from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.session_api import SessionContext
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.stream import Model, StreamFn
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
    extra_extensions: list[str] | None = None,
    atom_configs: dict[str, dict[str, Any]] | None = None,
    services: ServiceRegistry | None = None,
    max_turns: int | None = None,
) -> Session:
    """Create a v2 Session from a scenario manifest.

    Loads extensions from the scenario, creates the Session, then installs
    each atom by passing the Session directly.
    """

    extensions, resolved_scenario_dir = _load_scenario_extensions(
        scenario, scenario_dir
    )

    if extra_extensions:
        for ext in extra_extensions:
            extensions.append((ext, {}))

    # Apply per-atom config overrides
    if atom_configs:
        extensions = [
            (mod, {**cfg, **atom_configs.get(mod, {})})
            for mod, cfg in extensions
        ]

    ctx = SessionContext(
        session_id="",
        root_session_id="",
        cwd=cwd or "",
        purpose=purpose,
        scenario=scenario,
        scenario_dir=resolved_scenario_dir,
    )

    session = Session(
        ctx=ctx,
        stream_fn=stream_fn,
        model=model,
        system=system,
        store=store,
        max_turns=max_turns,
        services=services,
        cwd=cwd,
        purpose=purpose,
    )

    # Install atoms -- pass Session directly
    for module_path, config in extensions:
        if not module_path:
            continue
        try:
            result = load_extension(module_path, session, config)
            if inspect.isawaitable(result):
                await result
            logger.debug("installed atom: {}", module_path)
        except Exception:
            logger.exception("failed to install atom: {}", module_path)

    return session


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

    if config.extensions:
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
