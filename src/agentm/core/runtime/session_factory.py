"""Session factory — create a v2 Session from a scenario manifest.

The factory reads the scenario's manifest.yaml, resolves extensions,
creates a v2 Session, then installs each atom by calling
module.install(adapter, config).

The adapter bridges v1 ExtensionAPI calls to the v2 Session so atoms
written against v1 work without modification.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

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
    pkg_root = Path(__file__).parents[3]
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


# --- v1 ExtensionAPI adapter -----------------------------------------------

class _ExtensionAPIAdapter:
    """Bridges v1 ExtensionAPI calls to a v2 Session.

    Atoms call api.on(), api.register_tool(), api.set_service(), etc.
    This adapter routes those to the Session's v2 equivalents.
    """

    def __init__(self, session: Session, *, owner: str = "") -> None:
        self._session = session
        self._owner = owner
        self._providers: dict[str, Any] = {}

    # --- Identity ---
    @property
    def session_id(self) -> str:
        return self._session.id

    @property
    def root_session_id(self) -> str:
        return self._session.ctx.root_session_id

    @property
    def parent_session_id(self) -> str | None:
        return self._session.ctx.parent_session_id

    @property
    def cwd(self) -> str:
        return self._session.ctx.cwd

    @property
    def scenario_dir(self) -> str | None:
        return self._session.ctx.scenario_dir

    @property
    def purpose(self) -> str:
        return self._session.ctx.purpose

    @property
    def scenario(self) -> str | None:
        return self._session.ctx.scenario

    @property
    def lineage(self) -> dict[str, Any] | None:
        return None

    @property
    def experiment(self) -> dict[str, Any] | None:
        return None

    # --- Bus ---
    def on(self, channel: str, handler: Any, *, priority: int = 500) -> Any:
        return self._session.bus.on(channel, handler, priority=priority, owner=self._owner)

    @property
    def events(self) -> Any:
        return self._session.bus

    # --- Registration ---
    def register_tool(self, tool: Any) -> None:
        self._session.register_tool(tool)

    def register_command(self, name: str, spec: Any) -> None:
        logger.debug("v1 compat: register_command({!r}) is a no-op in v2", name)

    def register_provider(self, name: str, config: Any) -> None:
        self._providers[name] = config

    def has_provider(self, name: str) -> bool:
        return name in self._providers

    def register_operations(self, *, bash: Any) -> None:
        self._session.services.register("_operations_bash", bash)

    def register_message_renderer(self, custom_type: str, renderer: Any) -> None:
        logger.debug("v1 compat: register_message_renderer({!r}) no-op in v2", custom_type)

    def register_tool_renderer(self, tool_name: str, renderer: Any) -> None:
        logger.debug("v1 compat: register_tool_renderer({!r}) no-op in v2", tool_name)

    def register_resource_writer(self, writer: Any) -> None:
        self._session.services.register("_resource_writer", writer)

    # --- Input ---
    def post_inbox(
        self,
        *,
        source: str,
        payload: Any,
        dedup_key: str | None = None,
        terminal: bool = False,
    ) -> None:
        from agentm.core.abi.trigger import BackgroundCompletion, SubagentResult
        text = str(payload) if payload is not None else ""
        if source == "subagent":
            self._session.push_trigger(SubagentResult(
                child_session_id=dedup_key or "",
                payload=text,
                terminal=terminal,
            ))
        else:
            self._session.push_trigger(BackgroundCompletion(
                task_id=dedup_key or source,
                payload=text,
                terminal=terminal,
            ))

    def track_background(self) -> Any:
        return self._session.track_background()

    def send_user_message(self, content: str | list[Any]) -> None:
        from agentm.core.abi.trigger import UserInput
        from agentm.core.abi.messages import TextContent
        text = content if isinstance(content, str) else str(content)
        self._session.push_trigger(UserInput(
            content=(TextContent(type="text", text=text),),
        ))

    async def wait_inbox_nonempty(self, sources: frozenset[str] | None = None) -> bool:
        return await self._session.triggers.wait_quiescent(timeout=None)

    # --- Services ---
    def set_service(self, name: str, obj: Any) -> None:
        self._session.services.register(name, obj)

    def get_service(self, name: str) -> Any:
        return self._session.services.get(name)

    # --- Query ---
    @property
    def tools(self) -> list[Any]:
        return self._session.tools

    @property
    def model(self) -> Any:
        return self._session.model

    @property
    def provider(self) -> Any:
        return None

    @property
    def session(self) -> Any:
        return self._session

    def get_operations(self) -> Any:
        bash = self._session.services.get("_operations_bash")
        if bash is None:
            raise RuntimeError(
                "no atom registered Operations; the active scenario manifest "
                "must list an atom that calls api.register_operations(...)"
            )
        from agentm.core.abi.operations import Operations
        return Operations(bash=bash)

    def get_project_layout(self) -> Any:
        return None

    def get_resource_writer(self) -> Any:
        return self._session.services.get("_resource_writer")

    def get_session_telemetry(self) -> Any:
        return None

    # --- Gateway (stubs) ---
    async def spawn_child_session(self, config: Any) -> Any:
        raise RuntimeError("spawn_child_session not available via v1 compat adapter")

    def reload_atom(self, name: str, new_source: str, *, agent_initiated: bool = True, rationale: str | None = None) -> Any:
        raise RuntimeError("reload_atom not available via v1 compat adapter")

    def install_atom(self, *, name: str, source: str, target_path: str | None = None, config: dict[str, Any] | None = None, rationale: str | None = None, agent_initiated: bool = True) -> Any:
        raise RuntimeError("install_atom not available via v1 compat adapter")

    def unload_atom(self, name: str, *, agent_initiated: bool = True) -> Any:
        raise RuntimeError("unload_atom not available via v1 compat adapter")

    def list_atoms(self) -> list[Any]:
        return []

    def freeze_current(self, name: str) -> str:
        raise RuntimeError("freeze_current not available via v1 compat adapter")

    def is_constitution_path(self, path: str) -> bool:
        return False

    @property
    def catalog(self) -> Any:
        return None

    def add_observer(self, callback: Any) -> Any:
        logger.debug("v1 compat: add_observer is a no-op in v2")
        return lambda: None


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
    each atom through the v1 compat adapter so existing atoms work unchanged.
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

    # Install atoms via compat adapter
    adapter = _ExtensionAPIAdapter(session, owner="factory")
    for module_path, config in extensions:
        if not module_path:
            continue
        try:
            result = load_extension(module_path, adapter, config)
            if inspect.isawaitable(result):
                await result
            logger.debug("installed atom: {}", module_path)
        except Exception:
            logger.exception("failed to install atom: {}", module_path)

    return session


__all__ = [
    "create_session",
]
