"""Session factory -- create a v2 Session from a scenario manifest.

The factory reads the scenario's manifest.yaml, resolves extensions,
creates a v2 Session, then installs each atom by calling
module.install(session, config).

Atoms receive the Session directly (no adapter).
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


__all__ = [
    "create_session",
]
