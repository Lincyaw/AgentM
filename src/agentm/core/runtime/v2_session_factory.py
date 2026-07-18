"""v2 Session factory — minimal session construction for v2 atoms.

Creates a v2 Session, loads extension modules, and calls
``install(session, config)`` for each atom.  Under 200 lines by design.

This factory replaces the v1 ``create_agent_session`` path for atoms
that have been migrated to the v2 Session API.
"""

from __future__ import annotations

import importlib
import inspect
import uuid
from typing import Any

from loguru import logger

from agentm.core.abi.stream import Model, StreamFn
from agentm.core.v2.abi.bus import EventBus
from agentm.core.v2.abi.events import DiagnosticEvent
from agentm.core.v2.abi.services import ServiceRegistry
from agentm.core.v2.abi.session_api import SessionContext
from agentm.core.v2.runtime.session import Session


async def create_session(
    *,
    stream_fn: StreamFn,
    model: Model,
    extensions: list[tuple[str, dict[str, Any]]] | None = None,
    system: str | None = None,
    cwd: str = "",
    purpose: str = "root",
    scenario: str | None = None,
    session_id: str | None = None,
    root_session_id: str | None = None,
    parent_session_id: str | None = None,
    max_turns: int | None = None,
    initial_services: dict[str, Any] | None = None,
) -> Session:
    """Create a v2 Session and install extensions.

    Each extension module must export ``install(session, config)``
    (sync or async).  The session is passed directly — no adapter.

    Parameters
    ----------
    stream_fn : StreamFn
        Provider stream function for LLM calls.
    model : Model
        Model identity and limits.
    extensions : list of (module_path, config) tuples
        Extensions to load in order.
    system : str, optional
        System prompt.
    cwd : str
        Working directory for the session.
    purpose : str
        Session purpose label.
    scenario : str, optional
        Scenario name.
    session_id : str, optional
        Override session ID.
    root_session_id : str, optional
        Root trace ID for the session graph.
    parent_session_id : str, optional
        Parent session ID for child sessions.
    max_turns : int, optional
        Maximum turns before auto-stop.
    initial_services : dict, optional
        Pre-registered services.

    Returns
    -------
    Session
        A configured v2 Session, ready to start.
    """
    sid = session_id or uuid.uuid4().hex[:16]
    rid = root_session_id or sid

    ctx = SessionContext(
        session_id=sid,
        root_session_id=rid,
        parent_session_id=parent_session_id,
        cwd=cwd,
        purpose=purpose,
        scenario=scenario,
    )

    services = ServiceRegistry()
    if initial_services:
        for name, obj in initial_services.items():
            services.register(name, obj)

    bus = EventBus()
    session = Session(
        ctx=ctx,
        bus=bus,
        stream_fn=stream_fn,
        model=model,
        system=system,
        max_turns=max_turns,
        services=services,
        cwd=cwd,
        purpose=purpose,
    )

    to_load = extensions or []
    for module_path, ext_config in to_load:
        try:
            await _install_extension(session, module_path, ext_config)
        except Exception as exc:  # noqa: BLE001
            logger.error("extension install failed: {}: {}", module_path, exc)
            await bus.emit(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(
                    level="error",
                    source="v2_session_factory",
                    message=f"{module_path}: {exc}",
                ),
            )

    return session


async def _install_extension(
    session: Session,
    module_path: str,
    config: dict[str, Any],
) -> None:
    """Import a module and call its install(session, config)."""
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to import {}: {}", module_path, exc)
        raise

    install_fn = getattr(module, "install", None)
    if install_fn is None or not callable(install_fn):
        logger.warning("module {} has no callable 'install'", module_path)
        return

    # Auto-validate config via MANIFEST.config_schema (Pydantic model).
    resolved_config: Any = config
    manifest = getattr(module, "MANIFEST", None)
    if manifest is not None:
        schema_cls = getattr(manifest, "config_schema", None)
        if schema_cls is not None:
            try:
                from pydantic import BaseModel as _PydanticBase

                if isinstance(schema_cls, type) and issubclass(
                    schema_cls, _PydanticBase
                ):
                    resolved_config = schema_cls.model_validate(config)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "config validation for {}: {}",
                    module_path,
                    exc,
                )

    result = install_fn(session, resolved_config)
    if inspect.isawaitable(result):
        await result


__all__ = ["create_session"]
