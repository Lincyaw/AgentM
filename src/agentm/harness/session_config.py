"""``AgentSessionConfig`` — extracted so extensions can build it.

Lives in its own module (rather than ``agentm.harness.session``) because
``harness.session`` is on the §11 forbidden-import list for extensions.
Extensions that legitimately need to construct a child session config (the
canonical case is the ``sub_agent`` builtin going through
``api.spawn_child_session``) import the dataclass from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentm.core.abi import AgentMessage, EventBus, LoopConfig
from agentm.harness.resource_loader import ResourceLoader
from agentm.harness.session_manager import SessionManager


@dataclass
class AgentSessionConfig:
    """Knobs handed to :func:`AgentSession.create`. Only ``cwd``, ``provider``
    are required; everything else has a sane default for embedded use."""

    cwd: str
    provider: tuple[str, dict[str, Any]]
    extensions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    scenario: str | None = None
    no_extensions: bool = False
    no_skills: bool = False
    no_prompt_templates: bool = False
    tool_allowlist: list[str] | None = None
    initial_messages: list[AgentMessage] = field(default_factory=list)
    session_manager: SessionManager | None = None
    resource_loader: ResourceLoader | None = None
    loop_config: LoopConfig | None = None
    bus: EventBus | None = None
    # --- Child-session lifecycle (used by sub-agent extensions) ----------
    parent_bus: EventBus | None = None
    """If set, ``child_session_start`` / ``child_session_end`` are emitted on
    this bus when the session is created and shut down. Used by the
    ``sub_agent`` extension to roll up nested sessions on the parent."""

    parent_session_id: str | None = None
    """Caller-supplied id of the parent session. Surfaces verbatim in the
    child-lifecycle events. ``None`` becomes ``"unknown"`` in the payload."""

    root_session_id: str | None = None
    """Stable root id shared across a session tree. ``None`` means this
    session is itself the root and should use its own session id."""

    task_id: str | None = None
    """Caller-defined task id for sub-agent dispatch provenance."""

    persona: str | None = None
    """Optional persona / role label for provenance-bearing extensions."""

    purpose: str = "root"
    """Caller-defined purpose label, e.g. ``"subagent:worker"``;
    surfaces verbatim in :class:`ChildSessionStartEvent`."""


__all__ = ["AgentSessionConfig"]
