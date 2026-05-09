"""``AgentSessionConfig`` — extracted so extensions can build it.

Lives in its own module (rather than ``agentm.harness.session``) because
``harness.session`` is on the §11 forbidden-import list for extensions.
Extensions that legitimately need to construct a child session config (the
canonical case is the ``sub_agent`` builtin going through
``api.spawn_child_session``) import the dataclass from here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from agentm.core.abi import AgentMessage, EventBus, LoopConfig, ProviderResolver
from agentm.harness.resource_loader import ResourceLoader
from agentm.harness.session_manager import SessionManager


def default_child_provider_factory(parent_provider: Any) -> tuple[str, dict[str, Any]]:
    """Return the inherit-provider atom spec for a child session."""
    from agentm.extensions.discover import discover_builtin
    from agentm.extensions.builtin.inherit_provider import PARENT_PROVIDER_CONFIG_KEY

    entry = discover_builtin()["inherit_provider"]
    return (entry.module_path, {PARENT_PROVIDER_CONFIG_KEY: parent_provider})


@dataclass
class AgentSessionConfig:
    """Knobs handed to :func:`AgentSession.create`. Only ``cwd`` is required.

    ``provider`` may be ``None`` when this config is handed to
    :meth:`ExtensionAPI.spawn_child_session`; the spawn factory then
    automatically wires the ``inherit_provider`` builtin so the child re-uses
    the parent's active :class:`ProviderConfig`. For root sessions, ``provider``
    must resolve to a real provider tuple either directly here or via the
    ``extensions`` list.
    """

    cwd: str
    provider: tuple[str, dict[str, Any]] | None = None
    extensions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    scenario: str | None = None
    extra_extensions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    """Atoms appended after the primary load step (auto-discover or scenario).

    Used by ``--extension`` on the CLI to mount third-party atoms on top of
    a base scenario or the default builtin discovery without forcing the
    user to copy the entire base list into a custom manifest. Ignored when
    ``no_extensions`` is set or when ``extensions`` is supplied explicitly
    (in which case the caller already controls the full list)."""
    no_extensions: bool = False
    no_skills: bool = False
    no_prompt_templates: bool = False
    tool_allowlist: list[str] | None = None
    initial_messages: list[AgentMessage] = field(default_factory=list)
    session_manager: SessionManager | None = None
    resource_loader: ResourceLoader | None = None
    loop_config: LoopConfig | None = None
    bus: EventBus | None = None

    provider_resolver: ProviderResolver[Any] | None = None
    """Selects the active provider after provider registrations complete."""

    child_provider_factory: Callable[[Any], tuple[str, dict[str, Any]]] | None = None
    """Builds the provider extension spec for child sessions that inherit a parent provider."""
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

    def with_bus(self, bus: EventBus) -> "AgentSessionConfig":
        return replace(self, bus=bus)


__all__ = ["AgentSessionConfig", "default_child_provider_factory"]
