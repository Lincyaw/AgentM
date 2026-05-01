"""Harness-level event payload types.

Implements §10b.1 of ``.claude/designs/extension-as-scenario.md``. These
events are emitted on an ``AgentSession``'s ``EventBus`` (or a parent's bus
for child-session events) and consumed by Phase 2 extensions:

- ``before_compact`` / ``after_compact``: compaction extensions
  (``micro_compact``, ``agent_memory``)
- ``child_session_start`` / ``child_session_end``: ``sub_agent`` lifecycle
  visibility for trajectory and cost rollups
- ``cost_budget_exceeded``: ``cost_budget`` extension signals the kernel
  loop / orchestrator to terminate with ``stop_reason='budget'``
- ``plan_submitted``: ``tool_submit_plan`` signals plan-mode completion
- ``session_ready``: emitted by ``AgentSession.create`` after every
  extension installed and the active provider is picked, but before the
  first ``prompt`` runs — the only timing point where every extension can
  observe the *final* tool list, command set, and model

Phase 2.0 defines the dataclasses; emission lives in the extension that
owns each event (cost_budget, plan_submitted) or in ``AgentSession``
itself (child_session_*, session_ready, before/after_compact when the
default compaction extension is loaded).

Layer purity: imports only stdlib + ``agentm.core.kernel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from agentm.core.kernel import AgentMessage, Model


@dataclass(slots=True)
class BeforeAgentStartEvent:
    """Fires at the top of ``AgentSession.prompt`` before the kernel loop runs.

    Mutability: this event is intentionally **not frozen**. Handlers may mutate
    ``system`` in place; alternatively they may return a ``dict[str, str]`` of
    shape ``{"system": "..."}`` and the harness will use the last non-None
    replacement to overwrite the system prompt. ``messages`` is the live list
    that will be passed to the loop — handlers should generally not rewrite it
    here (use ``context`` / ``before_send_to_llm`` for that).
    """

    messages: list[AgentMessage]
    system: str | None


@dataclass(frozen=True, slots=True)
class SessionShutdownEvent:
    """Fires when ``AgentSession.shutdown`` is called.

    Carries the session's cwd so cleanup handlers can locate session-scoped
    resources without holding a reference to the session itself.
    """

    cwd: str


@dataclass(slots=True)
class BeforeCompactEvent:
    """Fires before an extension performs context compaction.

    Handlers may return ``{"cancel": True}`` to abort the compaction, or
    ``{"replacement": <CompactionResult>}`` to fully replace it. Multiple
    extensions registering on this channel: last non-None replacement wins.

    Mutability: ``messages`` is intentionally mutable (not frozen) so a
    handler can adjust the in-flight buffer before compaction kicks off.
    """

    messages: list[AgentMessage]
    reason: str  # e.g. "auto_overflow", "manual", "scenario_request"


@dataclass(frozen=True, slots=True)
class AfterCompactEvent:
    """Fires after compaction is committed to the SessionManager."""

    summary: str
    kept_message_count: int
    discarded_message_count: int
    details: Any = None  # extension-specific (e.g. artifact index)


@dataclass(frozen=True, slots=True)
class ChildSessionStartEvent:
    """Fires on the parent bus when a child AgentSession is created."""

    child_session_id: str
    parent_session_id: str
    purpose: str  # e.g. "subagent:worker", caller-defined


@dataclass(frozen=True, slots=True)
class ChildSessionEndEvent:
    """Fires on the parent bus when a child AgentSession terminates."""

    child_session_id: str
    parent_session_id: str
    final_message_count: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CostBudgetExceededEvent:
    """Fires when the ``cost_budget`` extension's accumulator crosses the
    configured limit.

    ``AgentSession`` subscribes once at create-time and latches an internal
    flag; the next ``prompt`` short-circuits with an ``agent_end`` event
    carrying ``stop_reason='budget'``. Pure event-bus signalling — no
    exceptions cross handler boundaries.
    """

    used: float
    limit: float
    currency: str = "usd"


@dataclass(frozen=True, slots=True)
class PlanSubmittedEvent:
    """Fires when the ``tool_submit_plan`` tool runs to completion.

    Carries the plan id (entry id returned by ``ReadonlySession.append_entry``)
    so downstream extensions (``trajectory``, plan-mode controllers) can
    correlate the submission to its persisted entry.
    """

    plan_id: str
    plan_text: str


@dataclass(frozen=True, slots=True)
class SessionReadyEvent:
    """Fires once after ``AgentSession.create`` has loaded every extension
    and the active provider has been picked, but before the first ``prompt``.

    This is the only timing point where every extension is guaranteed to see
    the *final* tool list, command set, and model. ``tool_filter`` and
    similar "post-install scrub" extensions hook here.
    """

    cwd: str
    session_id: str
    tool_names: tuple[str, ...]
    command_names: tuple[str, ...]
    model: Model | None


@dataclass(frozen=True, slots=True)
class ExtensionInstallEvent:
    """Fires twice per ``load_extension`` call: ``"start"`` precedes
    ``install(api, config)``; ``"end"`` follows a successful return;
    ``"error"`` follows a thrown exception.
    """

    module_path: str
    config: dict[str, Any]
    phase: Literal["start", "end", "error"]
    duration_ns: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExtensionReloadEvent:
    """Fires when a live session reloads an already-loaded extension."""

    name: str
    old_hash: str | None
    new_hash: str
    trigger: Literal["agent", "human", "propose_change_approved"]
    tier: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ApiRegisterEvent:
    """Fires synchronously from ``ExtensionAPI`` register methods.

    Emitted via ``bus.emit_sync`` so it works from inside sync ``install``
    bodies. Lets subscribers see what each extension contributes.
    """

    kind: Literal["tool", "command", "provider", "renderer"]
    name: str
    extension: str
    payload: Any


@dataclass(frozen=True, slots=True)
class ApiSendUserMessageEvent:
    """Fires when an extension calls ``api.send_user_message``."""

    extension: str
    content: Any


@dataclass(frozen=True, slots=True)
class ResourcesDiscoverEvent:
    """Fires when an extension wants peers to contribute resource paths."""

    cwd: str
    reason: Literal["startup", "reload"]


__all__ = [
    "AfterCompactEvent",
    "ApiRegisterEvent",
    "ApiSendUserMessageEvent",
    "BeforeAgentStartEvent",
    "BeforeCompactEvent",
    "ChildSessionEndEvent",
    "ChildSessionStartEvent",
    "CostBudgetExceededEvent",
    "ExtensionInstallEvent",
    "ExtensionReloadEvent",
    "PlanSubmittedEvent",
    "ResourcesDiscoverEvent",
    "SessionReadyEvent",
    "SessionShutdownEvent",
]
