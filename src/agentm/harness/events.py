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

Layer purity: imports only stdlib + ``agentm.core.abi``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from agentm.core.abi import AgentMessage, Model


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

    CHANNEL: ClassVar[Literal["before_agent_start"]] = "before_agent_start"
    messages: list[AgentMessage]
    system: str | None


@dataclass(frozen=True, slots=True)
class SessionShutdownEvent:
    """Fires when ``AgentSession.shutdown`` is called.

    Carries the session's cwd so cleanup handlers can locate session-scoped
    resources without holding a reference to the session itself.
    """

    CHANNEL: ClassVar[Literal["session_shutdown"]] = "session_shutdown"
    cwd: str


@dataclass(slots=True)
class BeforeCompactEvent:
    """Fires before an extension performs context compaction.

    This is currently an observation-only channel: emitters (``llm_compaction``,
    ``micro_compact``) discard handler return values. Subscribers may inspect
    or mutate ``messages`` in place to influence the buffer that compaction
    will see, but cannot cancel or replace the compaction itself. Adding a
    cancel/replacement contract is on the backlog (PR #65 follow-up) and
    would require both emitters to opt in.

    Mutability: ``messages`` is intentionally mutable (not frozen) so a
    handler can adjust the in-flight buffer before compaction kicks off.
    """

    CHANNEL: ClassVar[Literal["before_compact"]] = "before_compact"
    messages: list[AgentMessage]
    reason: str  # e.g. "auto_overflow", "manual", "scenario_request"


@dataclass(frozen=True, slots=True)
class AfterCompactEvent:
    """Fires after compaction is committed to the SessionManager."""

    CHANNEL: ClassVar[Literal["after_compact"]] = "after_compact"
    summary: str
    kept_message_count: int
    discarded_message_count: int
    details: Any = None  # extension-specific (e.g. artifact index)


@dataclass(frozen=True, slots=True)
class ChildSessionStartEvent:
    """Fires on the parent bus when a child AgentSession is created."""

    CHANNEL: ClassVar[Literal["child_session_start"]] = "child_session_start"
    child_session_id: str
    parent_session_id: str
    purpose: str  # e.g. "subagent:worker", caller-defined


@dataclass(frozen=True, slots=True)
class ChildSessionEndEvent:
    """Fires on the parent bus when a child AgentSession terminates."""

    CHANNEL: ClassVar[Literal["child_session_end"]] = "child_session_end"
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

    CHANNEL: ClassVar[Literal["cost_budget_exceeded"]] = "cost_budget_exceeded"
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

    CHANNEL: ClassVar[Literal["plan_submitted"]] = "plan_submitted"
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

    CHANNEL: ClassVar[Literal["session_ready"]] = "session_ready"
    cwd: str
    session_id: str
    tool_names: tuple[str, ...]
    command_names: tuple[str, ...]
    extension_module_paths: tuple[str, ...]
    model: Model | None
    root_session_id: str
    task_id: str | None = None
    persona: str | None = None


@dataclass(frozen=True, slots=True)
class ResolveSubagentEvent:
    """Request persona metadata for a named sub-agent type.

    The ``sub_agent`` atom emits this typed channel before spawning a child
    session. Scenario atoms may return a mapping with ``body``, ``tools``,
    ``input_schema``, ``budget_defaults``, and ``artifact_kinds`` entries.
    """

    CHANNEL: ClassVar[Literal["resolve_subagent"]] = "resolve_subagent"
    name: str


@dataclass(frozen=True, slots=True)
class ExtensionInstallEvent:
    """Fires twice per ``load_extension`` call: ``"start"`` precedes
    ``install(api, config)``; ``"end"`` follows a successful return;
    ``"error"`` follows a thrown exception.

    ``trigger`` distinguishes who initiated the install. ``"bootstrap"``
    is the default for installs done by ``AgentSession.create`` from a
    scenario or auto-discovery; the other values flow through
    ``api.install_atom``. Subscribers (e.g. the TUI) use this to decide
    whether to surface a "★ self-modify" toast.
    """

    CHANNEL: ClassVar[Literal["extension_install"]] = "extension_install"
    module_path: str
    config: dict[str, Any]
    phase: Literal["start", "end", "error"]
    duration_ns: int = 0
    error: str | None = None
    trigger: Literal["bootstrap", "agent", "human", "propose_change_approved"] = (
        "bootstrap"
    )


@dataclass(frozen=True, slots=True)
class ExtensionReloadEvent:
    """Fires after a transactional reload succeeds or hits rollback failure."""

    CHANNEL: ClassVar[Literal["extension_reload"]] = "extension_reload"
    name: str
    old_hash: str | None
    new_hash: str
    trigger: Literal["agent", "human", "propose_change_approved"]
    tier: int
    error: str | None = None
    is_self_modify: bool = False


@dataclass(slots=True)
class BeforeInstallAtomEvent:
    """Veto hook between internal safety gates and the on-disk write of
    ``api.install_atom``. Handlers may return ``{"block": True, "reason":
    "..."}`` to refuse — first truthy block wins. ``config`` is mutable
    in place; ``source`` and ``name`` are read-only here.
    """

    CHANNEL: ClassVar[Literal["before_install_atom"]] = "before_install_atom"
    name: str
    module_path: str
    target_path: str
    source: str
    config: dict[str, Any]
    tier: int
    trigger: Literal["agent", "human", "propose_change_approved"]


@dataclass(slots=True)
class BeforeUnloadAtomEvent:
    """Veto hook before ``api.unload_atom`` removes an atom. Same contract
    as :class:`BeforeInstallAtomEvent` — return ``{"block": True, "reason":
    "..."}`` to refuse; first truthy block wins.
    """

    CHANNEL: ClassVar[Literal["before_unload_atom"]] = "before_unload_atom"
    name: str
    module_path: str
    tier: int
    trigger: Literal["agent", "human", "propose_change_approved"]


@dataclass(frozen=True, slots=True)
class CommandDispatchedEvent:
    """Fires when ``AgentSession.prompt`` dispatches a slash-command to a
    code-registered handler (i.e. a :class:`CommandSpec` in the session's
    ``commands`` registry). Observation only; the dispatch has already
    happened by the time this fires. Use ``input`` events instead if you
    need to rewrite the text BEFORE dispatch.

    ``args`` is the rest-of-line argument string passed to the command
    handler. ``owner`` is the module path of the atom that registered the
    command, mirroring the ``api_register`` attribution.
    """

    CHANNEL: ClassVar[Literal["command_dispatched"]] = "command_dispatched"
    name: str
    args: str
    owner: str


@dataclass(frozen=True, slots=True)
class ExtensionUnloadEvent:
    """Fires after a successful unload of an installed atom.

    Mirrors ``ExtensionReloadEvent`` so subscribers (TUI, observability)
    can treat install/reload/unload as one family. Provider extensions
    cannot be unloaded; constitution-path atoms cannot be unloaded.
    """

    CHANNEL: ClassVar[Literal["extension_unload"]] = "extension_unload"
    name: str
    module_path: str
    trigger: Literal["agent", "human", "propose_change_approved"]
    tier: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ApiRegisterEvent:
    """Fires synchronously from ``ExtensionAPI`` register methods.

    Emitted via ``bus.emit_sync`` so it works from inside sync ``install``
    bodies. Lets subscribers see what each extension contributes.
    """

    CHANNEL: ClassVar[Literal["api_register"]] = "api_register"
    kind: Literal["tool", "command", "provider", "renderer"]
    name: str
    extension: str
    payload: Any


@dataclass(frozen=True, slots=True)
class ApiSendUserMessageEvent:
    """Fires when an extension calls ``api.send_user_message``."""

    CHANNEL: ClassVar[Literal["api_send_user_message"]] = "api_send_user_message"
    extension: str
    content: Any


@dataclass(frozen=True, slots=True)
class ResourcesDiscoverEvent:
    """Fires when an extension wants peers to contribute resource paths."""

    CHANNEL: ClassVar[Literal["resources_discover"]] = "resources_discover"
    cwd: str
    reason: Literal["startup", "reload"]


@dataclass(frozen=True, slots=True)
class ResourceWriteEvent:
    """Fires when a managed resource write lands as a git commit."""

    CHANNEL: ClassVar[Literal["resource_write"]] = "resource_write"
    path: str
    pre_sha: str
    post_sha: str
    rationale: str
    author: Literal["agent", "human", "indexer"]


__all__ = [
    "AfterCompactEvent",
    "ApiRegisterEvent",
    "ApiSendUserMessageEvent",
    "BeforeAgentStartEvent",
    "BeforeCompactEvent",
    "ChildSessionEndEvent",
    "ChildSessionStartEvent",
    "CostBudgetExceededEvent",
    "BeforeInstallAtomEvent",
    "BeforeUnloadAtomEvent",
    "CommandDispatchedEvent",
    "ExtensionInstallEvent",
    "ExtensionReloadEvent",
    "ExtensionUnloadEvent",
    "PlanSubmittedEvent",
    "ResourceWriteEvent",
    "ResourcesDiscoverEvent",
    "ResolveSubagentEvent",
    "SessionReadyEvent",
    "SessionShutdownEvent",
]
