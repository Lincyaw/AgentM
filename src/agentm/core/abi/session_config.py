"""``AgentSessionConfig`` — atom-facing session construction contract.

Lives in the ABI layer so atoms (e.g. ``tool_eval_run``, llmharness adapter)
can build child-session configs without importing anything in
``agentm.core.runtime``. Runtime-typed fields (:class:`SessionManager`,
:class:`ResourceLoader`) are referenced via TYPE_CHECKING imports — atoms
construct them via CLI helpers or leave them ``None``.

Pluggability hard rule: this module imports only stdlib + sibling ABI
submodules at import time. Runtime references are lazy / TYPE_CHECKING.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from agentm.core.abi import AgentMessage, EventBus, LoopConfig, ProviderResolver
from agentm.core.abi.resource import ResourceWriter

if TYPE_CHECKING:
    # Forward-referenced under TYPE_CHECKING to keep `agentm.core.abi` free of
    # `core.runtime` imports (the layering rule). Field annotations stay as
    # forward-reference strings — combined with `from __future__ import
    # annotations` above, dataclass class-creation never resolves them. NOTE:
    # `typing.get_type_hints(AgentSessionConfig)` is intentionally NOT supported
    # — it would NameError because ABI cannot import runtime. Callers that need
    # the resolved types should reach into `agentm.core.runtime` directly.
    from agentm.core.runtime.resource_loader import ResourceLoader
    from agentm.core.runtime.session_manager import SessionManager



@dataclass(slots=True)
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
    scenario_dir: str | None = None
    extra_extensions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    """Atoms appended after the primary load step (auto-discover or scenario).

    Used by ``--extension`` on the CLI to mount third-party atoms on top of
    a base scenario or the default builtin discovery without forcing the
    user to copy the entire base list into a custom manifest. Ignored when
    ``no_extensions`` is set or when ``extensions`` is supplied explicitly
    (in which case the caller already controls the full list)."""
    extra_tools: list[Any] = field(default_factory=list)
    """FunctionTool instances registered after atom install.

    Lets a parent atom pass tools directly to a child session without
    creating a dedicated atom module. Registered after all extensions
    install, so they win on name conflicts (and the conflict check
    will catch duplicates)."""
    atom_config_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-atom config overrides keyed by ``MANIFEST.name``.

    Overlaid on top of the manifest-supplied ``config:`` and any
    ``AGENTM_<ATOM>_<KEY>`` env var by ``resolve_atom_configs`` at load time
    (precedence: overrides > env > manifest). Populated by the CLI
    ``--set <atom>.<key>=<value>`` flag; embedders may pass typed values
    directly. Keys an atom did not declare in ``config_schema`` are applied
    verbatim (best-effort coerced), so atoms with ``additionalProperties``
    still work. Ignored when ``no_extensions`` is set."""
    no_extensions: bool = False
    no_skills: bool = False
    no_prompt_templates: bool = False
    tool_allowlist: list[str] | None = None
    initial_messages: list[AgentMessage] = field(default_factory=list)
    session_manager: "SessionManager | None" = None
    resource_loader: "ResourceLoader | None" = None
    resource_writer: ResourceWriter | None = None
    """Override the session's `ResourceWriter`. Defaults to
    `GitBackedResourceWriter(cwd, session_id, bus)`. Substrate-injected
    (pre-atom-install) policy: `AtomReloader` and write-path tool atoms
    consume it via `api.get_resource_writer()`, but atoms cannot replace
    it via `register_*` because it must exist before atoms install."""

    auto_commit: bool = True
    """When False, the default `GitBackedResourceWriter` is constructed in
    advisory mode: managed writes still land bytes-on-disk but no `git
    commit` is ever issued. Lets callers safely point AgentM at a real
    user repo without auto-modifying its history. Ignored when
    `resource_writer` is explicitly supplied — the override is honoured
    verbatim."""

    protected_branches: frozenset[str] | None = None
    """Branch names that the default writer refuses to auto-commit to when
    pointed at the user's real repo. `None` keeps the writer's built-in
    default (`{'main', 'master'}`). An empty `frozenset()` disables the
    guard entirely. The check is a no-op when `auto_commit` is False or
    when commits flow through the internal shadow repo under
    `.agentm/repo/`."""
    loop_config: LoopConfig | None = None
    bus: EventBus | None = None

    initial_services: dict[str, Any] = field(default_factory=dict)
    """Entries are merged into the runtime service registry before the
    extension install loop; atoms see them via ``api.get_service()`` at
    install time. Use this when an atom must read a caller-supplied service
    during install or must subscribe to a creation-time event that fires
    inside ``create()``."""

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

    session_id: str | None = None
    """Optional caller-supplied OTel **span_id** for the session-root
    span — lowercase 16-char hex (8 bytes / OTel span_id shape). When
    provided the factory uses it verbatim as the session id and the
    observability sink writes ``<session_id>.jsonl``; otherwise it
    generates ``uuid.uuid4().hex[:16]``. Embedders that already maintain
    an OTel span id (an upstream collector, rcabench-platform, a
    workbuddy job …) can pass it through so the AgentM trace is
    identical to the caller's id without an external mapping. No
    runtime validation — pass a 16-hex string or downstream OTel
    tooling will reject the trace."""

    root_session_id: str | None = None
    """Optional caller-supplied OTel **trace_id** — lowercase 32-char
    hex (16 bytes / OTel trace_id shape). Shared by this session and
    every transitive child spawned via :meth:`spawn_child_session`.
    ``None`` means this session is the root of a fresh trace and the
    factory generates ``uuid.uuid4().hex``."""

    task_id: str | None = None
    """Caller-defined task id for sub-agent dispatch provenance."""

    persona: str | None = None
    """Optional persona / role label for provenance-bearing extensions."""

    purpose: str = "root"
    """Caller-defined purpose label, e.g. ``"subagent:worker"``;
    surfaces verbatim in :class:`ChildSessionStartEvent`."""

    log_trace_command: bool = True
    """When True, session creation logs a copy-pasteable ``agentm trace``
    command. Programmatic wrapper sessions that never receive prompts can set
    this False and surface their meaningful child-session handles instead."""

    trace_label: str | None = None
    """Optional human-facing label for the creation-time trace command. This is
    observability-only; ``purpose`` remains the lifecycle/atom-policy label."""

    lineage: dict[str, Any] | None = None
    """JSON-safe provenance metadata for reconstructing the session tree.

    This is intentionally descriptive rather than behavioral: the runtime only
    persists and emits it so external stores can join root sessions, forks,
    sub-agents, workflow workers, and other derived sessions without coupling
    that analysis to a specific harness implementation."""

    experiment: dict[str, Any] | None = None
    """JSON-safe experiment metadata for case studies and interventions.

    Callers can use this for reminder-injection runs, ablations, eval cases, or
    any other study-level context. AgentM does not interpret it during agent
    execution; observability records it on session start."""

    # --- Per-task evolution loop (see per-task-evolution-loop.md) --------

    task_class: str | None = None
    """Task class label populated on ``session.fingerprint.task_meta`` by the
    observability atom. Production scenarios declare this as a top-level
    ``task_class`` field in ``manifest.yaml``; ``tool_eval_run`` sets it on
    each child session."""

    eval_run_id: str | None = None
    """When set, this session is part of an eval run; surfaces on
    ``task_meta.eval_run_id``."""

    eval_task_id: str | None = None
    """Identifier of the eval-suite task this session is grading (``None`` for
    production traffic). Surfaces on ``task_meta.task_id``. Distinct from
    ``task_id`` (sub-agent dispatch provenance)."""

    atom_source_overrides: dict[str, str] | None = None
    """Map of ``atom_name → new_source``. After normal extension load
    completes, the session walks this map and applies each via the existing
    reload path WITHOUT writing to the working tree — overrides are written
    to a per-session sandbox temp dir, the atom's module file is redirected
    there, and the reload happens against that file. Used by
    ``tool_eval_run`` to evaluate proposed atom versions without mutating
    the source-of-truth tree. Cleaned up on ``AgentSession.shutdown``."""

    def with_bus(self, bus: EventBus) -> "AgentSessionConfig":
        return replace(self, bus=bus)


__all__ = ["AgentSessionConfig"]
