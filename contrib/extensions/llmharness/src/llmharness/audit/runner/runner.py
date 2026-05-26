"""HarnessRunner: single driver for the cognitive-audit pipeline.

P1 of the refactor described in
``.claude/designs/harness-runner.md``. This module lands the in-memory
:class:`CumulativeAuditState`, the :class:`ChildRunner` / :class:`OpSink`
:class:`typing.Protocol` boundaries, the :class:`SidecarWriter` helper,
and the :class:`HarnessRunner` itself. The live adapter
(:mod:`llmharness.adapters.agentm`) is the only consumer in P1 — offline
seams (``StandaloneChildRunner``, ``InMemorySink``,
``replay_pipeline_over_trajectory``) land in P2.

Invariants P1 must preserve (cf. design §4):

* Sidecar shape (``ReplayRecord``) is byte-identical to pre-refactor
  modulo ``ts_ns`` / ``extraction_run_id`` UUID / non-canonical provider
  ``config`` / LLM nondeterminism.
* Cumulative state is hydrated on adapter ``install`` from the live
  session log; thereafter the in-memory copy is authoritative — no
  re-reading of the session log per firing.
* Per-firing extension lists are built via the existing
  ``compose_extractor_extensions`` / ``compose_auditor_extensions``
  composers; payload shape is unchanged.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from agentm.core.abi.messages import AgentMessage, AssistantMessage, ToolResultMessage
from agentm.core.abi.session import SessionEntry

from ...replay.record import ReplayRecord, now_ns, write_record
from ...schema import Edge, Event, Phase, Reminder, Verdict
from ...tools.engine import PhaseResult, run_phase_standalone
from .. import entry_types as _et
from ..auditor import (
    SUBMIT_VERDICT_TOOL_NAME,
    AuditorOutputError,
    RawVerdictOutput,
    compose_auditor_extensions,
)
from ..extractor import (
    FINALIZE_EXTRACTION_TOOL_NAME,
    ExtractionState,
    RawExtractorOutput,
)
from ..graph.fold import fold_graph
from ..graph.ops import GraphOp, parse_op
from ..graph.phase import merge_to_phases
from ..registry import SERVICE_KEY as AUDIT_REGISTRY_SERVICE_KEY
from ..registry import AuditCheckRegistry, CheckContext
from ..seams.session import bind_extractor_state
from ..toolkit.extractor_directive import build_extractor_directive

_logger = logging.getLogger(__name__)

_DEFAULT_RECENT_VERDICTS: Final[int] = _et.RECENT_VERDICTS_FOR_AUDITOR


# --- settings dataclasses ---------------------------------------------------


@dataclass(frozen=True)
class ExtractorSettings:
    """Per-install knobs for assembling a per-firing extractor extension list."""

    extensions: list[tuple[str, dict[str, Any]]]
    compose_kwargs: dict[str, Any]

    @classmethod
    def from_compose_kwargs(
        cls,
        compose_kwargs: dict[str, Any],
        *,
        prompt_override: str | None = None,
    ) -> ExtractorSettings:
        """Build an ``ExtractorSettings`` from a record's ``compose_kwargs``.

        Used by single-firing replay (:func:`replay_extractor_record`) and
        the strict-A/B offline driver — both extract the per-firing
        compose state from a recorded :class:`ReplayRecord`. Honours the
        legacy ``prompt_override`` key from pre-profile replay sidecars.
        """
        from ..extractor import compose_extractor_extensions

        ck = dict(compose_kwargs or {})
        effective_prompt = (
            prompt_override
            if prompt_override is not None
            else ck.get("base_prompt") or ck.get("prompt_override")
        )
        extensions = compose_extractor_extensions(
            base_prompt=effective_prompt,
            observability_config=ck.get("observability_config"),
            tool_call_budget=ck.get("tool_call_budget"),
        )
        return cls(extensions=extensions, compose_kwargs=ck)

    @classmethod
    def default(cls) -> ExtractorSettings:
        """Built-in extractor defaults — same prompt + empty configs the live
        ``rca:harness.sync`` variant uses when no overrides are supplied.

        Used by callers (notably ``agentm_rca`` ``chained_fork`` with a
        ``rca:baseline`` control) that need to drive the offline runner
        without a recorded sidecar to crib settings from.

        Observability is **enabled** by default (``observability_config={}``
        rather than ``None``): the per-firing extractor child session
        writes OTLP/JSON ndjson to ``<cwd>/.agentm/observability/<sid>.jsonl``
        just like the live path does, so downstream consumers see
        symmetric traces for live and offline runs. Passing ``None``
        suppresses the atom entirely, which was the historical default
        and broke offline-trace parity — see commit log for the
        ``chained_fork`` investigation that surfaced this.
        """
        from ..extractor import compose_extractor_extensions
        from ..extractor.prompt import load_extractor_prompt

        base_prompt = load_extractor_prompt("default")
        compose_kwargs: dict[str, Any] = {
            "base_prompt": base_prompt,
            "observability_config": {},
        }
        extensions = compose_extractor_extensions(
            base_prompt=base_prompt,
            observability_config={},
            tool_call_budget=None,
        )
        return cls(extensions=extensions, compose_kwargs=compose_kwargs)


@dataclass(frozen=True)
class AuditorSettings:
    """Per-install knobs for assembling a per-firing auditor extension list.

    Mirrors the live adapter's ``_AuditorSettings`` exactly; field-for-field
    identical so behaviour matches byte-for-byte.
    """

    base_prompt: str
    observability_config: dict[str, Any] | None
    summary_threshold: int
    tools: tuple[str, ...]

    @classmethod
    def from_compose_kwargs(
        cls,
        compose_kwargs: dict[str, Any],
        *,
        prompt_override: str | None = None,
    ) -> AuditorSettings:
        """Build an ``AuditorSettings`` from a record's ``compose_kwargs``."""
        ck = dict(compose_kwargs or {})
        effective_prompt = (
            prompt_override
            if prompt_override is not None
            else ck.get("base_prompt") or ck.get("prompt_override")
        )
        tools_raw = ck.get("tools")
        tools_tuple: tuple[str, ...] = (
            tuple(str(t) for t in tools_raw)
            if isinstance(tools_raw, list)
            else (SUBMIT_VERDICT_TOOL_NAME,)
        )
        return cls(
            base_prompt=effective_prompt or "",
            observability_config=ck.get("observability_config"),
            summary_threshold=int(ck.get("summary_threshold", 30)),
            tools=tools_tuple,
        )

    @classmethod
    def empty(cls) -> AuditorSettings:
        """Sentinel settings for paths that don't fire the auditor.

        Used by :func:`replay_extractor_record`, which constructs a
        :class:`HarnessRunner` purely for its extractor seam — the
        auditor settings are never consulted but the runner's
        constructor demands a value.
        """
        return cls(
            base_prompt="",
            observability_config=None,
            summary_threshold=30,
            tools=(SUBMIT_VERDICT_TOOL_NAME,),
        )

    @classmethod
    def default(cls) -> AuditorSettings:
        """Built-in auditor defaults — same prompt + minimal-profile tools
        the live ``rca:harness.sync`` variant uses when no overrides are
        supplied.

        Distinct from :meth:`empty`: ``empty`` is a sentinel for paths
        that never fire the auditor (``base_prompt=""``); ``default``
        loads the canonical framing so the offline runner can actually
        execute auditor firings.

        Observability is **enabled** by default (``observability_config={}``
        rather than ``None``) so each auditor child session emits the
        same OTLP/JSON trace shape a live auditor would. See the
        symmetric note on :meth:`ExtractorSettings.default`.
        """
        from ..auditor.prompt import load_auditor_prompt

        return cls(
            base_prompt=load_auditor_prompt("minimal"),
            observability_config={},
            summary_threshold=30,
            tools=(SUBMIT_VERDICT_TOOL_NAME,),
        )


# --- cumulative state -------------------------------------------------------


def _bool_safe_int(raw: Any) -> int | None:
    """Coerce ``int`` while rejecting ``bool`` (which is an ``int`` subclass)."""
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


@dataclass
class CumulativeAuditState:
    """Event-sourced graph state + auditor side-channel state.

    Single source of truth for the cumulative view: ops accumulate across
    firings; :meth:`graph_view` folds them on demand into the
    ``(events, edges, phases)`` tuple consumed by the auditor and by the
    extractor's ``recent_graph`` payload.
    """

    ops: list[GraphOp] = field(default_factory=list)
    cursor_last_turn_index: int = -1
    recent_verdicts: collections.deque[dict[str, Any]] = field(
        default_factory=lambda: collections.deque(maxlen=_DEFAULT_RECENT_VERDICTS)
    )
    last_continuation_notes: list[str] = field(default_factory=list)
    firing_id_counter: int = 0
    # Cached fold view; invalidated whenever ``len(ops)`` changes.
    _cached_len: int = -1
    _cached_view: tuple[tuple[Event, ...], tuple[Edge, ...], tuple[Phase, ...]] | None = None
    # Phases live outside the op-log: they are produced by the mechanical
    # ``merge_to_phases`` pass per firing. We accumulate them so the
    # auditor sees the full phase history without re-reading the log.
    _phases: list[Phase] = field(default_factory=list)

    def graph_view(self) -> tuple[tuple[Event, ...], tuple[Edge, ...], tuple[Phase, ...]]:
        """Return ``(events, edges, phases)`` for the current cumulative state."""
        if self._cached_view is not None and self._cached_len == len(self.ops):
            return self._cached_view
        folded = fold_graph(self.ops)
        events = tuple(folded.nodes_list())
        edges = tuple(folded.edges_list())
        phases = tuple(self._phases)
        self._cached_view = (events, edges, phases)
        self._cached_len = len(self.ops)
        return self._cached_view

    def next_event_id(self) -> int:
        events, _edges, _phases = self.graph_view()
        return max((e.id for e in events), default=0) + 1

    def _invalidate_cache(self) -> None:
        self._cached_view = None
        self._cached_len = -1

    def absorb_extractor_firing(
        self,
        *,
        firing_ops: Sequence[GraphOp],
        firing_cursor: int,
        firing_id: int,
        firing_phases: Sequence[Phase] = (),
    ) -> None:
        self.ops.extend(firing_ops)
        self.cursor_last_turn_index = firing_cursor
        self._phases.extend(firing_phases)
        # Bump the counter to one past this firing's id so the next
        # ops-bearing firing gets a fresh slot. Failed / empty firings
        # don't burn an id (the caller controls when to bump via
        # ``firing_id_counter`` directly).
        if firing_id >= self.firing_id_counter:
            self.firing_id_counter = firing_id + 1
        self._invalidate_cache()

    def absorb_auditor_verdict(self, verdict: dict[str, Any], *, is_silent: bool) -> None:
        del is_silent  # currently informational only; both shapes append
        self.recent_verdicts.append(verdict)
        raw_notes = verdict.get("continuation_notes")
        if isinstance(raw_notes, list):
            self.last_continuation_notes = [n for n in raw_notes if isinstance(n, str)]

    @classmethod
    def fresh(cls) -> CumulativeAuditState:
        """Offline-mode seed: empty state."""
        return cls()

    def snapshot(self) -> CumulativeAuditState:
        """Return an independent deep copy of this cumulative state.

        Used by the fork-tree driver to capture the auditor state *as of*
        a surface point: the snapshot is handed to a child fork as its
        ``seed_cumulative`` and must not change when the backbone replay
        keeps mutating the live state afterward. Every mutable container
        (``ops``, ``recent_verdicts``, ``last_continuation_notes``,
        ``_phases``) is copied; the frozen :class:`GraphOp` / :class:`Phase`
        elements are safe to share but ``deepcopy`` copies them too. The
        ``recent_verdicts`` deque's ``maxlen`` is preserved.
        """
        return copy.deepcopy(self)

    @classmethod
    def hydrate_from_session_log(cls, branch: list[SessionEntry]) -> CumulativeAuditState:
        """Live-mode seed: walk session entries, populate ops + verdicts + cursor.

        Reads the v4 entry shape only:

        * ``AUDIT_GRAPH_OP`` → :func:`parse_op`-decoded :class:`GraphOp`.
        * ``VERDICT`` → recent-verdicts deque (bounded).
        * ``EXTRACTOR_CURSOR`` → ``cursor_last_turn_index``.

        Phases are not persisted to the entry tree — they are derived at
        ``graph_view()`` time from the folded ops.
        """
        ops: list[GraphOp] = []
        verdicts_all: list[dict[str, Any]] = []
        cursor_last_turn_index = -1

        for entry in branch:
            payload = entry.payload
            if not isinstance(payload, dict):
                continue
            if entry.type == _et.AUDIT_GRAPH_OP:
                try:
                    ops.append(parse_op(payload))
                except (KeyError, ValueError, TypeError):
                    continue
            elif entry.type == _et.VERDICT:
                verdicts_all.append(payload)
            elif entry.type == _et.EXTRACTOR_CURSOR:
                raw = _bool_safe_int(payload.get("last_turn_index"))
                if raw is not None:
                    cursor_last_turn_index = raw

        last_notes: list[str] = []
        if verdicts_all:
            raw_notes = verdicts_all[-1].get("continuation_notes")
            if isinstance(raw_notes, list):
                last_notes = [n for n in raw_notes if isinstance(n, str)]

        recent: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=_DEFAULT_RECENT_VERDICTS
        )
        for v in verdicts_all[-_DEFAULT_RECENT_VERDICTS:]:
            recent.append(v)

        # TODO(harness-runner-p2): legacy adapter installed
        # ``firing_counter = [0]`` unconditionally, so on session restart a
        # colliding firing_id was possible. P1 is a pure refactor — keep
        # that exact (broken) behaviour here; the collision-on-restart fix
        # belongs in a separate P2 change.
        state = cls(
            ops=ops,
            cursor_last_turn_index=cursor_last_turn_index,
            recent_verdicts=recent,
            last_continuation_notes=last_notes,
            firing_id_counter=0,
        )
        return state


# --- protocols --------------------------------------------------------------


@dataclass(frozen=True)
class AuditorChildResult:
    """Outcome of one :meth:`ChildRunner.run_auditor` call.

    Carries the parsed verdict (``None`` for spawn / prompt / no-call /
    malformed) plus diagnostics so the synthetic auditor
    :class:`ReplayRecord` constructed by
    :meth:`HarnessRunner.fire_auditor_once` can surface the same
    ``error`` / ``latency_ms`` the sidecar would carry. Without these
    fields the synthetic record drops them silently — CLI tools that
    read ``latency_ms`` then report 0 for every firing.
    """

    verdict: Verdict | None
    raw_blocks: list[dict[str, Any]]
    error: str | None = None
    latency_ms: int = 0


class ChildRunner(Protocol):
    """How a single child phase is invoked."""

    async def run_extractor(
        self,
        *,
        extensions: list[tuple[str, dict[str, Any]]],
        provider: tuple[str, dict[str, Any]] | None,
        payload: dict[str, Any],
        turn_window: list[int],
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Run the extractor child.

        Returns ``(terminator_called, raw_assistant_blocks)``. Spawn /
        prompt failures must be signalled by raising
        :class:`ExtractorSpawnError`.
        """
        ...

    async def run_auditor(
        self,
        *,
        extensions: list[tuple[str, dict[str, Any]]],
        provider: tuple[str, dict[str, Any]] | None,
        graph_events: list[Event],
        recent_verdicts: list[dict[str, Any]],
        continuation_notes_from_prior_firing: list[str],
    ) -> AuditorChildResult:
        """Run the auditor child.

        Returns an :class:`AuditorChildResult` with ``verdict`` set to
        ``None`` for spawn / prompt / no-call / malformed paths. The
        implementation is responsible for recording failures via the
        runner's sink (the live impl uses ``api.session.append_entry``).
        ``error`` / ``latency_ms`` are surfaced on the synthetic
        :class:`ReplayRecord` produced by the runner.
        """
        ...


class OpSink(Protocol):
    """Where ops + cursor + verdict entries are persisted.

    The live path appends to the AgentM session log; the (P2) offline
    path is a no-op.
    """

    def append_op(
        self,
        op: GraphOp,
        *,
        firing_id: int,
        op_index: int,
        turn_window: list[int],
    ) -> None: ...

    def append_cursor(self, *, last_turn_index: int) -> None: ...

    def append_verdict(self, verdict: dict[str, Any]) -> None: ...

    def append_failure(self, entry_type: str, payload: dict[str, Any]) -> None: ...

    def append_partial(self, payload: dict[str, Any]) -> None:
        """Persist an ``EXTRACTOR_PARTIAL`` entry without emitting a diagnostic.

        Distinct from :meth:`append_failure` because the legacy
        ``_drain_extractor`` wrote ``EXTRACTOR_PARTIAL`` via a plain
        ``api.session.append_entry`` — no ``DiagnosticEvent``. The other
        failure entries (``EXTRACTOR_NO_CALL`` / ``EXTRACTOR_ERROR`` /
        ``EXTRACTOR_EMPTY`` / ``AUDIT_NO_CALL`` / ``AUDIT_ERROR``) all
        kept their diagnostic emission in legacy, so they stay on
        :meth:`append_failure`.
        """
        ...


class ExtractorSpawnError(RuntimeError):
    """Wraps spawn / prompt failures so callers can route them to failure entries."""


# --- sidecar writer ---------------------------------------------------------


class SidecarWriter:
    """Wraps :func:`write_record` for a fixed sidecar path.

    Behaviour matches the legacy ``_record_replay_at`` chokepoint
    byte-for-byte (modulo non-deterministic fields like ``ts_ns``).
    Passing ``path=None`` short-circuits every call.
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path

    @property
    def path(self) -> Path | None:
        return self._path

    def record(
        self,
        *,
        phase: str,
        turn_index: int,
        session_id: str,
        trace_id: str,
        compose_kwargs: dict[str, Any] | None,
        payload: dict[str, Any] | None,
        provider: tuple[str, dict[str, Any]] | None,
        output: dict[str, Any] | None,
        status: str,
        error: str | None = None,
        latency_ms: int = 0,
        extras: dict[str, Any] | None = None,
        raw_assistant_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._path is None:
            return
        rec = ReplayRecord(
            phase=phase,  # type: ignore[arg-type]
            turn_index=turn_index,
            session_id=session_id,
            trace_id=trace_id,
            ts_ns=now_ns(),
            compose_kwargs=compose_kwargs or {},
            payload=payload or {},
            provider=[provider[0], provider[1]] if provider else None,
            output=output,
            status=status,  # type: ignore[arg-type]
            error=error,
            latency_ms=latency_ms,
            extras=extras or {},
            raw_assistant_messages=list(raw_assistant_messages or []),
        )
        write_record(self._path, rec)


# --- step result ------------------------------------------------------------


@dataclass(frozen=True)
class StepResult:
    """Outcome of one ``HarnessRunner.on_trajectory_progress`` invocation."""

    fired_extractor: bool
    fired_auditor: bool
    surfaced_reminder: Reminder | None
    auditor_record: ReplayRecord | None = None
    """Synthetic auditor :class:`ReplayRecord` for the firing, when one
    occurred. Populated even when the sidecar writer is ``None`` so
    offline drivers can capture per-firing records without a disk
    round-trip. ``None`` when no auditor firing happened or the firing
    produced no verdict (no-call)."""
    extractor_record: ReplayRecord | None = None
    """Synthetic extractor :class:`ReplayRecord` for the firing, when
    one occurred. Mirrors :attr:`auditor_record`: populated even when
    the sidecar writer is ``None`` so multi-node offline drivers
    (notably :func:`llmharness.replay.run_fork_tree_experiment`) can
    assemble a fork-tree replay sidecar without re-invoking the runner or
    routing through a per-node temp file. ``None`` when the firing
    short-circuited before producing a record (e.g. an empty
    window)."""


# --- helpers ----------------------------------------------------------------


def _window_is_non_trivial(messages_slice: list[AgentMessage]) -> bool:
    """True iff the slice contains any AssistantMessage or ToolResultMessage."""
    return any(isinstance(msg, (AssistantMessage, ToolResultMessage)) for msg in messages_slice)


def _render_message_text(msg: AgentMessage) -> str:
    """Concatenate every text-bearing block; mirror the live adapter exactly."""
    parts: list[str] = []
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            inner = getattr(block, "content", None)
            if isinstance(inner, list):
                for sub in inner:
                    sub_text = getattr(sub, "text", None)
                    if isinstance(sub_text, str) and sub_text:
                        parts.append(sub_text)
            args = getattr(block, "arguments", None)
            if isinstance(args, dict):
                with contextlib.suppress(TypeError, ValueError):
                    parts.append(json.dumps(args, ensure_ascii=False, default=str))
    return " ".join(parts)


def _serialize_block(block: Any) -> dict[str, Any] | None:
    """Mirror of the live ``_serialize_block``."""
    text = getattr(block, "text", None)
    if isinstance(text, str) and text:
        block_type = getattr(block, "type", None)
        return {
            "type": block_type if isinstance(block_type, str) and block_type else "text",
            "text": text,
        }

    name = getattr(block, "name", None)
    arguments = getattr(block, "arguments", None)
    if isinstance(name, str) and isinstance(arguments, dict):
        return {
            "type": "tool_call",
            "id": getattr(block, "id", None),
            "name": name,
            "arguments": dict(arguments),
        }

    tool_call_id = getattr(block, "tool_call_id", None)
    inner_content = getattr(block, "content", None)
    if isinstance(tool_call_id, str) and isinstance(inner_content, list):
        inner_blocks: list[dict[str, Any]] = []
        for inner in inner_content:
            inner_text = getattr(inner, "text", None)
            if isinstance(inner_text, str):
                inner_blocks.append({"type": "text", "text": inner_text})
            else:
                inner_blocks.append(
                    {
                        "type": getattr(inner, "type", inner.__class__.__name__),
                        "repr": repr(inner),
                    }
                )
        return {
            "type": "tool_result",
            "tool_call_id": tool_call_id,
            "content": inner_blocks,
            "is_error": bool(getattr(block, "is_error", False)),
        }

    return {"type": getattr(block, "type", block.__class__.__name__), "repr": repr(block)}


def _serialize_message_for_extractor(msg: AgentMessage, *, index: int) -> dict[str, Any] | None:
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return None
    blocks = [b for b in (_serialize_block(blk) for blk in content) if b is not None]
    if not blocks:
        return None
    return {
        "index": index,
        "role": getattr(msg, "role", "unknown"),
        "content": blocks,
    }


def _serialize_full_trajectory(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        serialized = _serialize_message_for_extractor(msg, index=i)
        if serialized is not None:
            out.append(serialized)
    return out


def _flatten_assistant_blocks(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for blk in content:
            serialized = _serialize_block(blk)
            if serialized is not None:
                blocks.append(serialized)
    return blocks


# --- runner -----------------------------------------------------------------


@dataclass(frozen=True)
class _ExtractorFiringResult:
    """Internal — returned from :meth:`HarnessRunner.fire_extractor_once`."""

    ok: bool
    cursor_advanced: bool
    record: ReplayRecord | None = None


@dataclass(frozen=True)
class _AuditorFiringResult:
    """Internal — returned from :meth:`HarnessRunner.fire_auditor_once`."""

    verdict: Verdict | None
    record: ReplayRecord | None = None


class HarnessRunner:
    """Single driver for the cognitive-audit pipeline.

    Owns cadence, windowing, cumulative-state threading, payload
    composition, and sidecar emission. Parameterised by a
    :class:`ChildRunner` (how a phase is spawned) and an :class:`OpSink`
    (where ops are persisted).
    """

    def __init__(
        self,
        *,
        cumulative: CumulativeAuditState,
        child: ChildRunner,
        sink: OpSink,
        sidecar: SidecarWriter | None,
        extractor_settings: ExtractorSettings,
        auditor_settings: AuditorSettings,
        extractor_interval: int,
        audit_interval: int,
        enable_auditor: bool,
        session_id: str,
        trace_id: str,
        provider_extractor: tuple[str, dict[str, Any]] | None,
        provider_auditor: tuple[str, dict[str, Any]] | None,
        # In P1 the live audit-check registry lives on the parent
        # ExtensionAPI; passing it in keeps the runner independent of
        # ExtensionAPI so the (P2) offline driver can supply a synthetic
        # registry or ``None``.
        audit_registry: AuditCheckRegistry | None = None,
        # ``cwd`` is only consulted by the single-firing replay seams
        # (``fire_extractor_from_record`` / ``fire_auditor_from_record``)
        # which spawn standalone sessions directly. Live + trajectory-
        # driven paths route through the ``child`` :class:`ChildRunner`
        # and never read this field — pass ``None`` for those callers.
        # Passing an explicit ``""`` was a footgun: empty-string ``cwd``
        # flows into ``AgentSessionConfig(cwd="")`` and silently writes
        # to the process CWD.
        cwd: str | None = None,
    ) -> None:
        self.cumulative = cumulative
        self._child = child
        self._sink = sink
        self._sidecar = sidecar
        self._extractor_settings = extractor_settings
        self._auditor_settings = auditor_settings
        self._extractor_interval = max(1, int(extractor_interval))
        self._audit_interval = max(1, int(audit_interval))
        self._enable_auditor = enable_auditor
        self._session_id = session_id
        self._trace_id = trace_id
        self._provider_extractor = provider_extractor
        self._provider_auditor = provider_auditor
        self._audit_registry = audit_registry
        self._cwd = cwd
        # Track whether the most recent extractor firing held the cursor;
        # mirrors ``_drain_queue``'s ``_last_extractor_held_cursor`` so
        # the auditor doesn't fire on top of stale state.
        self._last_extractor_held_cursor: bool = False

    def set_audit_registry(self, registry: AuditCheckRegistry | None) -> None:
        """Public seam for the live adapter to refresh the registry.

        Audit-check atoms may install themselves *after* the harness
        atom (the adapter chain composes them in any order), so a
        ``None`` at construction time doesn't mean "no checks ever".
        The live adapter re-resolves and calls this method each firing.
        """
        self._audit_registry = registry

    async def on_trajectory_progress(
        self,
        messages: list[AgentMessage],
        *,
        turn_count: int,
    ) -> StepResult:
        """One cadence step. Decides whether to fire extractor / auditor."""
        auditor_due = self._enable_auditor and (turn_count % self._audit_interval) == 0
        extractor_due = (turn_count % self._extractor_interval) == 0 or auditor_due

        fired_extractor = False
        fired_auditor = False
        surfaced: Reminder | None = None
        auditor_record: ReplayRecord | None = None
        extractor_record: ReplayRecord | None = None

        if extractor_due:
            result = await self.fire_extractor_once(messages)
            fired_extractor = True
            self._last_extractor_held_cursor = not result.ok
            extractor_record = result.record

        if auditor_due:
            if self._last_extractor_held_cursor:
                _logger.debug(
                    "HarnessRunner: skipping auditor — preceding extractor firing held the cursor"
                )
            else:
                audit_result = await self.fire_auditor_once(messages)
                fired_auditor = True
                auditor_record = audit_result.record
                if audit_result.verdict is not None and audit_result.verdict.surface_reminder:
                    text = audit_result.verdict.reminder_text
                    if text:
                        surfaced = Reminder(text=text)

        return StepResult(
            auditor_record=auditor_record,
            extractor_record=extractor_record,
            fired_extractor=fired_extractor,
            fired_auditor=fired_auditor,
            surfaced_reminder=surfaced,
        )

    # ----------------------------------------------------------------------
    # Phase 1 — extractor
    # ----------------------------------------------------------------------

    async def fire_extractor_once(
        self,
        messages: list[AgentMessage],
    ) -> _ExtractorFiringResult:
        """Run one extractor firing against the current cumulative state.

        Mirrors the legacy ``_drain_extractor`` exactly (windowing,
        payload composition, replay snapshot, cursor handling).
        """
        window_lo = max(self.cumulative.cursor_last_turn_index + 1, 0)
        window_hi_inclusive = len(messages) - 1
        window_messages = messages[window_lo:]
        if not window_messages:
            return _ExtractorFiringResult(ok=True, cursor_advanced=False)

        turn_window = [window_lo, window_hi_inclusive]

        events_cum, edges_cum, _phases_cum = self.cumulative.graph_view()

        state = ExtractionState()
        for i, msg in enumerate(window_messages, start=window_lo):
            state.turn_texts[i] = _render_message_text(msg)

        referenced_turns: set[int] = {t for e in events_cum for t in e.source_turns}
        for t in referenced_turns:
            if t in state.turn_texts:
                continue
            if 0 <= t < len(messages):
                state.turn_texts[t] = _render_message_text(messages[t])
        state.recent_graph = events_cum
        state.recent_graph_dict = {e.id: e for e in events_cum}
        state.recent_edges_dict = {(ed.src, ed.dst, ed.kind.value): ed for ed in edges_cum}
        state._refold()
        state.next_event_id = self.cumulative.next_event_id()

        new_turn_window = [
            s
            for s in (
                _serialize_message_for_extractor(msg, index=i)
                for i, msg in enumerate(window_messages, start=window_lo)
            )
            if s is not None
        ]

        recent_graph_payload: list[dict[str, Any]] = []
        for e in events_cum:
            entry = e.to_dict()
            entry["source_turn_texts"] = [state.turn_texts.get(t, "") for t in e.source_turns]
            recent_graph_payload.append(entry)

        payload = {
            "next_event_id": state.next_event_id,
            "new_turns": new_turn_window,
            "graph": {
                "nodes": recent_graph_payload,
                "edges": [ed.to_dict() for ed in edges_cum],
            },
            # Back-compat for older extractor prompts / replay records. New code
            # should read graph.{nodes,edges}.
            "recent_graph": recent_graph_payload,
            "recent_edges": [ed.to_dict() for ed in edges_cum],
        }
        tool_call_budget = self._extractor_settings.compose_kwargs.get("tool_call_budget")
        if isinstance(tool_call_budget, int) and tool_call_budget > 0:
            payload["tool_call_budget"] = tool_call_budget

        # Inject state into the per-firing extensions list. The base
        # list returned by ``compose_extractor_extensions`` is shared
        # across firings; ``bind_extractor_state`` returns a copy with
        # the ``state`` config knob set on the extractor-tools atom.
        firing_extensions = bind_extractor_state(
            self._extractor_settings.extensions,
            state=state,
        )

        # Always materialise the per-firing replay-record metadata so a
        # synthetic :class:`ReplayRecord` can be attached to the
        # returned :class:`StepResult` regardless of whether the sidecar
        # writer is mounted. Offline chained-fork drivers depend on this
        # to assemble a multi-segment replay file without re-invoking
        # the runner.
        replay_compose_kwargs: dict[str, Any] = dict(self._extractor_settings.compose_kwargs)
        replay_extras: dict[str, Any] = {
            "turn_texts": {str(k): v for k, v in state.turn_texts.items()},
        }

        raw_assistant_messages: list[dict[str, Any]] = []
        synth_record: ReplayRecord | None = None

        def _record(status: str, output: dict[str, Any] | None, error: str | None = None) -> None:
            nonlocal synth_record
            synth_record = ReplayRecord(
                phase="extractor",
                turn_index=window_hi_inclusive,
                session_id=self._session_id,
                trace_id=self._trace_id,
                ts_ns=now_ns(),
                compose_kwargs=dict(replay_compose_kwargs),
                payload=dict(payload),
                provider=(
                    [self._provider_extractor[0], self._provider_extractor[1]]
                    if self._provider_extractor
                    else None
                ),
                output=output,
                status=status,  # type: ignore[arg-type]
                error=error,
                latency_ms=0,
                extras=dict(replay_extras),
                raw_assistant_messages=list(raw_assistant_messages),
            )
            if self._sidecar is None:
                return
            self._sidecar.record(
                phase="extractor",
                turn_index=window_hi_inclusive,
                session_id=self._session_id,
                trace_id=self._trace_id,
                compose_kwargs=replay_compose_kwargs,
                payload=payload,
                provider=self._provider_extractor,
                output=output,
                status=status,
                error=error,
                extras=replay_extras,
                raw_assistant_messages=raw_assistant_messages,
            )

        try:
            terminator_called, raw_assistant_messages = await self._child.run_extractor(
                extensions=firing_extensions,
                provider=self._provider_extractor,
                payload=payload,
                turn_window=turn_window,
            )
        except ExtractorSpawnError as exc:
            self._sink.append_failure(
                _et.EXTRACTOR_ERROR,
                {"reason": str(exc), "turn_window": turn_window},
            )
            _record("spawn_error", output=None, error=str(exc))
            return _ExtractorFiringResult(ok=False, cursor_advanced=False, record=synth_record)

        if not terminator_called:
            self._sink.append_failure(
                _et.EXTRACTOR_NO_CALL,
                {
                    "reason": (f"child returned without calling {FINALIZE_EXTRACTION_TOOL_NAME}"),
                    "turn_window": turn_window,
                },
            )
            _record("no_call", output=None)
            return _ExtractorFiringResult(ok=False, cursor_advanced=False, record=synth_record)

        output = RawExtractorOutput.from_state(state)
        _record(
            "ok",
            output={
                "events": [e.to_dict() for e in output.events],
                "edges": [ed.to_dict() for ed in output.edges],
                "dropped_edges": list(output.dropped_edges),
                "ops": [op.to_dict() for op in state.pending_ops],
            },
        )

        has_ops = bool(state.pending_ops)

        if not has_ops and not output.dropped_edges:
            if _window_is_non_trivial(window_messages):
                self._sink.append_failure(_et.EXTRACTOR_EMPTY, {"turn_window": turn_window})
                return _ExtractorFiringResult(ok=False, cursor_advanced=False, record=synth_record)
            # Truly trivial window: still advance the cursor so we don't
            # re-extract the same prefix forever.
            self._sink.append_cursor(last_turn_index=window_hi_inclusive)
            self.cumulative.cursor_last_turn_index = window_hi_inclusive
            return _ExtractorFiringResult(ok=True, cursor_advanced=True, record=synth_record)

        firing_id = self.cumulative.firing_id_counter
        for op_index, op in enumerate(state.pending_ops):
            self._sink.append_op(
                op,
                firing_id=firing_id,
                op_index=op_index,
                turn_window=list(turn_window),
            )

        firing_phases = list(merge_to_phases(output.events))

        if output.dropped_edges:
            self._sink.append_partial(
                {
                    "dropped_edges": list(output.dropped_edges),
                    "turn_window": turn_window,
                },
            )

        self._sink.append_cursor(last_turn_index=window_hi_inclusive)

        self.cumulative.absorb_extractor_firing(
            firing_ops=list(state.pending_ops),
            firing_cursor=window_hi_inclusive,
            firing_id=firing_id,
            firing_phases=firing_phases,
        )

        return _ExtractorFiringResult(ok=True, cursor_advanced=True, record=synth_record)

    # ----------------------------------------------------------------------
    # Phase 2 — auditor
    # ----------------------------------------------------------------------

    async def fire_auditor_once(
        self,
        messages: list[AgentMessage],
    ) -> _AuditorFiringResult:
        events_tuple, edges_tuple, phases_tuple = self.cumulative.graph_view()

        findings: list[Any] = []
        check_errors: dict[str, str] = {}
        registry = self._audit_registry
        if isinstance(registry, AuditCheckRegistry):
            try:
                ctx = CheckContext(events=events_tuple, edges=edges_tuple)
                findings, check_errors = registry.run_all(ctx)
            except Exception:
                _logger.exception("audit-check registry run_all failed; using empty findings")
                findings, check_errors = [], {}

        trajectory_snapshot = _serialize_full_trajectory(messages)
        continuation_notes = list(self.cumulative.last_continuation_notes)
        recent_verdicts = list(self.cumulative.recent_verdicts)

        firing_extensions = compose_auditor_extensions(
            base_prompt=self._auditor_settings.base_prompt,
            observability_config=self._auditor_settings.observability_config,
            trajectory_snapshot=trajectory_snapshot,
            events=events_tuple,
            edges=edges_tuple,
            phases=phases_tuple,
            findings=list(findings),
            check_errors=dict(check_errors),
            continuation_notes=continuation_notes,
            summary_threshold=self._auditor_settings.summary_threshold,
            tools=self._auditor_settings.tools,
        )

        # Always compute the replay record metadata (independent of
        # sidecar emission) so the runner can return a synthetic
        # auditor :class:`ReplayRecord` per firing via
        # :class:`StepResult.auditor_record`. Offline drivers depend
        # on this for in-memory record capture without a disk
        # round-trip.
        replay_compose_kwargs: dict[str, Any] = {
            "base_prompt": self._auditor_settings.base_prompt,
            "observability_config": self._auditor_settings.observability_config,
            "trajectory_snapshot": trajectory_snapshot,
            "events": [e.to_dict() for e in events_tuple],
            "edges": [ed.to_dict() for ed in edges_tuple],
            "phases": [ph.to_dict() for ph in phases_tuple],
            "findings": [f.to_dict() for f in findings],
            "check_errors": dict(check_errors),
            "continuation_notes": continuation_notes,
            "summary_threshold": self._auditor_settings.summary_threshold,
            "tools": list(self._auditor_settings.tools),
        }
        replay_payload: dict[str, Any] = {
            "graph": [e.to_dict() for e in events_tuple],
            "recent_verdicts": recent_verdicts,
            "continuation_notes_from_prior_firing": continuation_notes,
        }

        child_result = await self._child.run_auditor(
            extensions=firing_extensions,
            provider=self._provider_auditor,
            graph_events=list(events_tuple),
            recent_verdicts=recent_verdicts,
            continuation_notes_from_prior_firing=continuation_notes,
        )
        verdict = child_result.verdict
        raw_blocks = child_result.raw_blocks
        latency_ms = child_result.latency_ms
        child_error = child_result.error
        turn_index = len(messages) - 1 if messages else -1

        if verdict is None:
            if self._sidecar is not None:
                self._sidecar.record(
                    phase="auditor",
                    turn_index=turn_index,
                    session_id=self._session_id,
                    trace_id=self._trace_id,
                    compose_kwargs=replay_compose_kwargs,
                    payload=replay_payload,
                    provider=self._provider_auditor,
                    output=None,
                    status="no_call",
                    error=child_error,
                    latency_ms=latency_ms,
                    raw_assistant_messages=raw_blocks,
                )
            no_call_record = ReplayRecord(
                phase="auditor",
                turn_index=turn_index,
                session_id=self._session_id,
                trace_id=self._trace_id,
                ts_ns=now_ns(),
                compose_kwargs=replay_compose_kwargs,
                payload=replay_payload,
                provider=(
                    [self._provider_auditor[0], self._provider_auditor[1]]
                    if self._provider_auditor
                    else None
                ),
                output=None,
                status="no_call",
                error=child_error,
                latency_ms=latency_ms,
                raw_assistant_messages=list(raw_blocks),
            )
            return _AuditorFiringResult(verdict=None, record=no_call_record)

        verdict_dict = verdict.to_dict()
        self._sink.append_verdict(verdict_dict)
        self.cumulative.absorb_auditor_verdict(verdict_dict, is_silent=not verdict.surface_reminder)
        if self._sidecar is not None:
            self._sidecar.record(
                phase="auditor",
                turn_index=turn_index,
                session_id=self._session_id,
                trace_id=self._trace_id,
                compose_kwargs=replay_compose_kwargs,
                payload=replay_payload,
                provider=self._provider_auditor,
                output=verdict_dict,
                status="ok",
                latency_ms=latency_ms,
                raw_assistant_messages=raw_blocks,
            )
        ok_record = ReplayRecord(
            phase="auditor",
            turn_index=turn_index,
            session_id=self._session_id,
            trace_id=self._trace_id,
            ts_ns=now_ns(),
            compose_kwargs=replay_compose_kwargs,
            payload=replay_payload,
            provider=(
                [self._provider_auditor[0], self._provider_auditor[1]]
                if self._provider_auditor
                else None
            ),
            output=verdict_dict,
            status="ok",
            latency_ms=latency_ms,
            raw_assistant_messages=list(raw_blocks),
        )
        return _AuditorFiringResult(verdict=verdict, record=ok_record)

    # ----------------------------------------------------------------------
    # Single-firing replay (degenerate case of the runner — invariant #2)
    # ----------------------------------------------------------------------

    async def fire_extractor_from_record(self, record: ReplayRecord) -> PhaseResult:
        """Replay one recorded extractor firing without windowing.

        Bypasses the ``messages → turn_window → payload`` construction
        in :meth:`fire_extractor_once`; the record already carries a
        finished ``payload`` (``graph.nodes`` / ``graph.edges``, plus
        legacy ``recent_graph`` / ``recent_edges``, ``next_event_id``,
        and ``new_turns``). We rebuild an :class:`ExtractionState` from
        ``payload`` + ``extras.turn_texts`` exactly as legacy
        ``replay_extractor_record`` did, then call
        :func:`run_phase_standalone` and snapshot
        :class:`RawExtractorOutput` from the bound state. The returned
        :class:`PhaseResult` schema (with ``events`` / ``edges`` /
        ``dropped_edges`` on success) is byte-identical to the legacy
        wrapper output — many tests pin this.
        """
        from ..extractor.state import ExtractionState

        if record.phase != "extractor":
            raise ValueError(f"expected extractor record, got phase={record.phase!r}")
        if self._cwd is None:
            raise ValueError(
                "HarnessRunner.fire_extractor_from_record requires cwd to be set at construction"
            )

        extras = record.extras or {}
        state = ExtractionState()
        # next_event_id: record payload carries the firing's id cursor.
        nxt = (record.payload or {}).get("next_event_id")
        if isinstance(nxt, int) and nxt >= 1:
            state.next_event_id = nxt

        # JSON-loaded turn_texts has string keys; ExtractionState
        # expects ints.
        for k, v in (extras.get("turn_texts") or {}).items():
            with contextlib.suppress(TypeError, ValueError):
                state.turn_texts[int(k)] = str(v)
        # Union in any prior-firing turn texts the caller supplied
        # (the CLI computes this from earlier records in the sidecar so
        # external_refs can be witnessed during replay — the live
        # adapter does the same enrichment at firing time).
        for k, v in (extras.get("prior_turn_texts") or {}).items():
            with contextlib.suppress(TypeError, ValueError):
                state.turn_texts.setdefault(int(k), str(v))

        # Enrich recent_graph entries with source_turn_texts and
        # populate state.recent_graph so external_refs can be
        # witnessed against trajectory text.
        payload = dict(record.payload or {})
        graph_obj = payload.get("graph")
        graph_raw: dict[str, Any] = graph_obj if isinstance(graph_obj, dict) else {}
        recent_graph_raw = graph_raw.get("nodes") or payload.get("recent_graph") or []
        recent_edges_raw = graph_raw.get("edges") or payload.get("recent_edges") or []
        enriched_recent: list[dict[str, Any]] = []
        recent_events: list[Event] = []
        for entry in recent_graph_raw:
            if not isinstance(entry, dict):
                continue
            copy = dict(entry)
            copy["source_turn_texts"] = [
                state.turn_texts.get(int(t), "")
                for t in (entry.get("source_turns") or [])
                if isinstance(t, int)
            ]
            enriched_recent.append(copy)
            try:
                recent_events.append(Event.from_dict(entry))
            except (KeyError, ValueError, TypeError):
                continue
        payload["graph"] = {"nodes": enriched_recent, "edges": list(recent_edges_raw)}
        payload["recent_graph"] = enriched_recent
        payload["recent_edges"] = list(recent_edges_raw)
        tool_call_budget = self._extractor_settings.compose_kwargs.get("tool_call_budget")
        if isinstance(tool_call_budget, int) and tool_call_budget > 0:
            payload["tool_call_budget"] = tool_call_budget
        state.recent_graph = tuple(recent_events)
        state.recent_graph_dict = {e.id: e for e in recent_events}

        recent_edges: list[Edge] = []
        for entry in recent_edges_raw:
            if not isinstance(entry, dict):
                continue
            try:
                recent_edges.append(Edge.from_dict(entry))
            except (KeyError, ValueError, TypeError):
                continue
        state.recent_edges_dict = {(ed.src, ed.dst, ed.kind.value): ed for ed in recent_edges}
        state._refold()

        extensions = bind_extractor_state(self._extractor_settings.extensions, state=state)

        result = await run_phase_standalone(
            cwd=self._cwd,
            extensions=extensions,
            provider=self._provider_extractor,
            payload=build_extractor_directive(payload)
            + json.dumps(payload, ensure_ascii=False, default=str),
            terminal_tool=FINALIZE_EXTRACTION_TOOL_NAME,
            purpose="cognitive_audit_extractor_replay",
        )
        if result.status == "ok":
            snapshot = RawExtractorOutput.from_state(state)
            result = PhaseResult(
                output={
                    "events": [e.to_dict() for e in snapshot.events],
                    "edges": [ed.to_dict() for ed in snapshot.edges],
                    "dropped_edges": list(snapshot.dropped_edges),
                    "ops": [op.to_dict() for op in state.pending_ops],
                },
                status=result.status,
                error=result.error,
                latency_ms=result.latency_ms,
                messages=result.messages,
            )
        return result

    async def fire_auditor_from_record(self, record: ReplayRecord) -> PhaseResult:
        """Replay one recorded auditor firing.

        Mirrors legacy ``replay_auditor_record``: composes auditor
        extensions from ``record.compose_kwargs`` (events / edges /
        phases / findings / continuation_notes / tools) and calls
        :func:`run_phase_standalone` with ``record.payload`` as the
        user message verbatim.
        """
        from ...schema import Finding

        if record.phase != "auditor":
            raise ValueError(f"expected auditor record, got phase={record.phase!r}")
        if self._cwd is None:
            raise ValueError(
                "HarnessRunner.fire_auditor_from_record requires cwd to be set at construction"
            )

        ck = record.compose_kwargs or {}
        s = self._auditor_settings
        extensions = compose_auditor_extensions(
            base_prompt=s.base_prompt or None,
            observability_config=s.observability_config,
            trajectory_snapshot=ck.get("trajectory_snapshot"),
            events=tuple(_coerce_schema_list(Event, ck.get("events") or [])),
            edges=tuple(_coerce_schema_list(Edge, ck.get("edges") or [])),
            phases=tuple(_coerce_schema_list(Phase, ck.get("phases") or [])),
            findings=_coerce_schema_list(Finding, ck.get("findings") or []),
            check_errors=dict(ck.get("check_errors") or {}),
            continuation_notes=list(ck.get("continuation_notes") or []),
            summary_threshold=s.summary_threshold,
            tools=s.tools or None,
        )
        return await run_phase_standalone(
            cwd=self._cwd,
            extensions=extensions,
            provider=self._provider_auditor,
            payload=record.payload or {},
            terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
            purpose="cognitive_audit_auditor_replay",
        )


def _coerce_schema_list(cls: Any, items: Any) -> list[Any]:
    """Best-effort dict-to-dataclass coercion for replay-record schema fields."""
    out: list[Any] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            out.append(cls.from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# Re-exported helpers — the adapter imports these so they stay in one place.
__all__ = [
    "AUDIT_REGISTRY_SERVICE_KEY",
    "FINALIZE_EXTRACTION_TOOL_NAME",
    "SUBMIT_VERDICT_TOOL_NAME",
    "AuditorChildResult",
    "AuditorOutputError",
    "AuditorSettings",
    "ChildRunner",
    "CumulativeAuditState",
    "ExtractorSettings",
    "ExtractorSpawnError",
    "HarnessRunner",
    "OpSink",
    "RawExtractorOutput",
    "RawVerdictOutput",
    "SidecarWriter",
    "StepResult",
    "_flatten_assistant_blocks",
    "_render_message_text",
    "_serialize_block",
    "_serialize_full_trajectory",
    "_serialize_message_for_extractor",
    "_window_is_non_trivial",
]
