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
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from agentm.core.abi.messages import AgentMessage, AssistantMessage, ToolResultMessage
from agentm.core.abi.session import SessionEntry

from ..replay.record import ReplayRecord, now_ns, write_record
from ..schema import Edge, Event, Phase, Reminder, Verdict
from . import entry_types as _et
from ._session_helpers import bind_extractor_state
from .auditor import (
    SUBMIT_VERDICT_TOOL_NAME,
    AuditorOutputError,
    RawVerdictOutput,
    compose_auditor_extensions,
)
from .extractor import (
    FINALIZE_EXTRACTION_TOOL_NAME,
    ExtractionState,
    RawExtractorOutput,
)
from .graph_fold import fold_graph
from .graph_ops import EdgeUpsert, GraphOp, NodeUpsert, parse_op
from .phase import merge_to_phases
from .registry import SERVICE_KEY as AUDIT_REGISTRY_SERVICE_KEY
from .registry import AuditCheckRegistry, CheckContext

_logger = logging.getLogger(__name__)

_DEFAULT_RECENT_VERDICTS: Final[int] = _et.RECENT_VERDICTS_FOR_AUDITOR


# --- settings dataclasses ---------------------------------------------------


@dataclass(frozen=True)
class ExtractorSettings:
    """Per-install knobs for assembling a per-firing extractor extension list."""

    extensions: list[tuple[str, dict[str, Any]]]
    compose_kwargs: dict[str, Any]


@dataclass(frozen=True)
class AuditorSettings:
    """Per-install knobs for assembling a per-firing auditor extension list.

    Mirrors the live adapter's ``_AuditorSettings`` exactly; field-for-field
    identical so behaviour matches byte-for-byte.
    """

    base_prompt: str
    cards_tools_config: dict[str, Any] | None
    observability_config: dict[str, Any] | None
    summary_threshold: int
    tools: tuple[str, ...]


# --- cumulative state -------------------------------------------------------


def _bool_safe_int(raw: Any) -> int | None:
    """Coerce ``int`` while rejecting ``bool`` (which is an ``int`` subclass)."""
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


@dataclass
class CumulativeAuditState:
    """Event-sourced graph state + auditor side-channel state.

    Single source of truth for the cumulative view; replaces both the
    live ``_scan_branch`` re-read-each-firing pattern and the (planned,
    P2) offline fold inside ``run_offline_auditor_over_control``.
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
    _cached_view: tuple[tuple[Event, ...], tuple[Edge, ...], tuple[Phase, ...]] | None = (
        None
    )
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
        self._cached_view = None
        self._cached_len = -1

    def absorb_auditor_verdict(
        self, verdict: dict[str, Any], *, is_silent: bool
    ) -> None:
        del is_silent  # currently informational only; both shapes append
        self.recent_verdicts.append(verdict)
        raw_notes = verdict.get("continuation_notes")
        if isinstance(raw_notes, list):
            self.last_continuation_notes = [n for n in raw_notes if isinstance(n, str)]

    @classmethod
    def fresh(cls) -> CumulativeAuditState:
        """Offline-mode seed: empty state."""
        return cls()

    @classmethod
    def hydrate_from_session_log(cls, branch: list[SessionEntry]) -> CumulativeAuditState:
        """Live-mode seed: walk session entries, populate ops + verdicts + phases.

        Mirrors the legacy ``_scan_branch`` accumulator so behaviour is
        unchanged on session restart. We collect:

        * ``AUDIT_GRAPH_OP`` entries → :func:`parse_op`-decoded :class:`GraphOp`.
        * Legacy ``AUDIT_EVENT`` → synthesized :class:`NodeUpsert` so a
          mixed pre-/post-event-sourcing session folds correctly.
        * Legacy ``AUDIT_EDGE`` → synthesized :class:`EdgeUpsert`.
        * ``AUDIT_PHASE`` → :class:`Phase` (kept in a parallel list; not
          part of the op log).
        * ``VERDICT`` → recent-verdicts deque (bounded).
        * ``EXTRACTOR_CURSOR`` → ``cursor_last_turn_index``.
        """
        ops: list[GraphOp] = []
        phases: list[Phase] = []
        verdicts_all: list[dict[str, Any]] = []
        cursor_last_turn_index = -1
        max_firing_id = -1

        for entry in branch:
            payload = entry.payload
            if not isinstance(payload, dict):
                continue
            if entry.type == _et.AUDIT_GRAPH_OP:
                try:
                    ops.append(parse_op(payload))
                except (KeyError, ValueError, TypeError):
                    continue
                fid_raw = payload.get("firing_id")
                fid = _bool_safe_int(fid_raw)
                if fid is not None and fid > max_firing_id:
                    max_firing_id = fid
            elif entry.type == _et.AUDIT_EVENT:
                try:
                    ev = Event.from_dict(payload)
                except (KeyError, ValueError, TypeError):
                    continue
                ops.append(
                    NodeUpsert(
                        id=ev.id,
                        kind=ev.kind.value,
                        summary=ev.summary,
                        source_turns=tuple(ev.source_turns),
                        external_refs=ev.external_refs,
                    )
                )
            elif entry.type == _et.AUDIT_EDGE:
                try:
                    ed = Edge.from_dict(payload)
                except (KeyError, ValueError, TypeError):
                    continue
                ops.append(
                    EdgeUpsert(
                        src=ed.src,
                        dst=ed.dst,
                        kind=ed.kind.value,
                        reason=ed.reason,
                        cited_entities=ed.cited_entities,
                        cited_quote=ed.cited_quote,
                        src_turns=ed.src_turns,
                        dst_turns=ed.dst_turns,
                    )
                )
            elif entry.type == _et.AUDIT_PHASE:
                try:
                    phases.append(Phase.from_dict(payload))
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

        state = cls(
            ops=ops,
            cursor_last_turn_index=cursor_last_turn_index,
            recent_verdicts=recent,
            last_continuation_notes=last_notes,
            firing_id_counter=max_firing_id + 1 if max_firing_id >= 0 else 0,
        )
        state._phases = phases
        return state


# --- protocols --------------------------------------------------------------


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
    ) -> tuple[Verdict | None, list[dict[str, Any]]]:
        """Run the auditor child; return ``(verdict | None, raw_blocks)``.

        ``None`` covers spawn / prompt / no-call / malformed paths; the
        implementation is responsible for recording failures via the
        runner's sink (the live impl uses ``api.session.append_entry``).
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

    # Legacy compatibility writes — preserved for P1 so the auditor /
    # viewer / aggregate pipelines that still read AUDIT_EVENT /
    # AUDIT_EDGE / AUDIT_PHASE keep working. Offline sinks no-op these.
    def append_legacy_event(self, ev: Event) -> None: ...

    def append_legacy_edge(self, ed: Edge) -> None: ...

    def append_legacy_phase(self, ph: Phase) -> None: ...


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
        root_session_id: str,
        compose_kwargs: dict[str, Any] | None,
        payload: dict[str, Any] | None,
        provider: tuple[str, dict[str, Any]] | None,
        output: dict[str, Any] | None,
        status: str,
        error: str | None = None,
        extras: dict[str, Any] | None = None,
        raw_assistant_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._path is None:
            return
        rec = ReplayRecord(
            phase=phase,  # type: ignore[arg-type]
            turn_index=turn_index,
            root_session_id=root_session_id,
            ts_ns=now_ns(),
            compose_kwargs=compose_kwargs or {},
            payload=payload or {},
            provider=[provider[0], provider[1]] if provider else None,
            output=output,
            status=status,  # type: ignore[arg-type]
            error=error,
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


# --- helpers ----------------------------------------------------------------


def _window_is_non_trivial(messages_slice: list[AgentMessage]) -> bool:
    """True iff the slice contains any AssistantMessage or ToolResultMessage."""
    return any(
        isinstance(msg, (AssistantMessage, ToolResultMessage)) for msg in messages_slice
    )


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


def _serialize_message_for_extractor(
    msg: AgentMessage, *, index: int
) -> dict[str, Any] | None:
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


@dataclass(frozen=True)
class _AuditorFiringResult:
    """Internal — returned from :meth:`HarnessRunner.fire_auditor_once`."""

    verdict: Verdict | None


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
        root_session_id: str,
        provider_extractor: tuple[str, dict[str, Any]] | None,
        provider_auditor: tuple[str, dict[str, Any]] | None,
        # In P1 the live audit-check registry lives on the parent
        # ExtensionAPI; passing it in keeps the runner independent of
        # ExtensionAPI so the (P2) offline driver can supply a synthetic
        # registry or ``None``.
        audit_registry: AuditCheckRegistry | None = None,
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
        self._root_session_id = root_session_id
        self._provider_extractor = provider_extractor
        self._provider_auditor = provider_auditor
        self._audit_registry = audit_registry
        # Track whether the most recent extractor firing held the cursor;
        # mirrors ``_drain_queue``'s ``_last_extractor_held_cursor`` so
        # the auditor doesn't fire on top of stale state.
        self._last_extractor_held_cursor: bool = False

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

        if extractor_due:
            result = await self.fire_extractor_once(messages)
            fired_extractor = True
            self._last_extractor_held_cursor = not result.ok

        if auditor_due:
            if self._last_extractor_held_cursor:
                _logger.debug(
                    "HarnessRunner: skipping auditor — preceding extractor "
                    "firing held the cursor"
                )
            else:
                audit_result = await self.fire_auditor_once(messages)
                fired_auditor = True
                if audit_result.verdict is not None and audit_result.verdict.surface_reminder:
                    text = audit_result.verdict.reminder_text
                    if text:
                        surfaced = Reminder(text=text)

        return StepResult(
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
        state.recent_edges_dict = {
            (ed.src, ed.dst, ed.kind.value): ed for ed in edges_cum
        }
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
            entry["source_turn_texts"] = [
                state.turn_texts.get(t, "") for t in e.source_turns
            ]
            recent_graph_payload.append(entry)

        payload = {
            "next_event_id": state.next_event_id,
            "new_turns": new_turn_window,
            "recent_graph": recent_graph_payload,
        }

        # Inject state into the per-firing extensions list. The base
        # list returned by ``compose_extractor_extensions`` is shared
        # across firings; ``bind_extractor_state`` returns a copy with
        # the ``state`` config knob set on the extractor-tools atom.
        firing_extensions = bind_extractor_state(
            self._extractor_settings.extensions,
            state=state,
        )

        replay_compose_kwargs: dict[str, Any] | None
        replay_extras: dict[str, Any] | None
        if self._sidecar is not None and self._sidecar.path is not None:
            replay_compose_kwargs = dict(self._extractor_settings.compose_kwargs)
            replay_extras = {
                "turn_texts": {str(k): v for k, v in state.turn_texts.items()},
            }
        else:
            replay_compose_kwargs = None
            replay_extras = None

        raw_assistant_messages: list[dict[str, Any]] = []

        def _record(
            status: str, output: dict[str, Any] | None, error: str | None = None
        ) -> None:
            if self._sidecar is None:
                return
            self._sidecar.record(
                phase="extractor",
                turn_index=window_hi_inclusive,
                root_session_id=self._root_session_id,
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
            return _ExtractorFiringResult(ok=False, cursor_advanced=False)

        if not terminator_called:
            self._sink.append_failure(
                _et.EXTRACTOR_NO_CALL,
                {
                    "reason": (
                        f"child returned without calling {FINALIZE_EXTRACTION_TOOL_NAME}"
                    ),
                    "turn_window": turn_window,
                },
            )
            _record("no_call", output=None)
            return _ExtractorFiringResult(ok=False, cursor_advanced=False)

        output = RawExtractorOutput.from_state(state)
        _record(
            "ok",
            output={
                "events": [e.to_dict() for e in output.events],
                "edges": [ed.to_dict() for ed in output.edges],
                "dropped_edges": list(output.dropped_edges),
            },
        )

        has_legacy_output = bool(output.events or output.edges or output.dropped_edges)
        has_ops = bool(state.pending_ops)

        if not has_legacy_output and not has_ops:
            if _window_is_non_trivial(window_messages):
                self._sink.append_failure(
                    _et.EXTRACTOR_EMPTY, {"turn_window": turn_window}
                )
                return _ExtractorFiringResult(ok=False, cursor_advanced=False)
            # Truly trivial window: still advance the cursor so we don't
            # re-extract the same prefix forever.
            self._sink.append_cursor(last_turn_index=window_hi_inclusive)
            self.cumulative.cursor_last_turn_index = window_hi_inclusive
            return _ExtractorFiringResult(ok=True, cursor_advanced=True)

        firing_id = self.cumulative.firing_id_counter
        for op_index, op in enumerate(state.pending_ops):
            self._sink.append_op(
                op,
                firing_id=firing_id,
                op_index=op_index,
                turn_window=list(turn_window),
            )

        # Legacy entries — the auditor / aggregate / viewer still read these.
        for ev in output.events:
            self._sink.append_legacy_event(ev)
        for ed in output.edges:
            self._sink.append_legacy_edge(ed)
        firing_phases = list(merge_to_phases(output.events))
        for ph in firing_phases:
            self._sink.append_legacy_phase(ph)

        if output.dropped_edges:
            self._sink.append_failure(
                _et.EXTRACTOR_PARTIAL,
                {
                    "dropped_edges": list(output.dropped_edges),
                    "turn_window": turn_window,
                },
            )

        self._sink.append_cursor(last_turn_index=window_hi_inclusive)

        # Mutate cumulative in-memory state. ``firing_id`` is bumped only
        # when the firing actually produced ops (matches the legacy
        # ``firing_counter`` invariant).
        if has_ops:
            self.cumulative.absorb_extractor_firing(
                firing_ops=list(state.pending_ops),
                firing_cursor=window_hi_inclusive,
                firing_id=firing_id,
                firing_phases=firing_phases,
            )
        else:
            # Legacy-only output (no maintainer ops). Still advance the
            # cursor + accumulate phases so subsequent firings see them.
            self.cumulative.cursor_last_turn_index = window_hi_inclusive
            self.cumulative._phases.extend(firing_phases)

        return _ExtractorFiringResult(ok=True, cursor_advanced=True)

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
                _logger.exception(
                    "audit-check registry run_all failed; using empty findings"
                )
                findings, check_errors = [], {}

        trajectory_snapshot = _serialize_full_trajectory(messages)
        continuation_notes = list(self.cumulative.last_continuation_notes)
        recent_verdicts = list(self.cumulative.recent_verdicts)

        firing_extensions = compose_auditor_extensions(
            base_prompt=self._auditor_settings.base_prompt,
            cards_tools_config=self._auditor_settings.cards_tools_config,
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

        replay_compose_kwargs: dict[str, Any] | None = None
        replay_payload: dict[str, Any] | None = None
        if self._sidecar is not None and self._sidecar.path is not None:
            replay_compose_kwargs = {
                "base_prompt": self._auditor_settings.base_prompt,
                "cards_tools_config": self._auditor_settings.cards_tools_config,
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
            replay_payload = {
                "graph": [e.to_dict() for e in events_tuple],
                "recent_verdicts": recent_verdicts,
                "continuation_notes_from_prior_firing": continuation_notes,
            }

        verdict, raw_blocks = await self._child.run_auditor(
            extensions=firing_extensions,
            provider=self._provider_auditor,
            graph_events=list(events_tuple),
            recent_verdicts=recent_verdicts,
            continuation_notes_from_prior_firing=continuation_notes,
        )
        turn_index = len(messages) - 1 if messages else -1

        if verdict is None:
            if self._sidecar is not None:
                self._sidecar.record(
                    phase="auditor",
                    turn_index=turn_index,
                    root_session_id=self._root_session_id,
                    compose_kwargs=replay_compose_kwargs,
                    payload=replay_payload,
                    provider=self._provider_auditor,
                    output=None,
                    status="no_call",
                    raw_assistant_messages=raw_blocks,
                )
            return _AuditorFiringResult(verdict=None)

        verdict_dict = verdict.to_dict()
        self._sink.append_verdict(verdict_dict)
        self.cumulative.absorb_auditor_verdict(
            verdict_dict, is_silent=not verdict.surface_reminder
        )
        if self._sidecar is not None:
            self._sidecar.record(
                phase="auditor",
                turn_index=turn_index,
                root_session_id=self._root_session_id,
                compose_kwargs=replay_compose_kwargs,
                payload=replay_payload,
                provider=self._provider_auditor,
                output=verdict_dict,
                status="ok",
                raw_assistant_messages=raw_blocks,
            )
        return _AuditorFiringResult(verdict=verdict)


# Re-exported helpers — the adapter imports these so they stay in one place.
__all__ = [
    "AUDIT_REGISTRY_SERVICE_KEY",
    "FINALIZE_EXTRACTION_TOOL_NAME",
    "SUBMIT_VERDICT_TOOL_NAME",
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
