"""Event-sourced cumulative audit state across extractor/auditor firings."""

from __future__ import annotations

import collections
import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from agentm.core.abi import SessionEntry

from . import schema as _et
from .agents.extractor.index_store import IndexOp, fold_index, parse_op
from .schema import Edge, Event, Phase

_DEFAULT_RECENT_VERDICTS: Final[int] = _et.RECENT_VERDICTS_FOR_AUDITOR


def _bool_safe_int(raw: Any) -> int | None:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


@dataclass
class CumulativeAuditState:
    """Event-sourced context-index state + auditor side-channel state across firings."""

    ops: list[IndexOp] = field(default_factory=list)
    cursor_last_turn_index: int = -1
    recent_verdicts: collections.deque[dict[str, Any]] = field(
        default_factory=lambda: collections.deque(maxlen=_DEFAULT_RECENT_VERDICTS)
    )
    last_continuation_notes: list[str] = field(default_factory=list)
    firing_id_counter: int = 0
    _cached_len: int = -1
    _cached_view: tuple[tuple[Event, ...], tuple[Edge, ...], tuple[Phase, ...]] | None = None
    _phases: list[Phase] = field(default_factory=list)

    def index_view(self) -> tuple[tuple[Event, ...], tuple[Edge, ...], tuple[Phase, ...]]:
        if self._cached_view is not None and self._cached_len == len(self.ops):
            return self._cached_view
        folded = fold_index(self.ops)
        events = tuple(folded.records_list())
        edges = tuple(folded.links_list())
        phases = tuple(self._phases)
        self._cached_view = (events, edges, phases)
        self._cached_len = len(self.ops)
        return self._cached_view

    def next_event_id(self) -> int:
        events, _edges, _phases = self.index_view()
        return max((e.id for e in events), default=0) + 1

    def _invalidate_cache(self) -> None:
        self._cached_view = None
        self._cached_len = -1

    def absorb_extractor_firing(
        self,
        *,
        firing_ops: Sequence[IndexOp],
        firing_cursor: int,
        firing_id: int,
        firing_phases: Sequence[Phase] = (),
    ) -> None:
        self.ops.extend(firing_ops)
        self.cursor_last_turn_index = firing_cursor
        self._phases.extend(firing_phases)
        if firing_id >= self.firing_id_counter:
            self.firing_id_counter = firing_id + 1
        self._invalidate_cache()

    def absorb_auditor_verdict(self, verdict: dict[str, Any]) -> None:
        self.recent_verdicts.append(verdict)
        raw_notes = verdict.get("continuation_notes")
        if isinstance(raw_notes, list):
            self.last_continuation_notes = [n for n in raw_notes if isinstance(n, str)]

    @classmethod
    def fresh(cls) -> CumulativeAuditState:
        return cls()

    def snapshot(self) -> CumulativeAuditState:
        return copy.deepcopy(self)

    @classmethod
    def hydrate_from_session_log(cls, branch: list[SessionEntry]) -> CumulativeAuditState:
        ops: list[IndexOp] = []
        verdicts_all: list[dict[str, Any]] = []
        cursor_last_turn_index = -1
        for entry in branch:
            payload = entry.payload
            if not isinstance(payload, dict):
                continue
            if entry.type == _et.AUDIT_INDEX_OP:
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
        return cls(
            ops=ops,
            cursor_last_turn_index=cursor_last_turn_index,
            recent_verdicts=recent,
            last_continuation_notes=last_notes,
            firing_id_counter=0,
        )
