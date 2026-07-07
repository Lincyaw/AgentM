"""Cumulative audit state across auditor firings."""

from __future__ import annotations

import collections
import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from agentm.core.abi import SessionEntry

from . import schema as _et

_DEFAULT_RECENT_VERDICTS: Final[int] = _et.RECENT_VERDICTS_FOR_AUDITOR


def _bool_safe_int(raw: Any) -> int | None:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


def _reminder_type_key(verdict: dict[str, Any]) -> str:
    """Extract a coarse key from a verdict to group same-type reminders.

    Uses the first continuation note (trimmed to 120 chars) as a fingerprint.
    Different issues produce different notes; the same recurring issue
    produces similar notes.
    """
    notes = verdict.get("continuation_notes")
    if isinstance(notes, list) and notes:
        first = str(notes[0])[:120].strip().lower()
        return first
    text = str(verdict.get("reminder_text", ""))[:120].strip().lower()
    return text


@dataclass(slots=True)
class CumulativeAuditState:
    """Auditor side-channel state across firings."""

    cursor_last_turn_index: int = -1
    recent_verdicts: collections.deque[dict[str, Any]] = field(
        default_factory=lambda: collections.deque(maxlen=_DEFAULT_RECENT_VERDICTS)
    )
    last_continuation_notes: list[str] = field(default_factory=list)
    firing_id_counter: int = 0
    consecutive_reminders: int = 0
    _last_reminder_key: str = ""

    def absorb_auditor_verdict(self, verdict: dict[str, Any]) -> None:
        self.recent_verdicts.append(verdict)
        raw_notes = verdict.get("continuation_notes")
        if isinstance(raw_notes, list):
            self.last_continuation_notes = [n for n in raw_notes if isinstance(n, str)]
        if verdict.get("surface_reminder"):
            reminder_key = _reminder_type_key(verdict)
            if reminder_key == self._last_reminder_key:
                self.consecutive_reminders += 1
            else:
                self.consecutive_reminders = 1
                self._last_reminder_key = reminder_key
        else:
            self.consecutive_reminders = 0
            self._last_reminder_key = ""

    @classmethod
    def fresh(cls) -> CumulativeAuditState:
        return cls()

    def snapshot(self) -> CumulativeAuditState:
        return copy.deepcopy(self)

    @classmethod
    def hydrate_from_session_log(cls, branch: Sequence[SessionEntry]) -> CumulativeAuditState:
        verdicts_all: list[dict[str, Any]] = []
        cursor_last_turn_index = -1
        for entry in branch:
            payload = entry.payload
            if not isinstance(payload, dict):
                continue
            if entry.type == _et.VERDICT:
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
            cursor_last_turn_index=cursor_last_turn_index,
            recent_verdicts=recent,
            last_continuation_notes=last_notes,
            firing_id_counter=0,
        )
