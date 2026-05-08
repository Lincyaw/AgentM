"""Rule-based summarizer (P0): Turn list -> Event list.

Heuristics are intentionally crude. The interface — `summarize_turns(turns,
prior_events, next_event_id) -> list[Event]` — stays stable when an LLM-backed
implementation replaces the rules in P1.
"""

from __future__ import annotations

import re

from .schema import Event, EventKind, Turn, TurnRole

_HYPOTHESIS_MARKERS = (
    "可能是",
    "猜测",
    "假设",
    "怀疑",
    "i think",
    "i suspect",
    "hypothesis",
    "maybe",
    "might be",
)
_CONCLUSION_MARKERS = (
    "因此",
    "所以",
    "结论",
    "根因是",
    "root cause is",
    "therefore",
    "conclude",
    "in conclusion",
)
_REFLECTION_MARKERS = (
    "等等",
    "重新考虑",
    "实际上不是",
    "actually,",
    "wait,",
    "on second thought",
    "let me reconsider",
)
_SUMMARY_LIMIT = 60


def _truncate(text: str, limit: int = _SUMMARY_LIMIT) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _matches_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in markers)


def _classify_assistant(content: str) -> EventKind | None:
    """Classify an assistant message body. None means 'no event'.

    Order matters: hedge ("I think", "可能") wins over assertive markers, so
    that "I think the root cause is X" stays a hypothesis rather than being
    upgraded to a conclusion. Reflection wins over conclusion so a turn that
    revises a prior claim isn't read as a fresh conclusion.
    """
    if not content.strip():
        return None
    if _matches_any(content, _HYPOTHESIS_MARKERS):
        return EventKind.HYPOTHESIS
    if _matches_any(content, _REFLECTION_MARKERS):
        return EventKind.REFLECTION
    if _matches_any(content, _CONCLUSION_MARKERS):
        return EventKind.CONCLUSION
    return None


def summarize_turns(
    turns: list[Turn],
    prior_events: list[Event],
    next_event_id: int,
) -> list[Event]:
    """Convert a delta of turns into events.

    ``prior_events`` is consulted only to find the latest TASK event (used for
    refs) and the latest ACTION event (used to ref evidence to it). The function
    does not modify it.
    """
    has_task = any(e.kind is EventKind.TASK for e in prior_events)
    last_task_id = next((e.id for e in reversed(prior_events) if e.kind is EventKind.TASK), None)
    last_action_id = next(
        (e.id for e in reversed(prior_events) if e.kind is EventKind.ACTION), None
    )

    events: list[Event] = []
    eid = next_event_id

    def _emit(kind: EventKind, summary: str, refs: list[int], source: list[int]) -> None:
        nonlocal eid
        events.append(
            Event(
                id=eid,
                kind=kind,
                summary=_truncate(summary),
                refs=[r for r in refs if r is not None],
                source_turns=source,
            )
        )
        eid += 1

    for turn in turns:
        if turn.role is TurnRole.USER:
            if not has_task:
                _emit(EventKind.TASK, turn.content, refs=[], source=[turn.index])
                last_task_id = events[-1].id
                has_task = True
            else:
                _emit(
                    EventKind.REFLECTION,
                    f"user redirect: {turn.content}",
                    refs=[last_task_id] if last_task_id is not None else [],
                    source=[turn.index],
                )
        elif turn.role is TurnRole.TOOL:
            _emit(
                EventKind.EVIDENCE,
                f"{turn.tool_name or 'tool'} result: {turn.content}",
                refs=[last_action_id] if last_action_id is not None else [],
                source=[turn.index],
            )
        elif turn.role is TurnRole.ASSISTANT:
            if turn.tool_name:
                args_str = ""
                if turn.tool_args:
                    args_str = " " + " ".join(f"{k}={v}" for k, v in turn.tool_args.items())
                _emit(
                    EventKind.ACTION,
                    f"call {turn.tool_name}{args_str}",
                    refs=[last_task_id] if last_task_id is not None else [],
                    source=[turn.index],
                )
                last_action_id = events[-1].id
                continue
            kind = _classify_assistant(turn.content)
            if kind is None:
                continue
            _emit(
                kind,
                turn.content,
                refs=[last_task_id] if last_task_id is not None else [],
                source=[turn.index],
            )

    return events
