# code-health: ignore-file[AM025] -- checklist YAML is untyped at the boundary
"""Trigger layer: checklist items with predicate gates.

Each item's ``when`` block declares:
- ``trigger``: a boolean expression over predicates from vocabulary.yaml
- ``checkpoint``: when to evaluate (continuous / stop)

A predicate is true when the tagger has emitted that tag for any turn
in the session. Matching is deterministic boolean logic — no LLM call.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger


@dataclass(slots=True, frozen=True)
class Gate:
    trigger: str
    checkpoint: str


@dataclass(slots=True, frozen=True)
class ChecklistItem:
    item_id: str
    dimension: str
    check: str
    advice: str
    deliver: str
    gate: Gate


def load_items(path: Path) -> dict[str, ChecklistItem]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.error("policy triggers: cannot load {}: {}", path, exc)
        return {}
    items: dict[str, ChecklistItem] = {}
    for entry in (raw or {}).get("items", []):
        if not isinstance(entry, Mapping):
            continue
        when = entry.get("when")
        when = when if isinstance(when, Mapping) else {}
        item = ChecklistItem(
            item_id=str(entry.get("id", "")),
            dimension=str(entry.get("dimension", "")),
            check=" ".join(str(entry.get("check", "")).split()),
            advice=" ".join(str(entry.get("advice", "")).split()),
            deliver=str(entry.get("deliver", "inject")),
            gate=Gate(
                trigger=str(when.get("trigger", "always")),
                checkpoint=str(when.get("checkpoint", "stop")),
            ),
        )
        if item.item_id:
            items[item.item_id] = item
    return items


# One item per message, deliberately. Surfacing several at once lets the agent
# absorb the relevant one among plausible neighbours and move on.

# Mid-work check, delivered as a user message while the agent is still working.
# It must not read as a request for a verdict — a bare question gets answered in
# prose and the work never changes — so it ends by naming the two moves that
# count, and both of them are actions.
_CHECK_TAIL = """\
If this holds, keep going. If it does not, fix it now — make the edit or run
the check. When the work is complete and verified, call the `submit` tool.
Do not reply to this note with an audit."""

# Submit rejection, delivered as the submit tool's own result. The agent is
# mid-tool-call here, so the register is already "do something"; what it needs
# is the way back out.
_REJECTION_HEAD = "Not submitted. One process check is outstanding."

_REJECTION_TAIL = """\
Do one of these, then call `submit` again:
- fix what the check names, and verify the fix;
- or, if the check does not apply to this task, call `submit` again and say why
  in the summary.

If the change would replace a result you already produced, replace it only by
demonstrating its own failure — run the check that indicts it and show that
check running. If you cannot construct a failing check, keep the result."""


def render_check(item: ChecklistItem) -> str:
    """Mid-work check, injected as a user message."""
    lines = [item.check]
    if item.advice:
        lines.append(item.advice)
    lines += ["", _CHECK_TAIL]
    return "\n".join(lines)


def render_rejection(item: ChecklistItem) -> str:
    """Stop-checkpoint check, returned as the submit tool's result."""
    lines = [_REJECTION_HEAD, "", item.check]
    if item.advice:
        lines.append(item.advice)
    lines += ["", _REJECTION_TAIL]
    return "\n".join(lines)


# -- Predicate expression evaluation ------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*|AND|OR|NOT|\(|\)")


def evaluate_trigger(expr: str, active_tags: frozenset[str]) -> bool:
    if expr == "always" or not expr.strip():
        return True
    tokens = _TOKEN_RE.findall(expr)
    if not tokens:
        return True
    pos = 0

    def _peek() -> str:
        return tokens[pos] if pos < len(tokens) else ""

    def _advance() -> str:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def _parse_or() -> bool:
        result = _parse_and()
        while _peek() == "OR":
            _advance()
            result = _parse_and() or result
        return result

    def _parse_and() -> bool:
        result = _parse_not()
        while _peek() == "AND":
            _advance()
            result = _parse_not() and result
        return result

    def _parse_not() -> bool:
        if _peek() == "NOT":
            _advance()
            return not _parse_atom()
        return _parse_atom()

    def _parse_atom() -> bool:
        tok = _peek()
        if tok == "(":
            _advance()
            result = _parse_or()
            if _peek() == ")":
                _advance()
            return result
        if tok:
            _advance()
            return tok in active_tags
        return False

    return _parse_or()


# -- Trigger engine ------------------------------------------------------------


@dataclass(slots=True)
class TriggerEngine:
    """Evaluates checklist items against tagger predicates."""

    items: dict[str, ChecklistItem]
    _fired: set[str] = field(default_factory=set)

    def next_triggered(
        self, *, stopping: bool, active_tags: frozenset[str] = frozenset()
    ) -> ChecklistItem | None:
        """The single highest-priority matching item, marked fired."""
        for item in self._matching(stopping=stopping, active_tags=active_tags):
            self._fired.add(item.item_id)
            return item
        return None

    def would_trigger(
        self, *, stopping: bool, active_tags: frozenset[str] = frozenset()
    ) -> bool:
        """Whether any item matches, without consuming it.

        Used when an inject is suppressed: the items must stay unfired so they
        can still land once the agent has done real work.
        """
        return any(self._matching(stopping=stopping, active_tags=active_tags))

    def _matching(
        self, *, stopping: bool, active_tags: frozenset[str]
    ) -> Iterator[ChecklistItem]:
        for item in self.items.values():
            if item.deliver != "inject" or item.item_id in self._fired:
                continue
            if item.gate.checkpoint == "stop" and not stopping:
                continue
            if not evaluate_trigger(item.gate.trigger, active_tags):
                continue
            yield item
