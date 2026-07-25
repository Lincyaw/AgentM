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


# These arrive as user messages, so they should sound like the person who asked
# for the work — someone raising one doubt, not a form to fill in. Headed,
# bulleted audit blocks got answered in kind: a written verdict per point, and
# no change to the work. One doubt, plainly put, is harder to answer in prose
# than a rubric is.
#
# One item per message, deliberately: surfacing several at once lets the agent
# absorb the relevant one among plausible neighbours and move on.

_CHECK_HEAD = "One thing while you're in this —"

_CHECK_TAIL = "If you've already got that covered, carry on. If not, sort it out now."

_STOP_HEAD = "Hold on, before you call this done — I want to be sure about one thing."

# The exit has to be named. Without one, a reply to a check is indistinguishable
# from a fresh attempt to finish, so it draws the next check, and the next,
# until the budget runs out: the agent cannot end the session, only outlast the
# engine.
_STOP_TAIL = """\
Nothing has been merged or scored yet, so if something is off it is still
fixable. Take a look and fix it if it needs fixing. If you have already covered
it, or it does not apply here, call `submit` and tell me why — either way close
it out with `submit`, since a written answer on its own leaves the task open.

One thing though: if fixing it means replacing something you already produced,
show me the old result actually fails — run the check that indicts it. If you
cannot make it fail, keep it."""


def render_check(item: ChecklistItem) -> str:
    """Mid-work check, injected while the agent is still working."""
    lines = [_CHECK_HEAD, "", item.check]
    if item.advice:
        lines.append(item.advice)
    lines += ["", _CHECK_TAIL]
    return "\n".join(lines)


def render_stop_check(item: ChecklistItem) -> str:
    """Check raised at the moment the agent wraps up in prose."""
    lines = [_STOP_HEAD, "", item.check]
    if item.advice:
        lines.append(item.advice)
    lines += ["", _STOP_TAIL]
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
