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


# Response contract. The message must license an edit, not an answer: a bare
# question gets graded ("evidence_adequacy — Partial") and the work never
# reopens. Every inject therefore ends in a first-person commitment whose only
# two forms are a change or a reasoned refusal, and the dimension name is
# withheld so the agent judges the work rather than the label.
_CONTRACT = """\
Work this in order, over what you have actually done so far:

1. Observe: name the concrete thing that is wrong, weak, or missing, with a
   verbatim fact from your own trace — a command you ran and its output, a
   line you read, a result you got. If nothing is wrong, say exactly what you
   checked.
2. Decide: commit to exactly one of
   - `change: <one focused change>`
   - `no change needed, because <reason resting on the fact you just cited>`

If you commit to a change, make it now. A verdict on its own is not a response
to this note. A well-supported `no change needed` is a valid outcome — never
manufacture work to satisfy the check."""

_ADVISORY = "Process check (advisory — this does not block you)."

# The exam register comes from the agent believing the work is already
# delivered. Naming the answer as provisional is what reopens it.
_GATE = """\
Process check before you finish.

Your answer is PROVISIONAL. It has not been submitted or scored, and the work
is still open. Do not treat this as a post-hoc audit, and do not report a
status such as pass, partial, complete, or accept-with-caveat — those are not
available here."""

# A committed answer is cheap to defend in prose and expensive to refute with a
# check. Requiring the refutation is what converts a verdict into a tool call.
_REPLACEMENT = """\
If the change would replace a result you already produced, that result may be
replaced only by demonstrating its own failure, never by demonstrating that an
alternative looks better. Construct and run the check that would indict it —
recompute it from its own stated inputs, re-read the value at its cited source,
or re-check a constraint the task states — and show that check running. If you
cannot construct a failing check, keep the result, however compelling the
alternative reading looks."""


def render_message(item: ChecklistItem, *, stopping: bool = False) -> str:
    """Render one checklist item into an inject message.

    One item per inject, deliberately. Surfacing several at once lets the agent
    absorb the relevant one among plausible neighbours and move on — the same
    dilution that makes a full-taxonomy dump weaker than a ranked single call.
    """
    lines = [_GATE if stopping else _ADVISORY, "", item.check]
    if item.advice:
        lines.append(item.advice)
    lines += ["", _CONTRACT]
    if stopping:
        lines += ["", _REPLACEMENT]
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
