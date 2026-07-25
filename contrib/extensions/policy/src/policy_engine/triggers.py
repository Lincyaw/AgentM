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
from collections.abc import Mapping
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


def render_message(items: list[ChecklistItem]) -> str:
    """Render a batch of checklist items into one inject message."""
    if len(items) == 1:
        item = items[0]
        lines = [f"Process check ({item.dimension}):"]
        lines.append(item.check)
        if item.advice:
            lines.append(item.advice)
        return "\n".join(lines)

    lines = [
        "Process check from the validation monitor — "
        "audit your process against each point below:"
    ]
    for i, item in enumerate(items, 1):
        lines.append(f"\n{i}. ({item.dimension}) {item.check}")
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

    def collect_triggered(
        self,
        *,
        stopping: bool,
        active_tags: frozenset[str] = frozenset(),
        max_items: int = 3,
    ) -> list[ChecklistItem]:
        """Collect all triggered items up to max_items. Batch mode."""
        triggered: list[ChecklistItem] = []
        for item in self.items.values():
            if item.deliver != "inject" or item.item_id in self._fired:
                continue
            if item.gate.checkpoint == "stop" and not stopping:
                continue
            if not evaluate_trigger(item.gate.trigger, active_tags):
                continue
            self._fired.add(item.item_id)
            triggered.append(item)
            if len(triggered) >= max_items:
                break
        return triggered
