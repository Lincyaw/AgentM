# code-health: ignore-file[AM025] -- checklist YAML is untyped at the boundary
"""Trigger layer: checklist items with gates.

Each item's ``when`` block declares:
- ``signal``: a named SQL query over the data plane (structural signals)
- ``trigger``: a predicate expression over tagger annotations
- ``checkpoint``: when to evaluate (continuous / stop)

An item fires when BOTH signal and trigger are satisfied (or when either
is ``always``). Signal queries the structural data plane; trigger queries
the semantic annotations from the tagger.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml
from loguru import logger


@runtime_checkable
class QuerySource(Protocol):
    """Anything that can execute SQL and return rows."""

    def query(
        self,
        sql: str,
        params: tuple[object, ...] | Mapping[str, object] = (),
    ) -> list[tuple]: ...


@dataclass(slots=True, frozen=True)
class SignalDef:
    """One signal = one SQL query over the data plane."""

    name: str
    query: str
    params: dict[str, float]
    evidence_query: str
    evidence_format: str


def load_signals(path: Path) -> dict[str, SignalDef]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.error("policy triggers: cannot load {}: {}", path, exc)
        return {}
    signals: dict[str, SignalDef] = {}
    for name, entry in ((raw or {}).get("signals") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        evidence = entry.get("evidence")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        params = entry.get("params")
        params = dict(params) if isinstance(params, Mapping) else {}
        signals[str(name)] = SignalDef(
            name=str(name),
            query=str(entry.get("query", "SELECT 0 WHERE 0")),
            params={str(k): float(v) for k, v in params.items()},
            evidence_query=str(evidence.get("query", "")),
            evidence_format=str(evidence.get("format", "{0}")),
        )
    return signals


@dataclass(slots=True, frozen=True)
class Gate:
    signal: str  # named SQL query from signal registry
    trigger: str  # predicate expression over tagger annotations
    checkpoint: str  # "continuous" | "stop"


@dataclass(slots=True, frozen=True)
class ChecklistItem:
    item_id: str
    dimension: str
    check: str
    advice: str
    deliver: str  # "inject" | "critic" | "offline"
    gate: Gate


@dataclass(slots=True, frozen=True)
class Firing:
    item: ChecklistItem
    facts: tuple[str, ...]


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
            deliver=str(entry.get("deliver", "critic")),
            gate=Gate(
                signal=str(when.get("signal", "always")),
                trigger=str(when.get("trigger", "always")),
                checkpoint=str(when.get("checkpoint", "stop")),
            ),
        )
        if item.item_id:
            items[item.item_id] = item
    return items


def _clip(text: str, limit: int = 110) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_message(firing: Firing) -> str:
    lines = [f"Process check from the validation monitor ({firing.item.dimension}):"]
    lines.append(firing.item.check)
    if firing.facts:
        lines.append("")
        lines.append("Observed in this session:")
        lines.extend(f"- {fact}" for fact in firing.facts)
    if firing.item.advice:
        lines.append(firing.item.advice)
    return "\n".join(lines)


# -- Predicate expression evaluation ------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*|AND|OR|NOT|\(|\)")


def evaluate_trigger(expr: str, active_tags: frozenset[str]) -> bool:
    """Evaluate a boolean predicate expression against active tags.

    Supports: predicate names, AND, OR, NOT, parentheses.
    ``always`` is unconditionally true.
    """
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
    """Evaluates items against structural signals + tagger predicates."""

    items: dict[str, ChecklistItem]
    signals: dict[str, SignalDef]
    param_overrides: dict[str, float] = field(default_factory=dict)
    _fired: set[str] = field(default_factory=set)

    def signal_true(self, source: QuerySource, name: str) -> bool:
        if name == "always":
            return True
        signal = self.signals.get(name)
        if signal is None:
            logger.warning("policy triggers: unknown signal {}", name)
            return False
        params = {**signal.params, **self.param_overrides}
        return bool(source.query(signal.query, params))

    def evidence(self, source: QuerySource, name: str) -> tuple[str, ...]:
        signal = self.signals.get(name)
        if signal is None or not signal.evidence_query:
            return ()
        rows = source.query(signal.evidence_query)
        facts = []
        for row in rows:
            try:
                facts.append(signal.evidence_format.format(*row))
            except (IndexError, KeyError) as exc:
                logger.warning(
                    "policy triggers: bad evidence format for {}: {}", name, exc
                )
        return tuple(facts)

    def _gate_open(
        self,
        item: ChecklistItem,
        source: QuerySource,
        *,
        stopping: bool,
        active_tags: frozenset[str],
    ) -> bool:
        if item.gate.checkpoint == "stop" and not stopping:
            return False
        if not self.signal_true(source, item.gate.signal):
            return False
        if not evaluate_trigger(item.gate.trigger, active_tags):
            return False
        return True

    def evaluate_inject(
        self,
        source: QuerySource,
        *,
        stopping: bool,
        active_tags: frozenset[str] = frozenset(),
    ) -> Firing | None:
        for item in self.items.values():
            if item.deliver != "inject" or item.item_id in self._fired:
                continue
            if not self._gate_open(
                item, source, stopping=stopping, active_tags=active_tags
            ):
                continue
            self._fired.add(item.item_id)
            return Firing(
                item=item,
                facts=self.evidence(source, item.gate.signal),
            )
        return None

    def open_critic_items(
        self,
        source: QuerySource,
        *,
        active_tags: frozenset[str] = frozenset(),
    ) -> tuple[ChecklistItem, ...]:
        return tuple(
            item
            for item in self.items.values()
            if item.deliver == "critic"
            and self._gate_open(item, source, stopping=True, active_tags=active_tags)
        )

    def critic_evidence(self, source: QuerySource) -> tuple[str, ...]:
        facts: list[str] = []
        for name in (
            "dependency_blind_spot",
            "definition_not_traced",
            "non_convergent",
        ):
            facts.extend(self.evidence(source, name)[:3])
        return tuple(facts)
