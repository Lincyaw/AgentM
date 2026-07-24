# code-health: ignore-file[AM025] -- checklist YAML is untyped at the boundary
"""Trigger layer: every checklist item carries a declared gate.

An item's ``when`` block is the DSL: it names a signal (a named query
against the data plane), the checkpoint it is evaluated at, and the
task-side arm flag. The gate decides WHEN an item becomes relevant; the
``deliver`` field decides what happens then:

- ``inject``  — the audited structural signals; fire once, render a message
  quoting the agent's own commands, inject immediately.
- ``critic``  — collected while gated-open at the stop decision and handed
  to the delivery layer (self-check injection or subagent review), which
  judges WHETHER the item is actually violated.
- ``offline`` — label-pipeline checks; never evaluated at runtime.

The gate vocabulary is deliberately small: signals are audited queries
over the plane schema; the DSL only composes them. Rates for every gate
come from ``python -m policy_engine replay`` over the recorded corpus.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

from .plane import DataPlane


@dataclass(slots=True, frozen=True)
class SignalDef:
    """One signal = one query over the data plane (defined in signals.yaml)."""

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
    signal: str  # named plane query from the signal registry
    checkpoint: str  # "continuous" | "stop"
    arm: str  # task flag name; "always" until the task classifier lands


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
    suggestion: str


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
                checkpoint=str(when.get("checkpoint", "stop")),
                arm=str(when.get("arm", "always")),
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
    if firing.suggestion:
        lines.append("")
        lines.append(firing.suggestion)
    if firing.item.advice:
        lines.append(firing.item.advice)
    return "\n".join(lines)


@dataclass(slots=True)
class TriggerEngine:
    """Generic gate evaluator: items and signals are data, this is mechanism.

    A signal is true when its query (with merged params) returns any row.
    Inject items latch after one firing and render evidence rows through
    the signal's format string.
    """

    items: dict[str, ChecklistItem]
    signals: dict[str, SignalDef]
    param_overrides: dict[str, float] = field(default_factory=dict)
    # None = task not classified yet: every item is armed (permissive).
    arm_flags: frozenset[str] | None = None
    _fired: set[str] = field(default_factory=set)

    def _armed(self, item: ChecklistItem) -> bool:
        if item.gate.arm == "always" or self.arm_flags is None:
            return True
        return item.gate.arm in self.arm_flags

    def signal_true(self, plane: DataPlane, name: str) -> bool:
        signal = self.signals.get(name)
        if signal is None:
            logger.warning("policy triggers: unknown signal {}", name)
            return False
        params = {**signal.params, **self.param_overrides}
        bound = {
            key: value for key, value in params.items() if f":{key}" in signal.query
        }
        return bool(plane.query(signal.query, bound))

    def evidence(self, plane: DataPlane, name: str) -> tuple[str, ...]:
        signal = self.signals.get(name)
        if signal is None or not signal.evidence_query:
            return ()
        rows = plane.query(signal.evidence_query)
        facts = []
        for row in rows:
            try:
                facts.append(signal.evidence_format.format(*row))
            except (IndexError, KeyError) as exc:
                logger.warning(
                    "policy triggers: bad evidence format for {}: {}", name, exc
                )
        return tuple(facts)

    # -- inject tier -----------------------------------------------------------

    def evaluate_inject(self, plane: DataPlane, *, stopping: bool) -> Firing | None:
        for item in self.items.values():
            if item.deliver != "inject" or item.item_id in self._fired:
                continue
            if not self._armed(item):
                continue
            if item.gate.checkpoint == "stop" and not stopping:
                continue
            if not self.signal_true(plane, item.gate.signal):
                continue
            self._fired.add(item.item_id)
            return Firing(
                item=item,
                facts=self.evidence(plane, item.gate.signal),
                suggestion="",
            )
        return None

    # -- critic tier -------------------------------------------------------------

    def open_critic_items(self, plane: DataPlane) -> tuple[ChecklistItem, ...]:
        return tuple(
            item
            for item in self.items.values()
            if item.deliver == "critic"
            and self._armed(item)
            and self.signal_true(plane, item.gate.signal)
        )

    def critic_evidence(self, plane: DataPlane) -> tuple[str, ...]:
        facts: list[str] = []
        for name in ("unresolved_red", "repeat_fail2"):
            facts.extend(self.evidence(plane, name)[:3])
        return tuple(facts)
