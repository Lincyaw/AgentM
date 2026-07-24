# code-health: ignore-file[AM025] -- checklist YAML is untyped at the boundary
"""Trigger layer: every checklist item carries a declared gate.

An item's ``when`` block is the DSL: it names a trajectory feature (from
``TrajectoryState.features()``), the checkpoint it is evaluated at, and the
task-side arm flag. The gate decides WHEN an item becomes relevant; the
``deliver`` field decides what happens then:

- ``inject``  — the audited structural signals; fire once, render a message
  quoting the agent's own commands, inject immediately.
- ``critic``  — collected while gated-open at the stop decision and handed
  to the delivery layer (self-check injection or subagent review), which
  judges WHETHER the item is actually violated.
- ``offline`` — label-pipeline checks; never evaluated at runtime.

The gate vocabulary is deliberately small: features are audited code in
``signals.py``; the DSL only composes them. Rates for every gate come from
``python -m policy_engine replay`` over the recorded corpus.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

from .signals import TrajectoryState


@dataclass(slots=True, frozen=True)
class Gate:
    signal: str  # feature name in TrajectoryState.features()
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
    """Evaluates every item's gate; inject items latch after one firing."""

    items: dict[str, ChecklistItem]
    min_narrow_runs: int = 2
    # None = task not classified yet: every item is armed (permissive).
    # Once the task classifier provides flags, arm conditions restrict.
    arm_flags: frozenset[str] | None = None
    _fired: set[str] = field(default_factory=set)

    def _armed(self, item: ChecklistItem) -> bool:
        if item.gate.arm == "always" or self.arm_flags is None:
            return True
        return item.gate.arm in self.arm_flags

    # -- inject tier -----------------------------------------------------------

    def evaluate_inject(
        self, state: TrajectoryState, *, stopping: bool
    ) -> Firing | None:
        features = state.features()
        for item in self.items.values():
            if item.deliver != "inject" or item.item_id in self._fired:
                continue
            if not self._armed(item):
                continue
            if item.gate.checkpoint == "stop" and not stopping:
                continue
            firing = self._evidence_firing(item, state, features)
            if firing is not None:
                self._fired.add(item.item_id)
                return firing
        return None

    def _evidence_firing(
        self,
        item: ChecklistItem,
        state: TrajectoryState,
        features: Mapping[str, bool | int],
    ) -> Firing | None:
        """Inject items render evidence via their signal's evaluator."""

        signal = item.gate.signal
        if signal == "narrow_only":
            groups = state.narrow_only_groups(min_runs=self.min_narrow_runs)
            if not groups:
                return None
            facts = [
                f"`{_clip(run.segment.raw)}` "
                f"(narrowing tokens: {', '.join(run.selectors[:4])})"
                for group in groups[:3]
                for run in group.runs[-2:]
            ]
            shortest = min(
                (run for group in groups for run in group.runs),
                key=lambda run: len(run.segment.ordered),
            )
            suggestion = (
                f"For example, re-run `{_clip(shortest.segment.raw)}` without "
                f"{', '.join(shortest.selectors[:3])}."
            )
            return Firing(item=item, facts=tuple(facts), suggestion=suggestion)
        if signal == "self_authored_only":
            authored = state.self_authored_only()
            if authored is None:
                return None
            facts = [
                f"test files you edited: {', '.join(authored.edited_test_stems[:4])}",
                *(
                    f"green run: `{_clip(segment.raw)}`"
                    for segment in authored.green_runs[-2:]
                ),
            ]
            return Firing(item=item, facts=tuple(facts), suggestion="")
        if signal == "under_validation":
            sparse = state.under_validation()
            if sparse is None:
                return None
            facts = [
                f"{sparse.mutation_count} file modifications vs "
                f"{sparse.validation_count} validation runs",
                *(
                    f"validation run: `{_clip(segment.raw)}`"
                    for segment in sparse.validations[-2:]
                ),
            ]
            return Firing(item=item, facts=tuple(facts), suggestion="")
        # Generic feature gate without a dedicated evidence renderer.
        if features.get(signal):
            return Firing(item=item, facts=(), suggestion="")
        return None

    # -- critic tier -------------------------------------------------------------

    def open_critic_items(self, state: TrajectoryState) -> tuple[ChecklistItem, ...]:
        """Critic items whose gate is open at the stop decision."""

        features = state.features()
        return tuple(
            item
            for item in self.items.values()
            if item.deliver == "critic"
            and self._armed(item)
            and bool(features.get(item.gate.signal))
        )

    def critic_evidence(self, state: TrajectoryState) -> tuple[str, ...]:
        """Noisy-signal digest handed to the critic at stop time."""

        facts: list[str] = []
        for record in state.unresolved_reds()[-3:]:
            facts.append(
                f"failed run never superseded: `{_clip(record.raw)}` "
                f"(exit {record.exit_code})"
            )
        for key, count in state.repeated_failures()[:3]:
            facts.append(f"repeated failure x{count}: {key}")
        return tuple(facts)
