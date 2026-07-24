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
    """Evaluates every item's gate against the data plane.

    Signals are named plane queries (the registry below); the DSL only
    composes them. Inject items latch after one firing.
    """

    items: dict[str, ChecklistItem]
    min_narrow_runs: int = 2
    under_validation_min_mutations: int = 8
    under_validation_ratio: float = 0.25
    # None = task not classified yet: every item is armed (permissive).
    arm_flags: frozenset[str] | None = None
    _fired: set[str] = field(default_factory=set)

    def _armed(self, item: ChecklistItem) -> bool:
        if item.gate.arm == "always" or self.arm_flags is None:
            return True
        return item.gate.arm in self.arm_flags

    # -- signal registry: name -> plane query ---------------------------------

    def signal_true(self, plane: DataPlane, signal: str) -> bool:
        if signal == "always":
            return True
        if signal == "narrow_only":
            return bool(self._narrow_scopes(plane))
        if signal == "self_authored_only":
            return self._self_authored(plane)
        if signal == "under_validation":
            return self._under_validation(plane) is not None
        if signal == "unresolved_red":
            return bool(plane.scalar("SELECT COUNT(*) FROM v_unresolved_reds"))
        if signal == "red_any":
            return bool(plane.scalar("SELECT COUNT(*) FROM v_reds"))
        if signal == "green_close":
            last = plane.scalar(
                "SELECT exit_code FROM v_validations ORDER BY id DESC LIMIT 1"
            )
            return last == 0
        if signal in ("repeat_fail2", "repeat_fail3"):
            need = 2 if signal == "repeat_fail2" else 3
            return bool(
                plane.scalar(
                    "SELECT COUNT(*) FROM (SELECT head, COUNT(*) AS n "
                    "FROM v_reds GROUP BY head HAVING n >= ?)",
                    (need,),
                )
            )
        if signal == "test_code_alternate":
            return (
                self.signal_true(plane, "repeat_fail2")
                and bool(
                    plane.scalar("SELECT COUNT(*) FROM plane_edits WHERE is_test = 1")
                )
                and bool(
                    plane.scalar("SELECT COUNT(*) FROM plane_edits WHERE is_test = 0")
                )
            )
        if signal == "no_measurement":
            return not plane.scalar(
                "SELECT COUNT(*) FROM plane_run_tokens WHERE token IN "
                "('bench','benchmark','hyperfine','criterion','timeit',"
                "'time','flamegraph','profile','perf')"
            )
        if signal == "no_boundary_probe":
            return not plane.scalar(
                "SELECT COUNT(*) FROM plane_run_tokens WHERE token IN "
                "('restart','rotate','reboot','kill','sighup','resume',"
                "'reopen','relaunch')"
            )
        logger.warning("policy triggers: unknown signal {}", signal)
        return False

    def _narrow_scopes(self, plane: DataPlane) -> list[tuple[str, int]]:
        """Scopes where >= min narrowed validation runs exist and no
        selector-free run references the scope."""

        return [
            (str(scope), int(n))
            for scope, n in plane.query(
                "SELECT rs.scope, COUNT(DISTINCT rs.run_id) AS n "
                "FROM plane_run_scopes rs "
                "JOIN v_validations v ON v.id = rs.run_id "
                "WHERE EXISTS (SELECT 1 FROM plane_run_selectors sel "
                "              WHERE sel.run_id = rs.run_id) "
                "AND rs.scope NOT IN ("
                "    SELECT rs2.scope FROM plane_run_scopes rs2 "
                "    JOIN v_validations v2 ON v2.id = rs2.run_id "
                "    WHERE NOT EXISTS (SELECT 1 FROM plane_run_selectors s2 "
                "                      WHERE s2.run_id = rs2.run_id)) "
                "GROUP BY rs.scope HAVING n >= ?",
                (self.min_narrow_runs,),
            )
        ]

    def _self_authored(self, plane: DataPlane) -> bool:
        edited_tests = plane.scalar(
            "SELECT COUNT(*) FROM plane_edits WHERE is_test = 1"
        )
        greens = plane.scalar("SELECT COUNT(*) FROM v_greens")
        independent = plane.scalar("SELECT COUNT(*) FROM v_independent_greens")
        return bool(edited_tests) and bool(greens) and not independent

    def _under_validation(self, plane: DataPlane) -> tuple[int, int] | None:
        mutations = int(str(plane.scalar("SELECT COUNT(*) FROM plane_edits") or 0))
        validations = int(str(plane.scalar("SELECT COUNT(*) FROM v_validations") or 0))
        if mutations < self.under_validation_min_mutations:
            return None
        if validations >= mutations * self.under_validation_ratio:
            return None
        return mutations, validations

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
                facts=self._evidence(plane, item.gate.signal),
                suggestion="",
            )
        return None

    def _evidence(self, plane: DataPlane, signal: str) -> tuple[str, ...]:
        if signal == "narrow_only":
            facts = []
            for scope, n in self._narrow_scopes(plane)[:3]:
                rows = plane.query(
                    "SELECT DISTINCT v.raw FROM v_validations v "
                    "JOIN plane_run_scopes rs ON rs.run_id = v.id "
                    "WHERE rs.scope = ? ORDER BY v.id DESC LIMIT 2",
                    (scope,),
                )
                facts.append(f"scope `{scope}`: {n} narrowed runs")
                facts.extend(f"  `{_clip(str(raw))}`" for (raw,) in rows)
            return tuple(facts)
        if signal == "self_authored_only":
            stems = plane.query(
                "SELECT DISTINCT stem FROM plane_edits WHERE is_test = 1 LIMIT 4"
            )
            runs = plane.query("SELECT raw FROM v_greens ORDER BY id DESC LIMIT 2")
            return (
                "test files you edited: " + ", ".join(str(stem) for (stem,) in stems),
                *(f"green run: `{_clip(str(raw))}`" for (raw,) in runs),
            )
        if signal == "under_validation":
            counts = self._under_validation(plane)
            if counts is None:
                return ()
            runs = plane.query("SELECT raw FROM v_validations ORDER BY id DESC LIMIT 2")
            return (
                f"{counts[0]} file modifications vs {counts[1]} validation runs",
                *(f"validation run: `{_clip(str(raw))}`" for (raw,) in runs),
            )
        return ()

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
        for raw, exit_code in plane.query(
            "SELECT raw, exit_code FROM v_unresolved_reds ORDER BY id DESC LIMIT 3"
        ):
            facts.append(
                f"failed run never superseded: `{_clip(str(raw))}` (exit {exit_code})"
            )
        for head, n in plane.query(
            "SELECT head, COUNT(*) AS n FROM v_reds GROUP BY head "
            "HAVING n >= 2 ORDER BY n DESC LIMIT 3"
        ):
            facts.append(f"repeated failure x{n}: {head}")
        return tuple(facts)
