"""Policy engine evaluator — event dispatch, predicate eval, effect firing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from .effects import build_diagnostic, resolve_effects
from .query import (
    EffectLogQuery,
    FileStateQuery,
    TableQuery,
    diff,
    group,
    lookup,
    ratio,
    sequence,
    streak,
    trend,
)
from .state import PolicyState, fingerprint
from .types import (
    EMPTY,
    EffectRecord,
    EffectSpec,
    EvidenceList,
    RuleInstance,
    ToolArgs,
    _EmptyType,
)

if TYPE_CHECKING:
    from .ifc import IFCEngine


# ---------------------------------------------------------------------------
# Field name constants — so {tool: "read"} works in eval namespace
# ---------------------------------------------------------------------------

_FIELD_NAMES = (
    "tool", "path", "cmd", "exit_code", "error", "error_fingerprint",
    "error_category", "args_hash", "turn", "rule_id", "effect",
    "mode", "duration_ms", "is_repeat", "repeat_count", "result_length",
)


# ---------------------------------------------------------------------------
# Event proxy — safe attribute access on event data
# ---------------------------------------------------------------------------


class EventProxy:
    """Wraps raw event data for safe access in rule expressions."""

    __slots__ = ("tool_name", "args", "result", "taint")

    def __init__(
        self,
        tool_name: str = "",
        args: ToolArgs | None = None,
        result: dict[str, str | None] | None = None,
        taint: set[str] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.args = args or {}
        self.result = result or {}
        self.taint = taint or set()


# ---------------------------------------------------------------------------
# Session state proxy
# ---------------------------------------------------------------------------


class SessionProxy:
    """Exposes session metadata to rule expressions."""

    __slots__ = ("_state",)

    def __init__(self, state: PolicyState) -> None:
        self._state = state

    @property
    def turn_count(self) -> int:
        return self._state.turn_count

    @property
    def id(self) -> str:
        return ""


# ---------------------------------------------------------------------------
# Labels proxy
# ---------------------------------------------------------------------------


class _LabelsProxy:
    """Allows `labels.secret in event.taint` syntax."""

    __slots__ = ("_taint",)

    def __init__(self, taint: set[str]) -> None:
        self._taint = taint

    def __getattr__(self, name: str) -> str:
        return name

    def __contains__(self, item: str) -> bool:
        return item in self._taint


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class PolicyEvaluator:
    """Dispatches events to rules, evaluates predicates, fires effects."""

    def __init__(self, rules: list[RuleInstance], state: PolicyState) -> None:
        self._rules = rules
        self._state = state
        self._entity_evidence_fn: Callable[[str], EvidenceList | _EmptyType] | None = None
        self._ifc_engine: IFCEngine | None = None
        self._cross_session_fn: Callable[..., int] | None = None

        self._tool_index: dict[str, list[RuleInstance]] = defaultdict(list)
        self._wildcard: dict[str, list[RuleInstance]] = defaultdict(list)

        self._build_indices()

    def set_entity_evidence_fn(self, fn: Callable[[str], EvidenceList | _EmptyType]) -> None:
        self._entity_evidence_fn = fn

    def set_ifc_engine(self, engine: IFCEngine) -> None:
        self._ifc_engine = engine

    def set_cross_session_fn(self, fn: Callable[..., int]) -> None:
        self._cross_session_fn = fn

    def _build_indices(self) -> None:
        self._tool_index.clear()
        self._wildcard.clear()

        for rule in self._rules:
            if rule.guard.tool_names:
                for tn in rule.guard.tool_names:
                    self._tool_index[f"{rule.channel}:{tn}"].append(rule)
            else:
                self._wildcard[rule.channel].append(rule)

    def evaluate(
        self,
        channel: str,
        tool_name: str = "",
        args: ToolArgs | None = None,
        result: dict[str, str | None] | None = None,
    ) -> list[tuple[str, str, str]]:
        """Evaluate all matching rules for an event.

        Returns list of (rule_id, effect_type, diagnostic_message) for
        rules that fired AND are in enforce mode.
        """
        taint: set[str] = set()
        if self._ifc_engine and args:
            taint = self._ifc_engine.check_event_taint(args)

        event = EventProxy(
            tool_name=tool_name,
            args=args,
            result=result,
            taint=taint,
        )

        # Candidate dispatch: indexed + wildcard
        candidates = self._tool_index.get(f"{channel}:{tool_name}", [])
        wildcards = self._wildcard.get(channel, [])

        # Build namespace ONCE for the entire event (efficiency fix)
        namespace = self._build_namespace(event)

        fired: list[tuple[str, EffectSpec, str]] = []
        turn = self._state.turn_count

        for rule in (*candidates, *wildcards):
            if rule.guard.reject(tool_name, args or {}):
                continue

            if rule.cooldown_turns > 0 and rule.last_fired_turn >= 0:
                if (turn - rule.last_fired_turn) < rule.cooldown_turns:
                    continue

            if rule.predicate is not None:
                try:
                    if not eval(rule.predicate, namespace):  # noqa: S307
                        continue
                except Exception as e:
                    logger.debug("rule '{}' predicate error: {}", rule.rule_id, e)
                    continue

            rule.last_fired_turn = turn

            ctx_value: object = None
            if rule.escalate_context_expr is not None:
                try:
                    ctx_value = eval(rule.escalate_context_expr, namespace)  # noqa: S307
                except Exception as exc:
                    logger.debug(
                        "rule '{}' context expression error: {}",
                        rule.rule_id,
                        exc,
                    )
                    ctx_value = None

            diagnostic = build_diagnostic(
                rule.rule_id,
                rule.effect,
                args or {},
                escalate_context_value=ctx_value,
            )

            self._state.effect_log.append(EffectRecord(
                rule_id=rule.rule_id,
                mode=rule.mode,
                channel=channel,
                effect=rule.effect.effect,
                reason=diagnostic,
                turn=turn,
            ))

            if rule.mode == "enforce":
                fired.append((rule.rule_id, rule.effect, diagnostic))

        if not fired:
            return []

        resolved = resolve_effects(fired)
        return [(rid, eff.effect, msg) for rid, eff, msg in resolved]

    def _build_namespace(self, event: EventProxy) -> dict[str, object]:
        """Build eval namespace once per event. Includes __builtins__={}."""
        state = self._state
        tool_log_q = TableQuery(
            state.tool_log.entries,
            cross_session_fn=self._cross_session_fn,
        )
        file_state_q = FileStateQuery(state.file_state)
        effect_log_q = EffectLogQuery(state.effect_log.entries)

        tables: dict[str, object] = {
            "context_state": state.context_state,
            "session_tree": state.session_tree,
            "turn_summary": state.turn_summary,
        }

        ns: dict[str, object] = {
            "__builtins__": {},
            "event": event,
            "session": SessionProxy(state),
            "labels": _LabelsProxy(event.taint),
            "tool_log": tool_log_q,
            "file_state": file_state_q,
            "effect_log": effect_log_q,
            "streak": lambda where, last=None: streak(
                state.tool_log.entries, where, last
            ),
            "trend": lambda field, where=None, last=None: trend(
                state.tool_log.entries, field, where, last
            ),
            "ratio": ratio,
            "sequence": lambda steps, last=None: sequence(
                state.tool_log.entries, steps, last
            ),
            "group": lambda field, where=None, last=None, top=10: group(
                state.tool_log.entries, field, where, last, top
            ),
            "diff": diff,
            "lookup": lambda table, key, field: lookup(tables, table, key, field),
            "entity_evidence": (
                self._entity_evidence_fn
                if self._entity_evidence_fn
                else lambda name: EMPTY
            ),
            "fingerprint": fingerprint,
            "hash": lambda x: hash(str(x)),
            "max": max,
            "min": min,
            "abs": abs,
            "len": len,
            "any": any,
            "all": all,
            "sorted": sorted,
            "True": True,
            "False": False,
            "None": None,
            "EMPTY": EMPTY,
        }
        ns.update({k: k for k in _FIELD_NAMES})
        return ns
