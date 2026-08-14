# code-health: ignore-file[AM025] -- checklist YAML is untyped at the boundary
"""When an intervention fires: one predicate namespace, one budget.

An item's ``when`` block declares a boolean expression over predicates and,
optionally, a fact precondition. A predicate is just a name that is true of
this turn, and where it comes from is not the expression's business:

- the tagger emits one for any turn it has read, and those accumulate
- the run itself supplies the rest, computed from the tool stream at no cost:
  ``stopping``, ``reworked_own_work``

That mixture is the point. The tagger costs a model call and the stream ones
are free, so a gate like ``reworked_own_work AND NOT has_run_tests`` is mostly
free and could not be written at all while the two lived in separate
mechanisms -- which they did, one as an expression and one as a hardcoded
branch in the atom.

``checkpoint: stop`` survives as sugar. It is read at load time and composed
into the expression as ``stopping``, so there is one thing to evaluate rather
than a parameter beside it.

Firing is bounded by a single :class:`Budget` rather than a counter per kind of
intervention. Four of those had accumulated, no two aware of each other, and
their sum was what a run could actually be interrupted -- a number nothing
computed and nobody had chosen.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

from agentm.core.abi import text_message
from agentm.core.abi.events import Inject

#: Answers whether an item's precondition holds for the live session. The
#: engine stays free of a database that way: it asks, the atom knows how.
FactCheck = Callable[[str], bool]

#: The agent is wrapping up. Supplied by the run, not the tagger.
STOPPING = "stopping"

#: The agent has just edited something it had itself written earlier this run.
#: True for that turn only -- see ``Moment``.
REWORKED = "reworked_own_work"


@dataclass(slots=True, frozen=True)
class Moment:
    """One turn, as the gates see it.

    Predicates arrive here already merged: the tagger's, which accumulate over
    the session, and the run's own, which are true of this turn and false of
    the next. Nothing downstream needs to know which is which, and keeping the
    distinction out of the expression language is what lets an item combine
    them.
    """

    turn_index: int
    predicates: frozenset[str] = frozenset()

    @property
    def stopping(self) -> bool:
        return STOPPING in self.predicates


@dataclass(slots=True, frozen=True)
class Gate:
    """When an item may fire.

    Two halves answering different questions, and both must hold. ``trigger``
    asks whether the concern *resembles* this turn, over the predicate
    namespace. ``precondition`` asks whether the situation the concern
    presupposes has actually arrived, from the session's recorded actions.

    The second half exists because of a measurement: one item was seen firing
    eleven times, and six of those sessions had run no test at all, while the
    item's own words describe green narrowed test runs. It was aimed at the right
    concern and arrived before there was anything to be concerned about.

    An empty ``precondition`` means unconditional, which is the honest default.
    """

    trigger: str
    precondition: str = ""


@dataclass(slots=True, frozen=True)
class ChecklistItem:
    item_id: str
    dimension: str
    check: str
    advice: str
    #: Which action delivers this. Looked up in the atom's action registry, so
    #: an unknown name is a checklist that names something not installed --
    #: skipped with a warning rather than silently never firing, which is what
    #: the old ``!= "inject"`` filter did to every value but one.
    deliver: str
    #: How many times this item may fire in one run. One is right for a check
    #: the agent either heeds or does not; a gate on a recurring moment wants
    #: more, and had to be a separate mechanism to get it.
    max_fires: int
    gate: Gate


@dataclass(slots=True)
class Budget:
    """What the engine may spend interrupting one run.

    One pool, not one counter per mechanism, because the thing actually being
    rationed is the agent's attention and it does not care which mechanism took
    it.

    Per-item limits sit alongside the total: an item that has said its piece
    should not say it again, whatever room is left.
    """

    total: int
    spent: int = 0
    fires: dict[str, int] = field(default_factory=dict)

    def may_spend(self, label: str, limit: int = 1) -> bool:
        if self.spent >= self.total:
            return False
        return self.fires.get(label, 0) < max(limit, 1)

    def spend(self, label: str) -> None:
        self.spent += 1
        self.fires[label] = self.fires.get(label, 0) + 1

    def may_fire(self, item: ChecklistItem) -> bool:
        return self.may_spend(item.item_id, item.max_fires)

    def charge(self, item: ChecklistItem) -> None:
        self.spend(item.item_id)


def compose_trigger(trigger: str, checkpoint: str) -> str:
    """Fold ``checkpoint`` into the expression, so there is one thing to read.

    Kept as sugar rather than removed: it reads better than ``AND stopping`` on
    forty-odd items, and it was already the vocabulary of the checklist. What
    it must not stay is a second gate evaluated beside the first, which is how
    a moment that is neither `stop` nor a tag ended up with nowhere to live.
    """
    expr = trigger.strip() or "always"
    side = STOPPING if checkpoint == "stop" else f"NOT {STOPPING}"
    if expr == "always":
        return side
    return f"{side} AND ({expr})"


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
            max_fires=max(int(entry.get("max_fires", 1) or 1), 1),
            gate=Gate(
                trigger=compose_trigger(
                    str(when.get("trigger", "always")),
                    str(when.get("checkpoint", "stop")),
                ),
                precondition=" ".join(str(when.get("precondition", "")).split()),
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


def build_injection(message: str) -> Inject:
    """Wrap a rendered check as the loop action that delivers it."""
    return Inject(messages=(text_message(message, timestamp=time.time()),))


def _render(item: ChecklistItem, head: str, tail: str) -> str:
    lines = [head, "", item.check]
    if item.advice:
        lines.append(item.advice)
    lines += ["", tail]
    return "\n".join(lines)


def render_check(item: ChecklistItem) -> str:
    """Mid-work check, injected while the agent is still working."""
    return _render(item, _CHECK_HEAD, _CHECK_TAIL)


def render_stop_check(item: ChecklistItem) -> str:
    """Check raised at the moment the agent wraps up in prose."""
    return _render(item, _STOP_HEAD, _STOP_TAIL)


def render_finish_reminder() -> str:
    """Sent when the agent wraps up and no item matched, with review on.

    Carries no concern of its own: the reviewer supplies that. It exists because
    ``submit`` is how finishing happens and the agent has to be told it is there.
    Leaving that to a checklist match made review a coincidence.
    """
    return (
        "Looks like you are wrapping up. Nothing is merged or scored yet, so "
        "have a last look at whether the work does what was asked, and fix "
        "anything that needs it.\n\n"
        "When you are satisfied, call `submit` with a short summary of what you "
        "changed. A written summary on its own leaves the task open."
    )


# -- Predicate expression evaluation ------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*|AND|OR|NOT|\(|\)")


def evaluate_trigger(expr: str, predicates: frozenset[str]) -> bool:
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
            return tok in predicates
        return False

    return _parse_or()


# -- Trigger engine ------------------------------------------------------------


@dataclass(slots=True)
class TriggerEngine:
    """Selects the item to fire at a moment. Selection only -- what firing
    *does* belongs to the action the item names."""

    items: dict[str, ChecklistItem]
    budget: Budget

    def next_triggered(
        self,
        moment: Moment,
        *,
        fact_check: FactCheck | None = None,
    ) -> ChecklistItem | None:
        """The first matching item this moment can afford, charged to the budget.

        ``fact_check`` evaluates an item's precondition against the live session.
        Omitting it treats every precondition as unmet, so an item gated on facts
        stays silent rather than firing blind: a caller that cannot answer the
        question has not answered it yes.

        The budget is charged on selection rather than on delivery. An action
        that runs a reviewer and hears nothing back has still spent the turn and
        the wall-clock, and charging only for findings would let a quiet
        reviewer be called on every turn of the run.
        """
        for item in self._matching(moment, fact_check=fact_check):
            self.budget.charge(item)
            return item
        return None

    def _matching(
        self,
        moment: Moment,
        *,
        fact_check: FactCheck | None = None,
    ) -> Iterator[ChecklistItem]:
        for item in self.items.values():
            if not self.budget.may_fire(item):
                continue
            if not evaluate_trigger(item.gate.trigger, moment.predicates):
                continue
            if item.gate.precondition and (
                fact_check is None or not fact_check(item.gate.precondition)
            ):
                continue
            yield item
