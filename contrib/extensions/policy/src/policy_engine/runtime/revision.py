"""When has the agent just reconsidered its own design?

The engine could until now act at two moments: on a tag match mid-run, and at
the exit. Neither is where work is decided. In the one measured run whose score
moved, the approach was chosen before turn 28 and the detail that decided the
score -- where in the transaction a lock is taken -- was settled at turn 57, by
an edit that moved code the same run had written six turns earlier. Every other
attempt at that task chose the same approach and put the lock in the wrong
place. What separated them was a revision, not a decision.

So the moment to look for is not the first edit. It is the agent editing what
it already built: extending a design is cheap and reversible, reworking one
means the first shape was wrong, and that is when a second opinion is worth
paying for.

Detected without a model call, from the tool stream alone, at whatever
granularity the call itself supports:

- an edit that says what it replaced, when that text overlaps what this run
  wrote to the same file earlier
- a write, or an edit that names only line numbers, to a file this run has
  already written to -- the coordinates have moved but the filename has not

Measured on three recorded attempts at one task, two of which failed:

    definition                     fires/run   caught turn 57
    every edit                        23-25    yes, among 23 others
    first edit per file                 5      no
    revision of own work (this)      7-12      yes
      + 5-turn cooldown                5-8     yes

The cooldown collapses bursts -- turns 51, 52, 54, 57, 59 are one sitting with
the same code -- and is what makes the count affordable when each firing costs
a blocking review. At 8 turns the cluster still fires but lands on 59 rather
than 57, and at 12 it is missed; 5 is the largest gap that kept the turn we
know mattered.

One positive case is one positive case. What this rules out is the cheaper
definitions: first-edit-per-file never fires at the moment in question, and
every-edit fires so often that it cannot fund a review.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from loguru import logger

#: Tools that change a file. A revision is only visible through these.
_EDIT_TOOLS = frozenset({"edit", "write"})

#: Lines shorter than this match across unrelated code -- a brace, an ``else``,
#: a bare ``return`` -- and would make almost any edit look like a revision.
_MIN_LINE = 12

#: Turns to wait after firing. Reworking a design takes several edits in a row
#: and they are one event, not five.
_COOLDOWN = 5


class ToolCall(Protocol):
    """What this needs from a tool call, and nothing more.

    A Protocol rather than the engine's own record type: the same detector runs
    over a live session and over rows read back from the trajectory, and those
    are different classes with these fields in common.
    """

    @property
    def name(self) -> str: ...

    @property
    def arguments(self) -> Mapping[str, object]: ...

    @property
    def is_error(self) -> bool: ...


def _significant_lines(text: str) -> frozenset[str]:
    return frozenset(
        line.strip() for line in text.splitlines() if len(line.strip()) >= _MIN_LINE
    )


def _is_test_path(path: str) -> bool:
    """Edits to tests are excluded.

    A test is where an agent is expected to iterate, so rewriting one is its
    normal working rhythm rather than a change of mind about the design.
    """
    lowered = path.lower()
    return "test" in lowered or "spec" in lowered


@dataclass(slots=True)
class RevisionDetector:
    """Fires on the turns where the agent reworked what it had built.

    Stateful across a run: it has to remember what this run wrote in order to
    recognise the same text coming back as something being replaced.
    """

    cooldown: int = _COOLDOWN
    #: Text this run has added, per file. Kept as line sets because an edit
    #: rarely replaces what was added verbatim -- it replaces a region that
    #: contains some of it.
    _added: dict[str, list[frozenset[str]]] = field(default_factory=dict)
    _last_fired: int = -1
    _fires: int = 0

    def observe(self, turn_index: int, calls: Iterable[ToolCall]) -> bool:
        """Record this turn's edits; answer whether it revised earlier work.

        Both halves happen here on purpose. An edit has to be remembered even
        when the turn does not fire, or a revision two turns later has nothing
        to recognise itself against.
        """
        revised = False
        for call in calls:
            if call.name not in _EDIT_TOOLS or call.is_error:
                continue
            path = str(call.arguments.get("path", ""))
            if not path or _is_test_path(path):
                continue
            known = self._added.get(path, ())

            if call.name == "write":
                # A write replaces the entire file, so if this run put anything
                # in that file already, the write has certainly replaced it.
                if known:
                    revised = True
                added = _significant_lines(str(call.arguments.get("content", "")))
            else:
                replaced = _significant_lines(str(call.arguments.get("old_string", "")))
                if replaced:
                    revised = revised or any(replaced & e for e in known)
                elif known:
                    # ``edit``'s other mode names line numbers rather than the
                    # text it replaces, and those coordinates have shifted with
                    # every edit since -- there is nothing to match against
                    # without tracking offsets through the whole run.
                    #
                    # So it falls back to the granularity that is known for
                    # certain: which file was touched. Same rule as ``write``,
                    # and weaker than text matching, since the edit may well be
                    # extending a part of the file the run never wrote.
                    #
                    # Measured over three recorded attempts, the fallback added
                    # three firings in one and none in the other two. What it
                    # does cost is ordering: in that session the first three
                    # firings moved from turns 30/68/97 to 30/38/44, so a
                    # confirmed rework lost its slot to two guesses. Earlier is
                    # the better place to spend a review, and one edit in
                    # thirteen being invisible is the worse failure.
                    revised = True
                added = _significant_lines(str(call.arguments.get("new_string", "")))

            if added:
                self._added.setdefault(path, []).append(added)

        if not revised:
            return False
        if self._last_fired >= 0 and turn_index - self._last_fired < self.cooldown:
            return False
        self._last_fired = turn_index
        self._fires += 1
        logger.info(
            "revision: turn {} reworked earlier work (#{})", turn_index, self._fires
        )
        return True

    @property
    def fires(self) -> int:
        return self._fires


__all__ = ["RevisionDetector", "ToolCall"]
