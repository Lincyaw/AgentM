"""What firing does. One interface, so adding a kind is adding an implementation.

``ChecklistItem.deliver`` names an action and the atom looks it up in a
registry. That field already existed and was already parsed, but the engine
filtered on ``deliver != "inject"`` and every one of the forty-two items said
``inject`` -- so it read as a capability while being a branch that was always
taken. The first genuine second kind, running a reviewer, went in as a
hardcoded arm of the decide handler instead of as a value here, which is the
shape this module exists to stop.

An action returns the ``LoopAction`` that carries it to the agent, or ``None``
when it has nothing to say. ``None`` is not failure: a reviewer that looked and
found nothing is the common case and must cost the turn without costing the
run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from agentm.core.abi.events import LoopAction
from policy_engine.runtime.critic import Critic
from policy_engine.runtime.triggers import (
    ChecklistItem,
    Moment,
    build_injection,
    render_check,
    render_stop_check,
)


class Action(Protocol):
    """Delivers one fired item."""

    async def deliver(
        self, item: ChecklistItem, moment: Moment
    ) -> LoopAction | None: ...


@dataclass(slots=True)
class InjectAction:
    """Put the item's own words in front of the agent.

    The wording differs at the exit -- a check raised while the agent is
    wrapping up has to name ``submit``, or a reply to it is indistinguishable
    from a fresh attempt to finish. That is a property of the moment, not of
    the item, which is why the item does not choose between them.
    """

    async def deliver(self, item: ChecklistItem, moment: Moment) -> LoopAction | None:
        text = render_stop_check(item) if moment.stopping else render_check(item)
        return build_injection(text)


@dataclass(slots=True)
class ReviewAction:
    """Stop and have the critic look, then hand back what it found.

    Blocking, and that is the cost being budgeted: the agent waits out a full
    review. It buys the one thing injected text cannot -- the critic runs
    things, and in the only measured run whose score moved, what changed the
    agent's mind was a real error from a real execution rather than a prompt.

    ``prompt_for`` is supplied by the atom because building it needs the run so
    far, which is the atom's to hold. This module stays free of session state.
    """

    reviewer: Critic
    prompt_for: Callable[[Moment], str]

    async def deliver(self, item: ChecklistItem, moment: Moment) -> LoopAction | None:
        verdict = await self.reviewer.review(self.prompt_for(moment))
        if not verdict.criterion_settled:
            logger.info(
                "policy_engine: {} could not settle the criterion at turn {}",
                item.item_id,
                moment.turn_index,
            )
            return build_injection(verdict.as_ambiguity_message())
        if verdict.accepted:
            logger.info(
                "policy_engine: {} reviewed turn {}, nothing found",
                item.item_id,
                moment.turn_index,
            )
            return None
        logger.info(
            "policy_engine: {} broke turn {}: {}",
            item.item_id,
            moment.turn_index,
            verdict.finding[:120],
        )
        # At the exit the agent is holding a submission and the way forward is
        # to resubmit; mid-task there is nothing to resubmit and saying so
        # invites a reply, which ends the run.
        message = (
            verdict.as_message() if moment.stopping else verdict.as_revision_message()
        )
        return build_injection(message)


__all__ = ["Action", "InjectAction", "ReviewAction"]
