"""One-shot reminder seeder for prefix-replay sessions.

Pairs with :mod:`llmharness.replay.cli` ``agent-from-reminder``: when a
session is branched from the end of turn t and resumed, this atom delivers
the recorded auditor reminder as the FIRST injection of the new session,
then unsubscribes itself so it never fires again.

Why one-shot? The whole point of prefix-replay is reproducing the
post-reminder behaviour with a different model / prompt / config — the
seed is the experimental stimulus, not a recurring nudge. Subsequent
reminders, if any, come from the still-live auditor running in the
branched session (configured at the CLI layer to keep observing but not
re-inject).

contract: single file, no atom-to-atom imports, no
``agentm.core.runtime.*`` import, no ``agentm.core._internal`` import.
"""

from __future__ import annotations

import time
from typing import Any

from agentm.core.abi import (
    DecideTurnActionEvent,
    ExtensionAPI,
    Inject,
    LoopAction,
    Stop,
    UserMessage,
    text_message,
)
from agentm.extensions import ExtensionManifest
from loguru import logger
from pydantic import BaseModel, Field

from llmharness.schema import REMINDER_DELIVERED

REMINDER_OPEN = "<system-reminder>\n"
REMINDER_CLOSE = "\n</system-reminder>"


def _build_reminder_message(text: str) -> UserMessage:
    return text_message(f"{REMINDER_OPEN}{text}{REMINDER_CLOSE}", timestamp=time.time())


class ReplayReminderSeedConfig(BaseModel):
    text: str = Field(
        min_length=1,
        description=(
            "Reminder text to inject on the next "
            "DecideTurnActionEvent. The preamble is "
            "prepended automatically; pass only the body."
        ),
    )


MANIFEST = ExtensionManifest(
    name="replay_reminder_seed",
    description=(
        "One-shot reminder injector for prefix-replay sessions. Drains "
        "exactly one configured reminder text on the first "
        "DecideTurnActionEvent after install (mirroring the live "
        "adapter's Inject path), persists a REMINDER_DELIVERED entry, "
        "and then unsubscribes. Used by ``llmharness-replay "
        "agent-from-reminder`` so a branched session resumed at the end "
        "of turn t sees the recorded auditor reminder exactly once at "
        "turn t+1 — same wire shape as the live audit loop would have "
        "produced."
    ),
    registers=("event:decide_turn_action",),
    config_schema=ReplayReminderSeedConfig,
    affects=("event:decide_turn_action",),
    requires=("operations",),
    api_version=1,
    tier=1,
)


def install(api: ExtensionAPI, config: ReplayReminderSeedConfig) -> None:
    """Wire the one-shot reminder seeder."""
    text: str = config.text

    fired: list[bool] = [False]
    unsubscribe: list[Any] = []

    def _on_decide(event: DecideTurnActionEvent) -> LoopAction | None:
        if fired[0]:
            return None
        default = event.observation.default_action
        if isinstance(default, Stop) and default.cause.final:
            logger.warning(
                f"replay_reminder_seed: first DecideTurnActionEvent has final-cause Stop ({type(default.cause).__name__}); reminder will not be delivered, leaving seed armed"
            )
            return None
        message = _build_reminder_message(text)
        try:
            api.session.append_entry(REMINDER_DELIVERED, {"text": text})
        except Exception:
            logger.exception("replay_reminder_seed: failed to persist reminder_delivered entry")
        fired[0] = True
        if unsubscribe:
            try:
                unsubscribe[0]()
            except Exception:
                logger.exception("replay_reminder_seed: unsubscribe callback raised; ignoring")
        return Inject(messages=[message])

    unsubscribe.append(api.on(DecideTurnActionEvent.CHANNEL, _on_decide))


__all__ = ["MANIFEST", "install"]
