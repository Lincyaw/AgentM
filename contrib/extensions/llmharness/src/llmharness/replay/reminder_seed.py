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

Message format is delegated to ``audit/_reminder_format.py`` so the
output is byte-identical to what the live adapter would have produced
for the same text — train/inference parity is load-bearing for the
distill loop.

§11 contract: single file, no atom-to-atom imports (the
``audit/_reminder_format`` module is a shared helper, not an atom — it
ships no ``MANIFEST``), no ``agentm.core.runtime.*`` import, no
``agentm.core._internal`` import.
"""

from __future__ import annotations

import logging
from typing import Any

from agentm.core.abi import DecideTurnActionEvent, Inject, LoopAction, Stop
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from ..audit._reminder_format import build_reminder_message
from ..audit.entry_types import REMINDER_DELIVERED

_logger = logging.getLogger(__name__)


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
    config_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Reminder text to inject on the next "
                    "DecideTurnActionEvent. The preamble defined in "
                    "audit/_reminder_format.REMINDER_PREAMBLE is "
                    "prepended automatically; pass only the body."
                ),
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    affects=("event:decide_turn_action",),
    requires=("operations_local",),
    api_version=1,
    tier=1,
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    """Wire the one-shot reminder seeder.

    Validation of ``text`` is performed here rather than relying purely
    on the JSON-schema layer: an empty / missing reminder is a
    programmer error from the CLI driver and should fail loudly.
    """
    text_raw = config.get("text")
    if not isinstance(text_raw, str) or not text_raw.strip():
        raise ValueError(
            "replay_reminder_seed: config['text'] must be a non-empty string"
        )
    text: str = text_raw

    # ``fired`` is a list-of-one to keep the closure mutable from inside
    # the handler without resorting to ``nonlocal`` on a primitive —
    # matches the pattern used elsewhere in the adapter for queue-like
    # state captured by an event handler closure.
    fired: list[bool] = [False]
    unsubscribe: list[Any] = []

    def _on_decide(event: DecideTurnActionEvent) -> LoopAction | None:
        if fired[0]:
            return None
        default = event.observation.default_action
        if isinstance(default, Stop) and default.cause.final:
            # Mirrors live adapter behaviour: a final-cause stop
            # (MaxTurnsExhausted / SignalAborted) ignores overrides.
            # Warn and leave ``fired`` False so a later non-final turn —
            # if one ever materialises in this branched session — still
            # gets the seed. In practice this branch is unreachable for
            # the prefix-replay flow (the loop wouldn't have stopped on
            # the very first decide call) but keeping the guard
            # consistent with the live path avoids surprise behaviour.
            _logger.warning(
                "replay_reminder_seed: first DecideTurnActionEvent has "
                "final-cause Stop (%s); reminder will not be delivered, "
                "leaving seed armed",
                type(default.cause).__name__,
            )
            return None
        message = build_reminder_message(text)
        try:
            api.session.append_entry(REMINDER_DELIVERED, {"text": text})
        except Exception:
            _logger.exception(
                "replay_reminder_seed: failed to persist reminder_delivered entry"
            )
        fired[0] = True
        if unsubscribe:
            try:
                unsubscribe[0]()
            except Exception:
                _logger.exception(
                    "replay_reminder_seed: unsubscribe callback raised; ignoring"
                )
        return Inject(messages=[message])

    unsubscribe.append(api.on(DecideTurnActionEvent.CHANNEL, _on_decide))


__all__ = ["MANIFEST", "install"]
