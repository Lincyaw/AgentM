"""Shared reminder-message format.

Both the live cognitive-audit adapter (``llmharness.adapters.agentm``) and
the offline prefix-replay seed atom (``llmharness.replay.reminder_seed``)
build the user-message that carries an auditor reminder into the main
agent's turn loop. The format MUST stay identical between the two so that
a model that learned to react to the live shape behaves the same way when
the seed atom replays a recorded reminder at the head of a branched
session.

Single source of truth: :data:`REMINDER_PREAMBLE` + :func:`build_reminder_message`.
"""

from __future__ import annotations

from agentm.core.abi.messages import UserMessage, text_message

# Prefix every reminder carries. Mirrored byte-for-byte in
# ``adapters/agentm.py::_REMINDER_PREAMBLE`` (which re-exports this constant
# under its own name for backwards-compat with the existing reminder-injector
# tests). Do not localise / vary.
REMINDER_PREAMBLE = (
    "[harness advisory — meta-injection from cognitive audit, not from the human user]\n"
)


def build_reminder_message(text: str, *, timestamp: float = 0.0) -> UserMessage:
    """Build the synthetic user message that carries a reminder into the loop.

    ``text`` is the raw reminder text from a ``Verdict.reminder_text`` /
    ``ReplayRecord.output["surface_reminder"]`` field; the preamble is
    prepended here so callers never have to remember to add it.
    """
    return text_message(REMINDER_PREAMBLE + text, timestamp=timestamp)


__all__ = ["REMINDER_PREAMBLE", "build_reminder_message"]
