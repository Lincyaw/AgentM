# code-health: ignore-file[AM025] -- narrows persisted message/content unions
"""Compute a session's terminal result from its committed turns.

The terminating tool and its ``reason`` are recorded on the final turn's
``outcome.cause`` (a :class:`ToolTerminated`), and the tool's post-hook result
is in that turn's ``tool_results``. Reading from there lets callers obtain what
a child produced without guessing at terminal-tool names in the message stream.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentm.core.abi.messages import AssistantMessage, TextContent
from agentm.core.abi.session_api import SessionResult
from agentm.core.abi.termination import ToolTerminated
from agentm.core.abi.trajectory import Turn


def _text(blocks: Sequence[object]) -> str:
    return "".join(b.text for b in blocks if isinstance(b, TextContent))


def compute_session_result(turns: Sequence[Turn]) -> SessionResult | None:
    """Return the terminal result of the last committed turn, or None if empty."""

    if not turns:
        return None
    last = turns[-1]
    cause = last.outcome.cause
    if isinstance(cause, ToolTerminated):
        # The terminating tool is named on the cause; find its result in this
        # turn (last match wins when a same-named tool was called repeatedly).
        for record in reversed(last.tool_results):
            if record.call.name == cause.tool_name:
                return SessionResult(
                    reason=cause.reason,
                    text=_text(record.result.content),
                )
        return SessionResult(reason=cause.reason, text="")
    # Ended by the model finishing its turn / running out of turns: the trailing
    # assistant text is the produced result.
    response = last.response
    if isinstance(response, AssistantMessage):
        return SessionResult(reason=None, text=_text(response.content))
    return SessionResult(reason=None, text="")
