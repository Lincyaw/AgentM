"""Pre-completion checklist middleware -- reminds the agent to verify work.

When the most recent AI message has no ``tool_calls`` (indicating the
agent intends to finish), this middleware injects a ``HumanMessage``
with a configurable verification checklist.

The reminder fires **at most once** per conversation to avoid creating
an infinite "please verify" loop.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage

from agentm.middleware import AgentMMiddleware

_DEFAULT_CHECKLIST = (
    "Before you finish, please verify:\n"
    "1. Have you tested your solution?\n"
    "2. Does the output match the task requirements?\n"
    "3. Are there any edge cases you haven't handled?"
)

# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------


def build_pre_completion_hook(
    checklist: str | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a ``pre_model_hook`` that injects a verification reminder.

    The reminder is injected when the last AI message has no
    ``tool_calls`` (the agent appears to be finishing).  It fires at
    most once per hook instance to prevent infinite loops.

    Parameters
    ----------
    checklist:
        Custom checklist text.  Falls back to a sensible default when
        ``None``.
    """
    resolved_checklist = checklist or _DEFAULT_CHECKLIST
    triggered = {"value": False}

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("llm_input_messages") or state.get("messages", [])
        use_llm_input = "llm_input_messages" in state
        out_key = "llm_input_messages" if use_llm_input else "messages"

        if triggered["value"]:
            return {out_key: messages}

        # Find the last AI message
        last_ai = None
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "ai":
                last_ai = msg
                break

        if last_ai is None:
            return {out_key: messages}

        tool_calls = getattr(last_ai, "tool_calls", None) or []
        if tool_calls:
            # Agent still calling tools -- not finishing yet
            return {out_key: messages}

        # Agent wants to finish -- inject checklist
        triggered["value"] = True
        reminder = HumanMessage(content=resolved_checklist)
        return {out_key: [*messages, reminder]}

    return hook


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------


class PreCompletionChecklistMiddleware(AgentMMiddleware):
    """Inject a verification reminder when the agent appears to be finishing.

    If the last AI message has no ``tool_calls`` (agent wants to stop),
    injects a ``HumanMessage`` reminding it to verify its work first.
    Only triggers once per conversation to avoid infinite loops.
    """

    def __init__(self, checklist: str | None = None) -> None:
        self._checklist = checklist
        self._hook = build_pre_completion_hook(checklist=checklist)

    def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._hook(state)
