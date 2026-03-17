"""Budget middleware — injects remaining-step awareness.

Counts every AI message with tool calls (including ``think``) and
injects urgency messages when the step budget is running low.
When budget is fully exhausted, exposes an ``exhausted`` property
so the worker graph can strip tool bindings and force a plain-text exit.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agentm.middleware import AgentMMiddleware


class BudgetMiddleware(AgentMMiddleware):
    """Pre-model middleware that warns the LLM when step budget is low.

    Exposes ``exhausted`` so that the worker ``llm_call`` node can invoke
    the LLM without tools, physically preventing further tool calls.
    """

    def __init__(self, max_steps: int) -> None:
        self._max_steps = max_steps
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        """True once every step in the budget has been consumed."""
        return self._exhausted

    def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        max_steps = self._max_steps

        # Count every AI message with tool_calls — think is NOT free.
        step = sum(
            1
            for m in messages
            if getattr(m, "type", "") == "ai"
            and getattr(m, "tool_calls", None)
        )
        remaining = max(0, max_steps - step)

        if remaining <= 0:
            self._exhausted = True
            urgency = (
                f"BUDGET EXHAUSTED: All {max_steps} steps used. "
                "STOP immediately — do NOT call any tool including think. "
                "Write your conclusion as plain text now."
            )
        elif remaining <= 3:
            urgency = (
                f"WARNING: {remaining}/{max_steps} steps left. "
                "Summarize your findings NOW. Do NOT call any more tools."
            )
        elif remaining <= max_steps // 3:
            urgency = (
                f"BUDGET: {remaining}/{max_steps} steps left. "
                "Start wrapping up — prioritize the most important remaining queries."
            )
        else:
            return {"messages": messages}

        return {"messages": [*messages, HumanMessage(content=urgency)]}
