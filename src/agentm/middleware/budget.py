"""Budget middleware — injects remaining-step awareness.

Wraps the existing ``_build_budget_hook`` logic from
``agents/react/sub_agent.py`` as an ``AgentMMiddleware`` subclass.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agentm.middleware import AgentMMiddleware


class BudgetMiddleware(AgentMMiddleware):
    """Pre-model middleware that warns the LLM when step budget is running low."""

    def __init__(self, max_steps: int) -> None:
        self._max_steps = max_steps

    def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        max_steps = self._max_steps

        # Count AI messages with real tool calls (think is free).
        step = 0
        for m in messages:
            if getattr(m, "type", "") != "ai":
                continue
            tool_calls = getattr(m, "tool_calls", None)
            if not tool_calls or any(
                tc.get("name") != "think" for tc in tool_calls
            ):
                step += 1
        remaining = max(0, max_steps - step)

        if remaining <= 3:
            urgency = (
                f"WARNING: You have {remaining} steps remaining out of {max_steps}. "
                f"You MUST summarize your findings NOW and produce your final report. "
                f"Do NOT call any more tools — write your conclusion immediately."
            )
        elif remaining <= max_steps // 3:
            urgency = (
                f"BUDGET: {remaining}/{max_steps} steps remaining. "
                f"Start wrapping up — prioritize the most important remaining queries, "
                f"then produce your summary."
            )
        else:
            return {"messages": messages}

        budget_msg = HumanMessage(content=urgency)
        return {"messages": [*messages, budget_msg]}
