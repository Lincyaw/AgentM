"""Budget middleware — injects remaining-step awareness.

Tracks two independent budgets:

- **max_steps**: total LLM rounds (every AI message with tool calls,
  including ``think``).  When exhausted the ``exhausted`` flag is set
  and the worker graph strips tool bindings.
- **tool_call_budget**: real tool invocations only (``think`` is
  excluded).  When exhausted the LLM is told to stop calling tools
  but may still ``think``.

Either limit being hit sets ``exhausted = True``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agentm.middleware import AgentMMiddleware

_THINK_TOOL = "think"


class BudgetMiddleware(AgentMMiddleware):
    """Pre-model middleware that warns the LLM when step budget is low.

    Exposes ``exhausted`` so that the worker ``llm_call`` node can invoke
    the LLM without tools, physically preventing further tool calls.
    """

    def __init__(
        self,
        max_steps: int,
        tool_call_budget: int | None = None,
    ) -> None:
        self._max_steps = max_steps
        self._tool_call_budget = tool_call_budget
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        """True once any budget is fully consumed."""
        return self._exhausted

    def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])

        # --- count steps and tool calls from message history ---
        steps = 0
        tool_calls = 0
        for m in messages:
            if getattr(m, "type", "") != "ai":
                continue
            tcs = getattr(m, "tool_calls", None)
            if not tcs:
                continue
            steps += 1
            tool_calls += sum(
                1 for tc in tcs if tc.get("name") != _THINK_TOOL
            )

        step_remaining = max(0, self._max_steps - steps)
        tool_remaining = (
            max(0, self._tool_call_budget - tool_calls)
            if self._tool_call_budget is not None
            else None
        )

        # --- determine urgency ---
        urgency = self._step_urgency(step_remaining)
        tool_urgency = self._tool_urgency(tool_remaining)

        # Pick the more severe message (exhausted > warning > budget > none).
        # If both are present, concatenate them.
        if urgency is None and tool_urgency is None:
            return {"messages": messages}

        parts = [u for u in (urgency, tool_urgency) if u is not None]
        return {"messages": [*messages, HumanMessage(content="\n".join(parts))]}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_urgency(self, remaining: int) -> str | None:
        ms = self._max_steps
        if remaining <= 0:
            self._exhausted = True
            return (
                f"BUDGET EXHAUSTED: All {ms} steps used. "
                "STOP immediately — do NOT call any tool including think. "
                "Write your conclusion as plain text now."
            )
        if remaining <= 3:
            return (
                f"WARNING: {remaining}/{ms} steps left. "
                "Summarize your findings NOW. Do NOT call any more tools."
            )
        if remaining <= ms // 3:
            return (
                f"BUDGET: {remaining}/{ms} steps left. "
                "Start wrapping up — prioritize the most important remaining queries."
            )
        return None

    def _tool_urgency(self, remaining: int | None) -> str | None:
        if remaining is None:
            return None
        budget = self._tool_call_budget
        if remaining <= 0:
            self._exhausted = True
            return (
                f"TOOL BUDGET EXHAUSTED: All {budget} tool calls used. "
                "STOP calling tools. Write your conclusion as plain text now."
            )
        if remaining <= 3:
            return (
                f"TOOL WARNING: {remaining}/{budget} tool calls left. "
                "Wrap up your investigation — use remaining calls wisely."
            )
        if budget is not None and remaining <= budget // 3:
            return (
                f"TOOL BUDGET: {remaining}/{budget} tool calls left. "
                "Start wrapping up — prioritize the most critical queries."
            )
        return None
