"""Loop detection middleware -- detects repetitive tool call patterns.

When an agent repeatedly calls the same tool with similar arguments, it
is likely stuck in a "doom loop".  This middleware counts recent tool
calls within a sliding window and injects a warning ``HumanMessage``
when the repetition count exceeds a configurable threshold.

The warning nudges the LLM to reconsider its approach rather than
retrying the same action.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from agentm.middleware import AgentMMiddleware

# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------


def build_loop_detection_hook(
    threshold: int = 5,
    window_size: int = 20,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a ``pre_model_hook`` that warns when tool calls are looping.

    Parameters
    ----------
    threshold:
        Number of times a (tool_name, args) pair may appear within the
        window before a warning is injected.
    window_size:
        Number of most-recent AI messages to inspect.
    """

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("llm_input_messages") or state.get("messages", [])
        use_llm_input = "llm_input_messages" in state

        # Collect (tool_name, args_key) pairs from the last `window_size`
        # AI messages that carry tool_calls.
        ai_messages = [m for m in messages if getattr(m, "type", "") == "ai"]
        recent_ai = ai_messages[-window_size:] if window_size else ai_messages

        call_counter: Counter[str] = Counter()
        for msg in recent_ai:
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                tc_name = tc.get("name", "")
                tc_args = tc.get("args", {})
                args_key = json.dumps(tc_args, sort_keys=True, default=str)
                call_counter[f"{tc_name}:{args_key}"] += 1

        # Find calls that exceed the threshold
        repeated = [key for key, count in call_counter.items() if count >= threshold]

        out_key = "llm_input_messages" if use_llm_input else "messages"
        if repeated:
            lines = []
            for key in repeated:
                name, _, args_json = key.partition(":")
                lines.append(f"- `{name}({args_json})`")
            warning_text = (
                "LOOP DETECTION WARNING: The following tool calls have been "
                f"repeated {threshold}+ times in the last {window_size} steps. "
                "You appear to be stuck in a loop. Stop and reconsider your "
                "approach -- try a different strategy or tool.\n" + "\n".join(lines)
            )
            warning_msg = HumanMessage(content=warning_text)
            return {out_key: [*messages, warning_msg]}

        return {out_key: messages}

    return hook


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------


class LoopDetectionMiddleware(AgentMMiddleware):
    """Detect and warn about repetitive tool call patterns.

    Tracks per-tool-call counts within a sliding window of recent AI
    messages.  When the same tool is called with identical arguments
    more than ``threshold`` times, injects a ``HumanMessage`` warning
    the agent to reconsider its approach.
    """

    def __init__(
        self,
        threshold: int = 5,
        window_size: int = 20,
    ) -> None:
        self._threshold = threshold
        self._window_size = window_size
        self._hook = build_loop_detection_hook(
            threshold=threshold,
            window_size=window_size,
        )

    def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._hook(state)
