"""Loop detection middleware -- detects repetitive tool call patterns.

Two detection modes:

1. **Exact-match** (original): counts identical ``(tool_name, args)`` pairs
   in a sliding window.  Effective for tools with deterministic arguments
   (e.g. ``query_sql`` with the same SQL).

2. **Think-stall**: counts consecutive AI messages where the *only* tool
   call is ``think`` (arguments ignored, since every thought is unique).
   Detects "planning paralysis" where the LLM keeps reasoning but never
   takes an action.

Both modes inject a ``HumanMessage`` warning when their threshold is
exceeded, nudging the LLM to take a concrete action.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from agentm.middleware import AgentMMiddleware

_THINK_TOOL = "think"

# ---------------------------------------------------------------------------
# Hook factory — exact-match detection
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
# Hook factory — think-stall detection
# ---------------------------------------------------------------------------


def _count_trailing_think_only(messages: list[Any]) -> int:
    """Count consecutive AI messages from the tail where the only tool is ``think``.

    Walks backward through the message list. An AI message counts as
    "think-only" if it carries exactly one tool call named ``think``
    (or carries no tool calls at all — pure text continuation).  The
    streak breaks as soon as an AI message invokes any other tool.
    """
    streak = 0
    for msg in reversed(messages):
        if getattr(msg, "type", "") != "ai":
            continue
        tool_calls = getattr(msg, "tool_calls", None) or []
        tool_names = {tc.get("name", "") for tc in tool_calls}
        if tool_names and tool_names != {_THINK_TOOL}:
            break  # found an action tool — streak ends
        streak += 1
    return streak


def build_think_stall_hook(
    max_consecutive: int = 3,
    warning_template: str | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a ``pre_model_hook`` that breaks think-only stalls.

    When the last *max_consecutive* AI messages contain only ``think``
    tool calls (no action tools), injects a directive telling the LLM
    to take a concrete action immediately.

    Parameters
    ----------
    max_consecutive:
        Number of consecutive think-only rounds before the warning fires.
    warning_template:
        Custom warning message template.  May contain ``{streak}`` which
        is replaced with the actual streak count.  When ``None``, a
        generic default is used.
    """
    _default_template = (
        "THINK-STALL WARNING: You have called only `think` for the "
        "last {streak} rounds without taking any action. Thinking "
        "alone does not advance the investigation.\n\n"
        "You MUST call an action tool NOW — one of:\n"
        "- dispatch_agent (to send a worker for data collection)\n"
        "- update_hypothesis (to formalize your reasoning)\n"
        "- finalize via <decision>finalize</decision>\n\n"
        "Do NOT call think again until you have taken an action."
    )
    template = warning_template if warning_template is not None else _default_template

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("llm_input_messages") or state.get("messages", [])
        use_llm_input = "llm_input_messages" in state
        out_key = "llm_input_messages" if use_llm_input else "messages"

        streak = _count_trailing_think_only(messages)
        if streak >= max_consecutive:
            warning_text = template.format(streak=streak)
            return {out_key: [*messages, HumanMessage(content=warning_text)]}

        return {out_key: messages}

    return hook


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------


class LoopDetectionMiddleware(AgentMMiddleware):
    """Detect and warn about repetitive tool call patterns.

    Combines two detection strategies:

    1. **Exact-match**: identical ``(tool_name, args)`` repeated within a
       sliding window → warns about specific repeated calls.
    2. **Think-stall**: consecutive rounds where the only tool is ``think``
       → forces the LLM to take a concrete action.
    """

    def __init__(
        self,
        threshold: int = 5,
        window_size: int = 20,
        think_stall_limit: int = 3,
        think_stall_warning: str | None = None,
    ) -> None:
        self._threshold = threshold
        self._window_size = window_size
        self._think_stall_limit = think_stall_limit
        self._exact_hook = build_loop_detection_hook(
            threshold=threshold,
            window_size=window_size,
        )
        self._think_hook = build_think_stall_hook(
            max_consecutive=think_stall_limit,
            warning_template=think_stall_warning,
        )

    def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        # Think-stall fires first (more specific); exact-match as fallback.
        result = self._think_hook(state)
        # If think-stall already injected a warning, skip exact-match.
        messages_in = state.get("llm_input_messages") or state.get("messages", [])
        messages_out = result.get("llm_input_messages") or result.get("messages", [])
        if len(messages_out) > len(messages_in):
            return result
        return self._exact_hook(state)
