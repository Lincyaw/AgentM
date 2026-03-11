"""Tool call deduplication for Sub-Agent ReAct loops.

Two-layer dedup:
1. Pre-model hook (soft): reminds LLM it already called a tool with the same args.
2. Tool wrapper (hard): blocks repeat calls with a short rejection message.

The goal is token savings — repeat calls are rejected, not re-served from cache.

Ref: designs/tool-dedup.md
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Callable

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel


class DedupTracker:
    """Per-task cache for tool call deduplication.

    Uses OrderedDict with FIFO eviction to bound memory.
    """

    def __init__(self, max_cache_size: int = 50) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_cache_size

    def make_key(self, tool_name: str, args: dict[str, Any]) -> str:
        """Deterministic cache key — arg order does not matter."""
        args_str = json.dumps(args, sort_keys=True, default=str)
        return f"{tool_name}:{args_str}"

    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache

    def lookup(self, key: str) -> str | None:
        """Return cached result or None if not found."""
        return self._cache.get(key)

    def store(self, key: str, result: str) -> None:
        """Store result, evicting oldest entry if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = result
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)


_EXCLUDED_TOOLS = frozenset({"think"})


def build_dedup_hook(
    tracker: DedupTracker,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a pre_model_hook that scans history and injects dedup reminders.

    1. Scans message history for (AIMessage tool_call → ToolMessage result) pairs,
       populating the tracker with each unique call.
    2. For any tool calls already in the cache, injects a HumanMessage telling
       the LLM not to repeat those calls (no result content — to save tokens).

    Excludes tools in _EXCLUDED_TOOLS (e.g. 'think').
    """

    def hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("llm_input_messages") or state.get("messages", [])
        use_llm_input = "llm_input_messages" in state

        # --- Phase 1: Scan history to populate tracker ---
        pending_calls: dict[str, tuple[str, dict[str, Any]]] = {}
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            if msg_type == "ai":
                tool_calls = getattr(msg, "tool_calls", None) or []
                for tc in tool_calls:
                    tc_name = tc.get("name", "")
                    if tc_name in _EXCLUDED_TOOLS:
                        continue
                    tc_id = tc.get("id", "")
                    tc_args = tc.get("args", {})
                    if tc_id:
                        pending_calls[tc_id] = (tc_name, tc_args)
            elif msg_type == "tool":
                tc_id = getattr(msg, "tool_call_id", "")
                if tc_id in pending_calls:
                    tool_name, tool_args = pending_calls.pop(tc_id)
                    key = tracker.make_key(tool_name, tool_args)
                    result = getattr(msg, "content", "")
                    if isinstance(result, str):
                        tracker.store(key, result)

        # --- Phase 2: Build reminder for already-cached calls ---
        # Check the last AIMessage's tool calls against the tracker.
        # Only warn if a call is in the tracker AND does NOT have a
        # corresponding ToolMessage result in the current history — i.e.
        # the LLM is about to repeat a call it already made before.
        # Calls whose results already appear in history are legitimate
        # (already executed) and should not trigger a warning.
        answered_tc_ids: set[str] = set()
        for msg in messages:
            if getattr(msg, "type", "") == "tool":
                tc_id = getattr(msg, "tool_call_id", "")
                if tc_id:
                    answered_tc_ids.add(tc_id)

        reminders: list[str] = []
        for msg in reversed(messages):
            msg_type = getattr(msg, "type", "")
            if msg_type == "ai":
                tool_calls = getattr(msg, "tool_calls", None) or []
                for tc in tool_calls:
                    tc_name = tc.get("name", "")
                    if tc_name in _EXCLUDED_TOOLS:
                        continue
                    tc_id = tc.get("id", "")
                    # Skip if this call already has a ToolMessage result
                    if tc_id in answered_tc_ids:
                        continue
                    tc_args = tc.get("args", {})
                    key = tracker.make_key(tc_name, tc_args)
                    if tracker.has(key):
                        args_str = json.dumps(tc_args, sort_keys=True, default=str)
                        reminders.append(
                            f"- `{tc_name}({args_str})` — you already have this result"
                        )
                break  # Only check the last AIMessage

        out_key = "llm_input_messages" if use_llm_input else "messages"
        if reminders:
            reminder_text = (
                "\n\nDEDUP WARNING: You have already called the following tools with "
                "identical arguments. Do NOT repeat these calls — use your earlier results:\n"
                + "\n".join(reminders)
            )
            dedup_msg = HumanMessage(content=reminder_text)
            return {out_key: [*messages, dedup_msg]}

        return {out_key: messages}

    return hook


def wrap_tool_with_dedup(tool: BaseTool, tracker: DedupTracker) -> BaseTool:
    """Wrap a tool so repeated (name, args) calls are blocked.

    First call: executes normally, records the key in tracker.
    Repeat call: returns a short rejection message without re-executing.
    The goal is token savings — no cached result is returned.
    Preserves tool name, description, and args_schema.
    """
    if tool.name in _EXCLUDED_TOOLS:
        return tool

    original_func = tool.func if hasattr(tool, "func") else None
    original_coroutine = tool.coroutine if hasattr(tool, "coroutine") else None

    _BLOCKED_MSG = (
        "BLOCKED: You already called this tool with identical arguments. "
        "Refer to the earlier result in your conversation history."
    )

    def _resolve_args(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            return kwargs
        if args:
            schema = getattr(tool, "args_schema", None)
            if schema and hasattr(schema, "model_fields"):
                field_names = list(schema.model_fields.keys())
                return dict(zip(field_names, args))
            return {"_positional": list(args)}
        return {}

    def dedup_func(*args: Any, **kwargs: Any) -> str:
        tool_args = _resolve_args(*args, **kwargs)
        key = tracker.make_key(tool.name, tool_args)
        if tracker.has(key):
            return _BLOCKED_MSG

        result = original_func(*args, **kwargs) if original_func else ""
        result_str = str(result)
        tracker.store(key, result_str)
        return result_str

    async def dedup_coroutine(*args: Any, **kwargs: Any) -> str:
        tool_args = _resolve_args(*args, **kwargs)
        key = tracker.make_key(tool.name, tool_args)
        if tracker.has(key):
            return _BLOCKED_MSG

        result = await original_coroutine(*args, **kwargs) if original_coroutine else ""
        result_str = str(result)
        tracker.store(key, result_str)
        return result_str

    schema = getattr(tool, "args_schema", None)
    tool_kwargs: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description or "",
        "func": dedup_func,
        "coroutine": dedup_coroutine if original_coroutine else None,
    }
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        tool_kwargs["args_schema"] = schema

    return StructuredTool(**tool_kwargs)
