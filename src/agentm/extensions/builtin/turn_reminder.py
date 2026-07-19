"""Builtin ``turn_reminder`` atom -- budget-aware, cache-friendly runway warnings.

Pairs with the ``loop_budget`` atom. As the agent approaches the loop's
``max_turns`` / ``max_tool_calls`` cap, this atom injects a short reminder so
the model can wrap up and submit a final response instead of being
hard-stopped mid-thought (a hard budget stop gives it no chance to summarize).

Cache discipline -- the whole point of *where* we inject:

* The reminder is appended to the **end of the last message** in the
  send-list, never to the system prompt. Touching the system prompt would
  shift the very front of the context and invalidate the entire KV / prefix
  cache every turn. The last message is the freshest, not-yet-cached tail, so
  appending there keeps the cached prefix (system + all prior messages)
  byte-identical -- maximum cache hit.
* The override is rebuilt from the pristine ``event.messages`` on every
  ``before_send``, so a re-fired preflight never double-appends.

Budget is read live from the :data:`LOOP_BUDGET_SERVICE` (registered by the
``loop_budget`` atom). With no cap the atom stays silent.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    BeforeRunEvent,
    BeforeSendEvent,
    LOOP_BUDGET_SERVICE,
    LoopConfig,
    TextContent,
    ToolResultEvent,
    ToolResultMessage,
    TurnBeginEvent,
    UserMessage,
)
from agentm.extensions import ExtensionManifest


class TurnReminderConfig(BaseModel):
    warn_within: int = 5
    finalize_tool: str = ""


@dataclass(slots=True)
class _TurnReminderState:
    turn_index: int = 0
    tool_calls_used: int = 0

    def reset(self) -> None:
        self.turn_index = 0
        self.tool_calls_used = 0


MANIFEST = ExtensionManifest(
    name="turn_reminder",
    description=(
        "Injects a budget runway warning at the tail of the last message as "
        "the loop approaches its max_turns / max_tool_calls cap."
    ),
    registers=(
        "event:before_run",
        "event:turn_begin",
        "event:tool_result",
        "event:before_send",
    ),
    config_schema=TurnReminderConfig,
    requires=(),
    priority=AtomInstallPriority.CONTEXT,
)


class _TurnReminderRuntime:
    def __init__(self, api: AtomAPI, config: TurnReminderConfig) -> None:
        self._api = api
        self._warn_within = config.warn_within
        self._finalize_tool = config.finalize_tool
        # Per-run counters. ``turn_index`` is authoritative from ``turn_begin``;
        # ``tool_calls_used`` mirrors the loop's own counter (both reset per run).
        self._state = _TurnReminderState()

    def install(self) -> None:
        self._api.on(BeforeRunEvent.CHANNEL, self.on_before_run)
        self._api.on(TurnBeginEvent.CHANNEL, self.on_turn_begin)
        self._api.on(ToolResultEvent.CHANNEL, self.on_tool_result)
        self._api.on(BeforeSendEvent.CHANNEL, self.before_send)

    def on_before_run(self, _: BeforeRunEvent) -> None:
        # Each run starts a fresh budget frame (range(max_turns) from 0).
        self._state.reset()

    def on_turn_begin(self, event: TurnBeginEvent) -> None:
        self._state.turn_index = event.turn_index

    def on_tool_result(self, _: ToolResultEvent) -> None:
        self._state.tool_calls_used += 1

    def before_send(self, event: BeforeSendEvent) -> dict[str, list[Any]] | None:
        runway = self._runway()
        if runway is None:
            return None
        turns_left, tools_left = runway
        if not _warning_triggered(turns_left, tools_left, self._warn_within):
            return None

        messages = list(event.messages)
        if _last_step(turns_left, tools_left, threshold=2) and self._finalize_tool:
            messages.append(_finalize_now_message(self._finalize_tool))
            return {"messages": messages}

        text = _format_warning(turns_left, tools_left, self._finalize_tool)
        updated = _append_to_last_message(messages, text)
        if updated is None:
            return None
        return {"messages": updated}

    def _runway(self) -> tuple[int | None, int | None] | None:
        cfg = self._api.services.get(LOOP_BUDGET_SERVICE)
        if not isinstance(cfg, LoopConfig):
            return None
        max_turns = cfg.max_turns
        max_tool_calls = cfg.max_tool_calls
        # No cap on either axis => nothing to warn about.
        if max_turns is None and max_tool_calls is None:
            return None

        turns_left = (
            max_turns - self._state.turn_index if max_turns is not None else None
        )
        tools_left = (
            max_tool_calls - self._state.tool_calls_used
            if max_tool_calls is not None
            else None
        )
        return turns_left, tools_left


def install(api: AtomAPI, config: TurnReminderConfig) -> None:
    _TurnReminderRuntime(api, config).install()


def _warning_triggered(
    turns_left: int | None,
    tools_left: int | None,
    warn_within: int,
) -> bool:
    return (turns_left is not None and turns_left <= warn_within) or (
        tools_left is not None and tools_left <= warn_within
    )


def _last_step(
    turns_left: int | None,
    tools_left: int | None,
    *,
    threshold: int,
) -> bool:
    return (turns_left is not None and turns_left <= threshold) or (
        tools_left is not None and tools_left <= threshold
    )


def _finalize_now_message(finalize_tool: str) -> UserMessage:
    return UserMessage(
        role="user",
        content=[
            TextContent(
                type="text",
                text=(
                    f"SYSTEM: Your investigation time is up. You MUST call "
                    f"`{finalize_tool}` NOW with your best findings. "
                    f"Do NOT make any more investigation calls."
                ),
            )
        ],
        timestamp=_time.time(),
    )


def _format_warning(
    turns_left: int | None,
    tools_left: int | None,
    finalize_tool: str = "",
) -> str:
    parts: list[str] = []
    if turns_left is not None:
        parts.append(f"{max(turns_left, 0)} turn(s)")
    if tools_left is not None:
        parts.append(f"{max(tools_left, 0)} tool call(s)")
    budget = " and ".join(parts)
    tool_hint = (
        f" Call `{finalize_tool}` NOW."
        if finalize_tool
        else " Submit your final response NOW (e.g. via the response/finalize tool)."
    )
    last = (turns_left is not None and turns_left <= 1) or (
        tools_left is not None and tools_left <= 1
    )
    if last:
        return (
            f"[budget] This is effectively your LAST step ({budget} left before a "
            f"hard stop with no chance to summarize).{tool_hint}"
        )
    return (
        f"[budget] Only {budget} remaining before a hard stop (no chance to "
        f"summarize). Start wrapping up.{tool_hint}"
    )


def _append_to_last_message(
    messages: list[Any], text: str
) -> list[Any] | None:
    """Return a copy of ``messages`` with ``text`` appended to the tail of the
    last message -- never the system prompt -- so the cached prefix stays
    byte-identical.

    ``UserMessage`` takes a trailing text block directly. A
    ``ToolResultMessage`` may only hold ``ToolResultBlock``s, so the text is
    appended inside the last tool-result block's content (a shape providers
    serialise as tool_result + text in the same user turn -- keeping role
    alternation valid; a standalone trailing user message would not). Other /
    empty tails leave the list unchanged (return ``None``).

    Messages are frozen dataclasses, so the tail message (and, for tool
    results, its last block) is rebuilt rather than mutated in place.
    """

    if not messages:
        return None
    last = messages[-1]
    block = TextContent(type="text", text=text)

    if isinstance(last, UserMessage):
        new_last = replace(last, content=[*last.content, block])
    elif isinstance(last, ToolResultMessage):
        if not last.content:
            return None
        blocks = list(last.content)
        last_block = blocks[-1]
        blocks[-1] = replace(last_block, content=[*last_block.content, block])
        new_last = replace(last, content=blocks)
    else:
        return None

    return [*messages[:-1], new_last]
