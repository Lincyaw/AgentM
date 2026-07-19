"""Builtin ``turn_reminder`` atom -- budget-aware, cache-friendly runway warnings.

Pairs with the ``loop_budget`` atom. As the agent approaches the loop's
``max_turns`` / ``max_tool_calls`` cap, this atom injects a short reminder so
the model can wrap up and submit a final response instead of being
hard-stopped mid-thought (``MaxTurnsExhausted`` / ``BudgetExhausted`` give it
no chance to summarize).

Cache discipline -- the whole point of *where* we inject:

* The reminder is appended to the **end of the last message** in the
  send-list, never to the system prompt. Touching the system prompt would
  shift the very front of the context and invalidate the entire KV / prefix
  cache every turn. The last message is the freshest, not-yet-cached tail, so
  appending there keeps the cached prefix (system + all prior messages)
  byte-identical -- maximum cache hit.
* We only ever append to the *current* last message and never rewrite a
  message that was already sent, so each reminder freezes into the prefix
  verbatim once sent and stays cache-stable on later turns.

The reminder is allowed to persist into history (it materialises into the
durable log when the last message is an in-run tool result). That is
deliberate: a frozen, verbatim reminder is cache-superior to an ephemeral one
(removing it later would change already-cached content and force a re-encode),
and a trail of countdown reminders is harmless context.

Budget is read live from ``None  # v2: readonly session pending.get_loop_config()`` (which reflects the
``loop_budget`` atom's registration). With no cap the atom stays silent.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import (
    BeforeRunEvent,
    BeforeSendEvent,
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
    last_injected_id: int = 0

    def reset(self) -> None:
        self.turn_index = 0
        self.tool_calls_used = 0
        self.last_injected_id = 0


MANIFEST = ExtensionManifest(
    name="turn_reminder",
    description=(
        "Injects a budget runway warning at the tail of the last message as "
        "the loop approaches its max_turns / max_tool_calls cap."
    ),
    registers=(
        "event:agent_start",
        "event:turn_start",
        "event:tool_result",
        "event:before_send_to_llm",
    ),
    config_schema=TurnReminderConfig,
    requires=(),
)


class _TurnReminderRuntime:
    def __init__(self, session: Any, config: TurnReminderConfig) -> None:
        self._session = session
        self._warn_within = config.warn_within
        self._finalize_tool = config.finalize_tool
        # Per-run counters. ``turn_index`` is authoritative from ``turn_start``;
        # ``tool_calls_used`` mirrors the loop's own counter (both reset per run).
        self._state = _TurnReminderState()

    def install(self) -> None:
        self._session.bus.on(BeforeRunEvent.CHANNEL, self.on_agent_start)
        self._session.bus.on(TurnBeginEvent.CHANNEL, self.on_turn_start)
        self._session.bus.on(ToolResultEvent.CHANNEL, self.on_tool_result)
        self._session.bus.on(BeforeSendEvent.CHANNEL, self.before_send)

    def on_agent_start(self, _: BeforeRunEvent) -> None:
        # Each loop.run starts a fresh budget frame (range(max_turns) from 0).
        self._state.reset()

    def on_turn_start(self, event: TurnBeginEvent) -> None:
        self._state.turn_index = event.index

    def on_tool_result(self, _: ToolResultEvent) -> None:
        self._state.tool_calls_used += 1

    def before_send(self, event: BeforeSendEvent) -> dict[str, list[object]] | None:
        runway = self._runway()
        if runway is None:
            return None
        turns_left, tools_left = runway
        if not _warning_triggered(turns_left, tools_left, self._warn_within):
            return None

        msgs = list(event.messages)
        if _last_step(turns_left, tools_left, threshold=2) and self._finalize_tool:
            msgs.append(_finalize_now_message(self._finalize_tool))
            return {"messages": msgs}

        text = _format_warning(turns_left, tools_left, self._finalize_tool)
        _append_to_last_message(msgs, text, self._state)
        return {"messages": msgs}

    def _runway(self) -> tuple[int | None, int | None] | None:
        from agentm.core.abi import LOOP_BUDGET_SERVICE
        svc = self._session.services.get(LOOP_BUDGET_SERVICE)
        max_turns = getattr(svc, "max_turns", None) if svc else None
        max_tool_calls = getattr(svc, "max_tool_calls", None) if svc else None
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


def install(session: Any, config: TurnReminderConfig) -> None:
    _TurnReminderRuntime(session, config).install()


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
    messages: list[Any], text: str, state: _TurnReminderState
) -> None:
    """Append ``text`` to the tail of the last message -- never the system
    prompt -- so the cached prefix stays byte-identical.

    ``UserMessage`` takes a trailing text block directly. A
    ``ToolResultMessage`` may only hold ``ToolResultBlock``s, so the text is
    appended inside the last tool-result block's content (a shape Anthropic
    serialises as tool_result + text in the same user turn -- keeping role
    alternation valid; a standalone trailing user message would not). Other /
    empty tails are skipped.
    """

    if not messages:
        return
    last = messages[-1]
    # Guard against re-injecting into the very same object (e.g. a retry that
    # re-fires before_send without producing a new tail message).
    if id(last) == state.last_injected_id:
        return

    block = TextContent(type="text", text=text)
    if isinstance(last, UserMessage):
        last.content.append(block)
    elif isinstance(last, ToolResultMessage):
        if not last.content:
            return
        # ToolResultBlock is frozen, but its ``content`` list is mutable.
        last.content[-1].content.append(block)
    else:
        return
    state.last_injected_id = id(last)
