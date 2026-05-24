"""``turn_reminder`` — budget-aware, cache-friendly runway warnings.

Pairs with the ``loop_budget`` atom. As the agent approaches the loop's
``max_turns`` / ``max_tool_calls`` cap, this atom injects a short reminder so
the model can wrap up and submit a final response instead of being
hard-stopped mid-thought (``MaxTurnsExhausted`` / ``BudgetExhausted`` give it
no chance to summarize).

Cache discipline — the whole point of *where* we inject:

* The reminder is appended to the **end of the last message** in the
  send-list, never to the system prompt. Touching the system prompt would
  shift the very front of the context and invalidate the entire KV / prefix
  cache every turn. The last message is the freshest, not-yet-cached tail, so
  appending there keeps the cached prefix (system + all prior messages)
  byte-identical → maximum cache hit.
* We only ever append to the *current* last message and never rewrite a
  message that was already sent, so each reminder freezes into the prefix
  verbatim once sent and stays cache-stable on later turns.

The reminder is allowed to persist into history (it materialises into the
durable log when the last message is an in-run tool result). That is
deliberate: a frozen, verbatim reminder is cache-superior to an ephemeral one
(removing it later would change already-cached content and force a re-encode),
and a trail of countdown reminders is harmless context.

Budget is read live from ``api.session.get_loop_config()`` (which reflects the
``loop_budget`` atom's registration). With no cap the atom stays silent.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi import TextContent, ToolResultMessage, UserMessage
from agentm.core.abi.events import (
    AgentStartEvent,
    BeforeSendToLlmEvent,
    ToolResultEvent,
    TurnStartEvent,
)
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

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
    config_schema={
        "type": "object",
        "properties": {
            "warn_within": {"type": "integer", "minimum": 1, "default": 5},
        },
        "additionalProperties": False,
    },
    requires=(),
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    warn_within = int(config.get("warn_within", 5))

    # Per-run counters. ``turn_index`` is authoritative from ``turn_start``;
    # ``tool_calls_used`` mirrors the loop's own counter (both reset per run).
    state = {"turn_index": 0, "tool_calls_used": 0, "last_injected_id": 0}

    def _on_agent_start(_: AgentStartEvent) -> None:
        # Each loop.run starts a fresh budget frame (range(max_turns) from 0).
        state["turn_index"] = 0
        state["tool_calls_used"] = 0
        state["last_injected_id"] = 0

    def _on_turn_start(event: TurnStartEvent) -> None:
        state["turn_index"] = event.turn_index

    def _on_tool_result(_: ToolResultEvent) -> None:
        state["tool_calls_used"] += 1

    def _before_send(event: BeforeSendToLlmEvent) -> None:
        cfg = api.session.get_loop_config()
        max_turns = cfg.max_turns
        max_tool_calls = cfg.max_tool_calls
        # No cap on either axis ⇒ nothing to warn about.
        if max_turns is None and max_tool_calls is None:
            return

        turns_left = (
            max_turns - state["turn_index"] if max_turns is not None else None
        )
        tools_left = (
            max_tool_calls - state["tool_calls_used"]
            if max_tool_calls is not None
            else None
        )

        triggered = (turns_left is not None and turns_left <= warn_within) or (
            tools_left is not None and tools_left <= warn_within
        )
        if not triggered:
            return

        text = _format_warning(turns_left, tools_left)
        _append_to_last_message(event.messages, text, state)

    api.on(AgentStartEvent.CHANNEL, _on_agent_start)
    api.on(TurnStartEvent.CHANNEL, _on_turn_start)
    api.on(ToolResultEvent.CHANNEL, _on_tool_result)
    api.on(BeforeSendToLlmEvent.CHANNEL, _before_send)


def _format_warning(turns_left: int | None, tools_left: int | None) -> str:
    parts: list[str] = []
    if turns_left is not None:
        parts.append(f"{max(turns_left, 0)} turn(s)")
    if tools_left is not None:
        parts.append(f"{max(tools_left, 0)} tool call(s)")
    budget = " and ".join(parts)
    last = (turns_left is not None and turns_left <= 1) or (
        tools_left is not None and tools_left <= 1
    )
    if last:
        return (
            f"[budget] This is effectively your LAST step ({budget} left before a "
            "hard stop with no chance to summarize). Submit your final response "
            "NOW (e.g. via the response/finalize tool)."
        )
    return (
        f"[budget] Only {budget} remaining before a hard stop (no chance to "
        "summarize). Start wrapping up and prepare to submit your final response."
    )


def _append_to_last_message(
    messages: list[Any], text: str, state: dict[str, int]
) -> None:
    """Append ``text`` to the tail of the last message — never the system
    prompt — so the cached prefix stays byte-identical.

    ``UserMessage`` takes a trailing text block directly. A
    ``ToolResultMessage`` may only hold ``ToolResultBlock``s, so the text is
    appended inside the last tool-result block's content (a shape Anthropic
    serialises as tool_result + text in the same user turn — keeping role
    alternation valid; a standalone trailing user message would not). Other /
    empty tails are skipped.
    """

    if not messages:
        return
    last = messages[-1]
    # Guard against re-injecting into the very same object (e.g. a retry that
    # re-fires before_send without producing a new tail message).
    if id(last) == state["last_injected_id"]:
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
    state["last_injected_id"] = id(last)
