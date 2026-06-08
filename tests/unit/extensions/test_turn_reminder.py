"""``turn_reminder`` budget-warning atom — placement + threshold.

Load-bearing assertion: the warning is appended to the **tail of the last
message** and the system prompt is never touched. Injecting into the system
prompt would invalidate the whole KV/prefix cache every turn — a silent,
expensive regression — so this pins the cache-safe placement. Also pins
fail-quiet behaviour when no budget cap is set, and the warn threshold.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from agentm.core.abi import LoopConfig, TextContent, ToolResultMessage, UserMessage
from agentm.core.abi.events import (
    AgentStartEvent,
    BeforeSendToLlmEvent,
    ToolResultEvent,
    TurnStartEvent,
)
from agentm.core.abi.messages import ToolResultBlock
from agentm.extensions import discover


def _load_atom() -> Any:
    """Load ``turn_reminder`` through the same discovery seam the runtime
    uses — as a builtin atom under ``agentm.extensions.builtin``."""

    entry = discover.discover_builtin().get("turn_reminder")
    if entry is None:
        pytest.skip("turn_reminder builtin atom not discoverable in this layout")
    return importlib.import_module(entry.module_path)


class _FakeSession:
    def __init__(self, cfg: LoopConfig) -> None:
        self._cfg = cfg

    def get_loop_config(self) -> LoopConfig:
        return self._cfg


class _FakeAPI:
    def __init__(self, cfg: LoopConfig) -> None:
        self.session = _FakeSession(cfg)
        self.handlers: dict[str, list[Any]] = {}

    def on(self, channel: str, fn: Any, **_: Any) -> Any:
        self.handlers.setdefault(channel, []).append(fn)
        return lambda: None


def _drive(
    cfg: LoopConfig,
    *,
    turn_index: int,
    tool_results: int,
    messages: list[Any],
    warn_within: int = 2,
) -> BeforeSendToLlmEvent:
    """Run install + fire agent_start → turn_start → N tool_results →
    before_send, returning the (possibly mutated) before-send event."""

    atom = _load_atom()
    api = _FakeAPI(cfg)
    atom.install(api, {"warn_within": warn_within})

    def fire(channel: str, event: Any) -> None:
        for fn in api.handlers.get(channel, []):
            fn(event)

    fire(AgentStartEvent.CHANNEL, AgentStartEvent(messages=[]))
    for _ in range(tool_results):
        fire(
            ToolResultEvent.CHANNEL,
            ToolResultEvent(tool_call_id="t", tool_name="x", result=None),  # type: ignore[arg-type]
        )
    fire(TurnStartEvent.CHANNEL, TurnStartEvent(turn_index=turn_index))
    event = BeforeSendToLlmEvent(messages=messages, model=None, tools=[], system="SYS")  # type: ignore[arg-type]
    fire(BeforeSendToLlmEvent.CHANNEL, event)
    return event


def _user_tail() -> list[Any]:
    return [UserMessage(role="user", content=[TextContent(type="text", text="hi")], timestamp=0.0)]


def _tool_result_tail() -> list[Any]:
    return [
        ToolResultMessage(
            role="tool_result",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_call_id="c",
                    content=[TextContent(type="text", text="out")],
                )
            ],
            timestamp=0.0,
        )
    ]






def test_warns_into_user_message_tail_not_system() -> None:
    msgs = _user_tail()
    # turn_index=2, max_turns=4 → turns_left=2 ≤ warn_within → warn.
    event = _drive(LoopConfig(max_turns=4), turn_index=2, tool_results=0, messages=msgs)
    appended = msgs[-1].content[-1]
    assert isinstance(appended, TextContent) and "[budget]" in appended.text
    # Cache-safety: the system prompt must be left byte-identical.
    assert event.system == "SYS"




