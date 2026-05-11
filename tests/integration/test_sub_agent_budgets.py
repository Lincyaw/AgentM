from __future__ import annotations

import json
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantMessage,
    AssistantStreamEvent,
    FunctionTool,
    LoopConfig,
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
    ToolResult,
)
from agentm.core.abi.extension import ProviderConfig
from agentm.core.abi.events import ResolveSubagentEvent
from agentm.core.runtime.resource_loader import InMemoryResourceLoader
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession


CHILD_PERSONA = "BUDGET CHILD PERSONA"


class _BudgetProvider:
    def __init__(
        self,
        *,
        child_goal_turns: int,
        dispatch_budget: dict[str, int] | None,
    ) -> None:
        self.child_goal_turns = child_goal_turns
        self.dispatch_budget = dispatch_budget
        self.parent_calls = 0

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del model, tools, signal, thinking
        if system and CHILD_PERSONA in system:
            return self._child_iter(messages)
        return self._parent_iter(messages)

    def _parent_iter(self, messages: list[Any]) -> AsyncIterator[AssistantStreamEvent]:
        self.parent_calls += 1
        if self.parent_calls == 1:
            arguments: dict[str, Any] = {
                "purpose": "budget child",
                "prompt": "keep working",
                "subagent_type": "worker",
            }
            if self.dispatch_budget is not None:
                arguments["budget"] = self.dispatch_budget
            return self._iter(
                _tool_call_msg(
                    call_id="dispatch-1",
                    name="dispatch_agent",
                    arguments=arguments,
                    ts=1.0,
                )
            )
        if self.parent_calls == 2:
            task_id = _extract_dispatched_task_id(messages)
            return self._iter(
                _tool_call_msg(
                    call_id="wait-1",
                    name="wait_subagent",
                    arguments={"task_id": task_id},
                    ts=2.0,
                )
            )
        return self._iter(_text_msg("done", ts=3.0))

    def _child_iter(self, messages: list[Any]) -> AsyncIterator[AssistantStreamEvent]:
        tool_result_turns = sum(
            1 for message in messages if getattr(message, "role", None) == "tool_result"
        )
        if tool_result_turns < self.child_goal_turns:
            return self._iter(
                _tool_call_msg(
                    call_id=f"ping-{tool_result_turns + 1}",
                    name="ping",
                    arguments={},
                    ts=10.0 + tool_result_turns,
                )
            )
        return self._iter(_text_msg("child done", ts=20.0))

    async def _iter(
        self, message: AssistantMessage
    ) -> AsyncIterator[AssistantStreamEvent]:
        yield MessageEnd(message=message)


class _PingCounter:
    def __init__(self) -> None:
        self.calls = 0


def _install_provider_module(name: str, provider: _BudgetProvider) -> str:
    module = types.ModuleType(name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-budget",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="fake-budget",
                    provider="fake",
                    context_window=10_000,
                    max_output_tokens=1_000,
                ),
                name="fake-budget",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


def _install_resolver_module(
    name: str, *, persona_budget: dict[str, int] | None
) -> str:
    module = types.ModuleType(name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.on(
            ResolveSubagentEvent.CHANNEL,
            lambda event: (
                {
                    "body": CHILD_PERSONA,
                    "tools": ["ping"],
                    "budget_defaults": persona_budget,
                }
                if isinstance(event, ResolveSubagentEvent) and event.name == "worker"
                else None
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


def _install_ping_module(name: str, counter: _PingCounter) -> str:
    module = types.ModuleType(name)

    async def _ping(_args: dict[str, Any]) -> ToolResult:
        counter.calls += 1
        return ToolResult(content=[TextContent(type="text", text="pong")])

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_tool(
            FunctionTool(
                name="ping",
                description="Test helper tool for sub-agent budget enforcement.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                fn=_ping,
            )
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


def _tool_call_msg(
    *, call_id: str, name: str, arguments: dict[str, Any], ts: float
) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call",
                id=call_id,
                name=name,
                arguments=arguments,
            )
        ],
        timestamp=ts,
        stop_reason="tool_use",
    )


def _text_msg(text: str, *, ts: float) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        timestamp=ts,
        stop_reason="end_turn",
    )


def _extract_dispatched_task_id(messages: list[Any]) -> str:
    for message in reversed(messages):
        if getattr(message, "role", None) != "tool_result":
            continue
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            block_content = getattr(block, "content", None)
            if isinstance(block_content, list):
                payload = "".join(
                    text_block.text
                    for text_block in block_content
                    if getattr(text_block, "type", None) == "text"
                )
            else:
                payload = str(getattr(block, "text", ""))
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            task_id = data.get("task_id")
            if isinstance(task_id, str):
                return task_id
    raise AssertionError("dispatch_agent result did not contain a task_id")


async def _run_budget_case(
    tmp_path: Path,
    *,
    parent_max_turns: int,
    persona_budget: dict[str, int] | None,
    dispatch_budget: dict[str, int] | None,
) -> tuple[_PingCounter, dict[str, Any]]:
    provider = _BudgetProvider(child_goal_turns=5, dispatch_budget=dispatch_budget)
    counter = _PingCounter()
    provider_module = _install_provider_module(
        f"tests.integration._fake_subagent_budget_provider_{id(provider)}", provider
    )
    resolver_module = _install_resolver_module(
        f"tests.integration._fake_subagent_budget_resolver_{id(provider)}",
        persona_budget=persona_budget,
    )
    ping_module = _install_ping_module(
        f"tests.integration._fake_subagent_budget_ping_{id(provider)}", counter
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[

                ("agentm.extensions.builtin.operations_local", {}),
                (
                    "agentm.extensions.builtin.sub_agent",
                    {
                        "inherit_extensions": ["operations_local", "ping"],
                        "available_inherited_extensions": {
                            "operations_local": (
                                "agentm.extensions.builtin.operations_local",
                                {},
                            ),
                            "ping": (ping_module, {}),
                        },
                    },
                ),
                (resolver_module, {}),
            ],
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
            loop_config=LoopConfig(max_turns=parent_max_turns),
        )
    )
    try:
        messages = await session.prompt("start")
        wait_payloads = [
            json.loads(block.content[0].text)
            for message in messages
            if getattr(message, "role", None) == "tool_result"
            for block in getattr(message, "content", [])
            if getattr(block, "tool_call_id", None) == "wait-1"
        ]
        assert len(wait_payloads) == 1
        return counter, wait_payloads[0]
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_dispatch_budget_argument_clamps_child_max_turns(tmp_path: Path) -> None:
    counter, payload = await _run_budget_case(
        tmp_path,
        parent_max_turns=9,
        persona_budget=None,
        dispatch_budget={"max_turns": 3},
    )

    assert counter.calls == 3
    assert payload["budget"]["max_turns"] == 3
    assert payload["final_text"] is None


@pytest.mark.asyncio
async def test_persona_budget_defaults_apply_when_dispatch_budget_is_missing(
    tmp_path: Path,
) -> None:
    counter, payload = await _run_budget_case(
        tmp_path,
        parent_max_turns=9,
        persona_budget={"max_turns": 2},
        dispatch_budget=None,
    )

    assert counter.calls == 2
    assert payload["budget"]["max_turns"] == 2
    assert payload["final_text"] is None


@pytest.mark.asyncio
async def test_dispatch_budget_overrides_persona_defaults(tmp_path: Path) -> None:
    counter, payload = await _run_budget_case(
        tmp_path,
        parent_max_turns=9,
        persona_budget={"max_turns": 2},
        dispatch_budget={"max_turns": 4},
    )

    assert counter.calls == 4
    assert payload["budget"]["max_turns"] == 4
    assert payload["final_text"] is None
