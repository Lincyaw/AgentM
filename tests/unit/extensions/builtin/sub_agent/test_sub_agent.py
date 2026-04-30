from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import pytest

from agentm.core.kernel import (
    AssistantMessage,
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallBlock,
    ToolCallEnd,
    ToolCallStart,
)
from agentm.harness.events import ChildSessionEndEvent, ChildSessionStartEvent
from agentm.harness.extension import ProviderConfig
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


class SharedProvider:
    def __init__(
        self,
        handler: Callable[..., Awaitable[list[AssistantStreamEvent]]],
    ) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

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
        _ = system, thinking
        snapshot = {
            "user_texts": [
                block.text
                for message in messages
                if getattr(message, "role", None) == "user"
                for block in getattr(message, "content", [])
                if isinstance(block, TextContent)
            ],
            "tool_names": [tool.name for tool in tools],
        }
        self.calls.append(snapshot)
        return self._iter(messages=messages, model=model, tools=tools, signal=signal)

    async def _iter(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        signal: Any,
    ) -> AsyncIterator[AssistantStreamEvent]:
        events = await self._handler(
            messages=messages,
            model=model,
            tools=tools,
            signal=signal,
        )
        for event in events:
            yield event


def _text_message_end(
    text: str,
    *,
    stop_reason: Literal[
        "end_turn", "tool_use", "max_tokens", "error", "aborted"
    ] = "end_turn",
) -> MessageEnd:
    return MessageEnd(
        message=AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text=text)],
            timestamp=1.0,
            stop_reason=stop_reason,
        )
    )


def _make_provider_module(name: str, provider: SharedProvider) -> str:
    module = types.ModuleType(name)

    def install(api: Any, config: dict[str, Any]) -> None:
        _ = config
        api.register_provider(
            "shared",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="shared",
                    provider="shared",
                    context_window=10000,
                    max_output_tokens=1000,
                ),
                name="shared",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


def _tool(session: AgentSession, name: str) -> Any:
    for tool in session.tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"missing tool: {name}")


async def _check_tasks(session: AgentSession) -> dict[str, Any]:
    result = await _tool(session, "check_tasks").execute({})
    assert isinstance(result.details, dict)
    return result.details


async def _wait_for_task_status(
    session: AgentSession,
    task_id: str,
    expected: str,
) -> dict[str, Any]:
    async def _poll() -> dict[str, Any]:
        while True:
            payload = await _check_tasks(session)
            for task in payload["tasks"]:
                if task["task_id"] == task_id and task["status"] == expected:
                    return task
            await asyncio.sleep(0)

    return await asyncio.wait_for(_poll(), timeout=1)


@pytest.mark.asyncio
async def test_sub_agent_smoke_emits_child_lifecycle_events(
    tmp_path: Path,
) -> None:
    async def handler(
        *, messages: list[Any], model: Model, tools: list[Any], signal: Any
    ) -> list[AssistantStreamEvent]:
        _ = model, signal
        tool_names = {tool.name for tool in tools}
        if "dispatch_agent" in tool_names:
            if any(getattr(message, "role", None) == "tool_result" for message in messages):
                return [TextDelta(text="parent done"), _text_message_end("parent done")]
            return [
                ToolCallStart(id="dispatch-1", name="dispatch_agent"),
                ToolCallArgsDelta(
                    id="dispatch-1",
                    args_json_delta='{"purpose":"smoke","prompt":"child says hi"}',
                ),
                ToolCallEnd(id="dispatch-1"),
                MessageEnd(
                    message=AssistantMessage(
                        role="assistant",
                        content=[
                            ToolCallBlock(
                                type="tool_call",
                                id="dispatch-1",
                                name="dispatch_agent",
                                arguments={
                                    "purpose": "smoke",
                                    "prompt": "child says hi",
                                },
                            )
                        ],
                        timestamp=1.0,
                        stop_reason="tool_use",
                    )
                ),
            ]
        last_user = next(
            block.text
            for message in reversed(messages)
            if getattr(message, "role", None) == "user"
            for block in getattr(message, "content", [])
            if isinstance(block, TextContent)
        )
        return [TextDelta(text=f"child:{last_user}"), _text_message_end(f"child:{last_user}")]

    provider = SharedProvider(handler)
    provider_module = _make_provider_module(
        "tests.unit.extensions.builtin.sub_agent._provider_smoke",
        provider,
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.sub_agent", {"inherit_extensions": []})
            ],
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    starts: list[ChildSessionStartEvent] = []
    ends: list[ChildSessionEndEvent] = []
    session.bus.on("child_session_start", lambda event: starts.append(event))
    session.bus.on("child_session_end", lambda event: ends.append(event))

    await session.prompt("launch child")

    tasks_payload = await _check_tasks(session)
    assert len(tasks_payload["tasks"]) == 1
    task_id = tasks_payload["tasks"][0]["task_id"]
    completed = await _wait_for_task_status(session, task_id, "completed")

    assert completed["final_message_count"] == 2
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0].child_session_id == ends[0].child_session_id
    assert ends[0].error is None

    await session.shutdown()


@pytest.mark.asyncio
async def test_inject_instruction_is_consumed_by_child_second_turn(
    tmp_path: Path,
) -> None:
    first_turn_release = asyncio.Event()
    observed_child_turns: list[str] = []

    async def handler(
        *, messages: list[Any], model: Model, tools: list[Any], signal: Any
    ) -> list[AssistantStreamEvent]:
        _ = model, tools, signal
        last_user = next(
            block.text
            for message in reversed(messages)
            if getattr(message, "role", None) == "user"
            for block in getattr(message, "content", [])
            if isinstance(block, TextContent)
        )
        observed_child_turns.append(last_user)
        if len(observed_child_turns) == 1:
            await first_turn_release.wait()
            return [TextDelta(text="first turn"), _text_message_end("first turn")]
        return [TextDelta(text=f"second:{last_user}"), _text_message_end(f"second:{last_user}")]

    provider = SharedProvider(handler)
    provider_module = _make_provider_module(
        "tests.unit.extensions.builtin.sub_agent._provider_inject",
        provider,
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.sub_agent", {"inherit_extensions": []})
            ],
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    dispatch = _tool(session, "dispatch_agent")
    inject = _tool(session, "inject_instruction")

    dispatch_result = await dispatch.execute(
        {"purpose": "inject", "prompt": "first prompt"}
    )
    task_id = dispatch_result.details["task_id"]
    inject_result = await inject.execute(
        {"task_id": task_id, "message": "second prompt"}
    )
    assert inject_result.is_error is False

    first_turn_release.set()
    completed = await _wait_for_task_status(session, task_id, "completed")

    assert observed_child_turns == ["first prompt", "second prompt"]
    assert completed["final_message_count"] == 4

    await session.shutdown()


@pytest.mark.asyncio
async def test_abort_task_emits_aborted_child_end_event(
    tmp_path: Path,
) -> None:
    async def handler(
        *, messages: list[Any], model: Model, tools: list[Any], signal: Any
    ) -> list[AssistantStreamEvent]:
        _ = messages, model, tools
        assert signal is not None
        await signal.wait()
        return [TextDelta(text="aborted"), _text_message_end("aborted", stop_reason="aborted")]

    provider = SharedProvider(handler)
    provider_module = _make_provider_module(
        "tests.unit.extensions.builtin.sub_agent._provider_abort",
        provider,
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.sub_agent", {"inherit_extensions": []})
            ],
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    ends: list[ChildSessionEndEvent] = []
    session.bus.on("child_session_end", lambda event: ends.append(event))

    dispatch = _tool(session, "dispatch_agent")
    abort = _tool(session, "abort_task")
    dispatch_result = await dispatch.execute(
        {"purpose": "abort", "prompt": "wait forever"}
    )
    task_id = dispatch_result.details["task_id"]

    abort_result = await abort.execute({"task_id": task_id})
    assert abort_result.is_error is False
    aborted = await _wait_for_task_status(session, task_id, "aborted")

    assert aborted["error"] == "aborted"
    assert len(ends) == 1
    assert ends[0].error == "aborted"

    await session.shutdown()


@pytest.mark.asyncio
async def test_dispatch_agent_enforces_max_workers_cap(
    tmp_path: Path,
) -> None:
    release = asyncio.Event()

    async def handler(
        *, messages: list[Any], model: Model, tools: list[Any], signal: Any
    ) -> list[AssistantStreamEvent]:
        _ = messages, model, tools, signal
        await release.wait()
        return [TextDelta(text="done"), _text_message_end("done")]

    provider = SharedProvider(handler)
    provider_module = _make_provider_module(
        "tests.unit.extensions.builtin.sub_agent._provider_cap",
        provider,
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                (
                    "agentm.extensions.builtin.sub_agent",
                    {"inherit_extensions": [], "max_workers": 4},
                )
            ],
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    dispatch = _tool(session, "dispatch_agent")

    for index in range(4):
        result = await dispatch.execute(
            {"purpose": f"worker-{index}", "prompt": f"prompt-{index}"}
        )
        assert result.is_error is False

    overflow = await dispatch.execute(
        {"purpose": "worker-overflow", "prompt": "extra prompt"}
    )
    assert overflow.is_error is True
    assert "max_workers" in overflow.details["error"]

    release.set()
    await session.shutdown()


@pytest.mark.asyncio
async def test_parent_shutdown_aborts_children_within_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentm.extensions.builtin import sub_agent

    monkeypatch.setattr(sub_agent, "_SHUTDOWN_GRACE_SECONDS", 0.01)

    async def handler(
        *, messages: list[Any], model: Model, tools: list[Any], signal: Any
    ) -> list[AssistantStreamEvent]:
        _ = messages, model, tools
        assert signal is not None
        await signal.wait()
        return [TextDelta(text="shutdown"), _text_message_end("shutdown", stop_reason="aborted")]

    provider = SharedProvider(handler)
    provider_module = _make_provider_module(
        "tests.unit.extensions.builtin.sub_agent._provider_shutdown",
        provider,
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.sub_agent", {"inherit_extensions": []})
            ],
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    ends: list[ChildSessionEndEvent] = []
    session.bus.on("child_session_end", lambda event: ends.append(event))

    dispatch_result = await _tool(session, "dispatch_agent").execute(
        {"purpose": "cleanup", "prompt": "hang"}
    )
    assert dispatch_result.is_error is False

    await asyncio.wait_for(session.shutdown(), timeout=1)

    assert len(ends) == 1
    assert ends[0].error == "aborted"


@pytest.mark.asyncio
async def test_concurrent_children_share_parent_stream_without_cross_talk(
    tmp_path: Path,
) -> None:
    async def handler(
        *, messages: list[Any], model: Model, tools: list[Any], signal: Any
    ) -> list[AssistantStreamEvent]:
        _ = model, signal
        if any(getattr(message, "role", None) == "tool_result" for message in messages):
            tool_result_text = next(
                block.text
                for message in reversed(messages)
                if getattr(message, "role", None) == "tool_result"
                for result_block in getattr(message, "content", [])
                for block in getattr(result_block, "content", [])
                if isinstance(block, TextContent)
            )
            return [
                TextDelta(text=f"done:{tool_result_text}"),
                _text_message_end(f"done:{tool_result_text}"),
            ]
        last_user = next(
            block.text
            for message in reversed(messages)
            if getattr(message, "role", None) == "user"
            for block in getattr(message, "content", [])
            if isinstance(block, TextContent)
        )
        assert any(tool.name == "echo" for tool in tools)
        return [
            ToolCallStart(id=f"echo-{last_user}", name="echo"),
            ToolCallArgsDelta(
                id=f"echo-{last_user}",
                args_json_delta=f'{{"text":"{last_user}"}}',
            ),
            ToolCallEnd(id=f"echo-{last_user}"),
            MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[
                        ToolCallBlock(
                            type="tool_call",
                            id=f"echo-{last_user}",
                            name="echo",
                            arguments={"text": last_user},
                        )
                    ],
                    timestamp=1.0,
                    stop_reason="tool_use",
                )
            ),
        ]

    provider = SharedProvider(handler)
    provider_module = _make_provider_module(
        "tests.unit.extensions.builtin.sub_agent._provider_crosstalk",
        provider,
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                ("agentm.extensions.builtin.sub_agent", {"inherit_extensions": []})
            ],
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )
    dispatch = _tool(session, "dispatch_agent")

    first = await dispatch.execute(
        {
            "purpose": "first",
            "prompt": "alpha",
            "extensions": [["tests.unit.harness_v2._fixtures.echo_ext", {}]],
        }
    )
    second = await dispatch.execute(
        {
            "purpose": "second",
            "prompt": "beta",
            "extensions": [["tests.unit.harness_v2._fixtures.echo_ext", {}]],
        }
    )

    await _wait_for_task_status(session, first.details["task_id"], "completed")
    await _wait_for_task_status(session, second.details["task_id"], "completed")

    child_calls = [call for call in provider.calls if "dispatch_agent" not in call["tool_names"]]
    alpha_calls = [call for call in child_calls if call["user_texts"][-1] == "alpha"]
    beta_calls = [call for call in child_calls if call["user_texts"][-1] == "beta"]

    assert alpha_calls and beta_calls
    assert all("beta" not in call["user_texts"] for call in alpha_calls)
    assert all("alpha" not in call["user_texts"] for call in beta_calls)

    await session.shutdown()
