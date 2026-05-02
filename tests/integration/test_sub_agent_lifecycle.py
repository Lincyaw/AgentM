from __future__ import annotations

import json
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import AssistantMessage, AssistantStreamEvent, MessageEnd, Model, TextContent, ToolCallBlock
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig
from agentm.harness.extension import ProviderConfig


CHILD_PERSONA = "CHILD PERSONA"


def _flatten_text(messages: list[Any]) -> str:
    chunks: list[str] = []
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


class _LifecycleProvider:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.parent_calls = 0
        self.parent_snapshots: list[str] = []

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
        del model, tools, thinking
        if system and CHILD_PERSONA in system:
            return self._child_iter(messages=messages, signal=signal)
        return self._parent_iter(messages)

    def _parent_iter(self, messages: list[Any]) -> AsyncIterator[AssistantStreamEvent]:
        self.parent_calls += 1
        snapshot = _flatten_text(messages)
        self.parent_snapshots.append(snapshot)

        if self.mode == "check_tasks":
            if self.parent_calls == 1:
                return self._iter(
                    _tool_call_msg(
                        call_id="dispatch-1",
                        name="dispatch_agent",
                        arguments={
                            "purpose": "collect result",
                            "prompt": "finish-now",
                            "subagent_type": "worker",
                        },
                        ts=1.0,
                    )
                )
            if self.parent_calls == 2:
                return self._iter(
                    _tool_call_msg(
                        call_id="check-1",
                        name="check_tasks",
                        arguments={},
                        ts=2.0,
                    )
                )
            assert "<subagent_result" not in snapshot
            return self._iter(_text_msg("done", ts=3.0))

        if self.mode == "wait_subagent":
            if self.parent_calls == 1:
                return self._iter(
                    _tool_call_msg(
                        call_id="dispatch-1",
                        name="dispatch_agent",
                        arguments={
                            "purpose": "collect result",
                            "prompt": "finish-now",
                            "subagent_type": "worker",
                        },
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
            assert "<subagent_result" not in snapshot
            return self._iter(_text_msg("done", ts=3.0))

        if self.mode == "auto_abort":
            if self.parent_calls == 1:
                return self._iter(
                    _tool_call_msg(
                        call_id="dispatch-1",
                        name="dispatch_agent",
                        arguments={
                            "purpose": "watch long task",
                            "prompt": "sleep-until-abort",
                            "subagent_type": "worker",
                        },
                        ts=1.0,
                    )
                )
            return self._iter(_text_msg("", ts=float(self.parent_calls)))

        raise AssertionError(f"unknown mode: {self.mode}")

    def _child_iter(
        self,
        *,
        messages: list[Any],
        signal: Any,
    ) -> AsyncIterator[AssistantStreamEvent]:
        prompt_text = _flatten_text(messages)
        if "sleep-until-abort" in prompt_text:
            return self._wait_then_finish(signal)
        return self._iter(_text_msg("child result", ts=10.0))

    async def _wait_then_finish(self, signal: Any) -> AsyncIterator[AssistantStreamEvent]:
        assert signal is not None
        await signal.wait()
        yield MessageEnd(message=_text_msg("aborted child summary", ts=10.0))

    async def _iter(self, message: AssistantMessage) -> AsyncIterator[AssistantStreamEvent]:
        yield MessageEnd(message=message)


def _install_provider_module(name: str, provider: _LifecycleProvider) -> str:
    module = types.ModuleType(name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-lifecycle",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="fake-lifecycle",
                    provider="fake",
                    context_window=10_000,
                    max_output_tokens=1_000,
                ),
                name="fake-lifecycle",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


def _install_resolver_module(name: str) -> str:
    module = types.ModuleType(name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.on(
            "resolve_subagent",
            lambda event: {"body": CHILD_PERSONA, "tools": []}
            if event.get("name") == "worker"
            else None,
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
            payload = "".join(
                text_block.text
                for text_block in getattr(block, "content", [])
                if getattr(text_block, "type", None) == "text"
            )
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            task_id = data.get("task_id")
            if isinstance(task_id, str):
                return task_id
    raise AssertionError("dispatch_agent result did not contain a task_id")


def _extensions(
    *,
    resolver_module: str,
    trajectory_path: Path | None = None,
    trajectory_channels: list[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    extensions: list[tuple[str, dict[str, Any]]] = [
        (
            "agentm.extensions.builtin.sub_agent",
            {
                "inherit_extensions": [],
                "available_inherited_extensions": {},
            },
        ),
        (resolver_module, {}),
    ]
    if trajectory_path is not None:
        extensions.append(
            (
                "agentm.extensions.builtin.trajectory",
                {
                    "path": str(trajectory_path),
                    "channels": trajectory_channels,
                },
            )
        )
    return extensions


@pytest.mark.asyncio
async def test_check_tasks_marks_results_read_so_before_agent_end_does_not_redeliver(
    tmp_path: Path,
) -> None:
    provider = _LifecycleProvider("check_tasks")
    provider_module = _install_provider_module(
        "tests.integration._fake_subagent_lifecycle_check_tasks_provider", provider
    )
    resolver_module = _install_resolver_module(
        "tests.integration._fake_subagent_lifecycle_check_tasks_resolver"
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=_extensions(resolver_module=resolver_module),
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    messages = await session.prompt("start")

    assert provider.parent_calls == 3
    assert "<subagent_result" not in provider.parent_snapshots[-1]
    assert all(
        "<subagent_result" not in block.text
        for message in messages
        if getattr(message, "role", None) == "user"
        for block in getattr(message, "content", [])
        if getattr(block, "type", None) == "text"
    )

    await session.shutdown()


@pytest.mark.asyncio
async def test_wait_subagent_returns_terminal_row_and_consumes_it(
    tmp_path: Path,
) -> None:
    provider = _LifecycleProvider("wait_subagent")
    provider_module = _install_provider_module(
        "tests.integration._fake_subagent_lifecycle_wait_provider", provider
    )
    resolver_module = _install_resolver_module(
        "tests.integration._fake_subagent_lifecycle_wait_resolver"
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=_extensions(resolver_module=resolver_module),
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    messages = await session.prompt("start")

    assert provider.parent_calls == 3
    assert "<subagent_result" not in provider.parent_snapshots[-1]
    wait_payloads = [
        json.loads(block.content[0].text)
        for message in messages
        if getattr(message, "role", None) == "tool_result"
        for block in getattr(message, "content", [])
        if getattr(block, "tool_call_id", None) == "wait-1"
    ]
    assert len(wait_payloads) == 1
    wait_payload = wait_payloads[0]
    assert wait_payload["purpose"] == "collect result"
    assert wait_payload["status"] == "completed"
    assert wait_payload["error"] is None
    assert wait_payload["final_text"] == "child result"
    assert isinstance(wait_payload["task_id"], str) and wait_payload["task_id"]
    assert isinstance(wait_payload["final_message_count"], int)

    await session.shutdown()


@pytest.mark.asyncio
async def test_running_only_second_cancel_auto_aborts_and_is_visible_in_trajectory(
    tmp_path: Path,
) -> None:
    provider = _LifecycleProvider("auto_abort")
    provider_module = _install_provider_module(
        "tests.integration._fake_subagent_lifecycle_abort_provider", provider
    )
    resolver_module = _install_resolver_module(
        "tests.integration._fake_subagent_lifecycle_abort_resolver"
    )
    trajectory_path = tmp_path / "trajectory.jsonl"
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=_extensions(
                resolver_module=resolver_module,
                trajectory_path=trajectory_path,
                trajectory_channels=["agent_end"],
            ),
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    messages = await session.prompt("start")

    assert provider.parent_calls == 3
    assert "<subagent_pending" in provider.parent_snapshots[-1]
    user_texts = [
        block.text
        for message in messages
        if getattr(message, "role", None) == "user"
        for block in getattr(message, "content", [])
        if getattr(block, "type", None) == "text"
    ]
    assert any("<subagent_pending" in text for text in user_texts)
    assert any(
        "Task aborted before producing final text." in text for text in user_texts
    )

    records = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()]
    agent_end_events = [record for record in records if record["channel"] == "agent_end"]
    assert agent_end_events
    serialized_messages = json.dumps(agent_end_events[-1]["event"]["messages"])
    assert "Task aborted before producing final text." in serialized_messages

    await session.shutdown()
