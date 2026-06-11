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
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
)
from agentm.core.abi import ResolveSubagentEvent
from agentm.core.runtime.resource_loader import InMemoryResourceLoader
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession
from agentm.core.abi import ProviderConfig


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
            # Step 5b: the finding arrives via the session inbox at the next
            # turn boundary (a <system-reminder>-wrapped user message); the
            # OLD floor-injected <subagent_result> branch is gone.
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
            # Step 5b: same as the check_tasks mode — the finding now lands
            # via the inbox-drain path, not the floor inject.
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

    async def _wait_then_finish(
        self, signal: Any
    ) -> AsyncIterator[AssistantStreamEvent]:
        assert signal is not None
        await signal.wait()
        yield MessageEnd(message=_text_msg("aborted child summary", ts=10.0))

    async def _iter(
        self, message: AssistantMessage
    ) -> AsyncIterator[AssistantStreamEvent]:
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
            ResolveSubagentEvent.CHANNEL,
            lambda event: (
                {"body": CHILD_PERSONA, "tools": []}
                if isinstance(event, ResolveSubagentEvent) and event.name == "worker"
                else None
            ),
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


def _extensions(
    *,
    resolver_module: str,
) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("agentm.extensions.builtin.operations", {}),
        (
            "agentm.extensions.builtin.sub_agent",
            {
                "inherit_extensions": ["operations"],
                "available_inherited_extensions": {
                    "operations": (
                        "agentm.extensions.builtin.operations",
                        {},
                    ),
                },
            },
        ),
        (resolver_module, {}),
    ]


@pytest.mark.asyncio
async def test_subagent_finding_arrives_via_inbox_after_next_turn_boundary(
    tmp_path: Path,
) -> None:
    """Step 5b: a completed child's finding rides through
    ``api.post_inbox(source="subagent")`` and lands as a
    ``<system-reminder>``-wrapped user message on the next turn — replacing
    the old ``decide_turn_action`` completed-unread inject branch. The
    finding text (a ``<subagent_result>`` block) MUST appear in user
    content; the previous "completed-unread is suppressed by ``check_tasks``"
    assertion no longer holds (the inbox path is authoritative)."""
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
    # The inbox-drained finding lands as a system-reminder-wrapped user
    # message containing the <subagent_result> block.
    user_texts = [
        block.text
        for message in messages
        if getattr(message, "role", None) == "user"
        for block in getattr(message, "content", [])
        if getattr(block, "type", None) == "text"
    ]
    assert any(
        "<subagent_result" in text
        and '<system-reminder source="subagent">' in text
        for text in user_texts
    ), f"expected an inbox-drained subagent finding; got user_texts={user_texts!r}"

    await session.shutdown()




@pytest.mark.asyncio
async def test_running_only_second_cancel_auto_aborts_and_surfaces_in_messages(
    tmp_path: Path,
) -> None:
    provider = _LifecycleProvider("auto_abort")
    provider_module = _install_provider_module(
        "tests.integration._fake_subagent_lifecycle_abort_provider", provider
    )
    resolver_module = _install_resolver_module(
        "tests.integration._fake_subagent_lifecycle_abort_resolver"
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

    # Auto-abort flow (post Major-2-option-a, 2026-05-28): turn 1
    # dispatches; turn 2 voluntarily ends, the still-running floor injects
    # <subagent_pending> and bumps the cancel counter; turn 3 voluntarily
    # ends again, the floor aborts every running child and returns None,
    # each child's _finalize_state.post_inbox queues its <subagent_result>
    # onto the session inbox, the runtime keep-alive floor turns the Stop
    # into Step; turn 4 sees the drained <subagent_result> and terminates.
    # No double-delivery (Major 2): the auto-abort branch no longer
    # Inject-s the same payload that the inbox-drain will deliver next
    # turn — that's pinned by
    # test_auto_abort_delivers_each_finding_exactly_once below.
    assert provider.parent_calls == 4
    assert "Task aborted before producing final text." in provider.parent_snapshots[-1]
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

    await session.shutdown()


@pytest.mark.asyncio
async def test_auto_abort_delivers_each_finding_exactly_once(
    tmp_path: Path,
) -> None:
    """Major-2 fail-stop (review of 821f4b23): each aborted child's
    ``<subagent_result>`` MUST appear EXACTLY ONCE across the consecutive
    turns following the auto-abort.

    Pre-fix bug: ``decide_turn_action`` Inject-ed
    ``_notification_message(pending=aborted)`` THIS turn AND
    ``_finalize_state.post_inbox`` queued the same finding for NEXT-turn
    inbox-drain delivery. The inbox dedup_key only dedupes within the
    inbox (not against the Inject), so on fan-outs of N aborted children
    each finding double-appeared. Post-fix (option a): the auto-abort
    branch returns ``None``; the runtime keep-alive floor sees the
    non-empty inbox and turns the parent's ``Stop`` into ``Step``; the
    next turn's context-drain delivers each finding exactly once.

    Uses the single-child ``auto_abort`` scenario from the suite; the
    invariant is "exactly one ``<subagent_result>`` per aborted child" so
    a single child is enough to fail-stop the regression (the bug
    duplicated each finding regardless of fan-out width).
    """
    provider = _LifecycleProvider("auto_abort")
    provider_module = _install_provider_module(
        "tests.integration._fake_subagent_lifecycle_once_provider", provider
    )
    resolver_module = _install_resolver_module(
        "tests.integration._fake_subagent_lifecycle_once_resolver"
    )
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=_extensions(resolver_module=resolver_module),
            provider=(provider_module, {}),
            resource_loader=InMemoryResourceLoader(),
        )
    )

    try:
        messages = await session.prompt("start")

        # Count <subagent_result> blocks in user-role messages across the
        # WHOLE final transcript (both the auto-abort turn and any
        # subsequent inbox-drained turn). Counting on user_texts catches
        # both the legacy Inject path and the inbox-drain path; the
        # invariant is the total count.
        user_texts = [
            block.text
            for message in messages
            if getattr(message, "role", None) == "user"
            for block in getattr(message, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        joined = "\n".join(user_texts)
        result_count = joined.count("<subagent_result")
        assert result_count == 1, (
            f"each aborted child's finding must appear exactly once; "
            f"got {result_count} occurrences of <subagent_result in "
            f"user_texts={user_texts!r}"
        )

        # Cross-check on the LAST parent snapshot too (the parent's final
        # turn context): the finding must be visible there (the model
        # needs to see it to terminate gracefully) and exactly once.
        last_snapshot_count = provider.parent_snapshots[-1].count(
            "<subagent_result"
        )
        assert last_snapshot_count == 1, (
            f"final parent snapshot should contain exactly one "
            f"<subagent_result; got {last_snapshot_count}"
        )
    finally:
        await session.shutdown()

