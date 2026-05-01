"""Shared test fixtures for AgentM test suite."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import pytest

from agentm.ai.api_registry import register_api_provider, unregister_api_providers
from agentm.ai.types import ProviderDefinition, ResolvedAuth
from agentm.core.kernel import (
    AssistantContent,
    AssistantMessage,
    Model,
    TextContent,
    ToolCallBlock,
)
from agentm.core.kernel.stream import StreamFn

from tests.integration.scenarios._fixtures.scripted_provider import ScriptedStream
from tests.unit.extensions.builtin import _helpers
from tests.unit.harness_v2._fixtures.fake_provider import FakeStream

_TEST_PROVIDER_SOURCE = "tests.conftest.providers"


def _register_provider(definition: ProviderDefinition) -> None:
    register_api_provider(definition, source_id=_TEST_PROVIDER_SOURCE)


@pytest.fixture(scope="session", autouse=True)
def register_test_providers() -> Iterator[None]:
    unregister_api_providers(_TEST_PROVIDER_SOURCE)

    def fake_model_factory(model_id: str) -> Model:
        return Model(
            id=model_id,
            provider="fake",
            context_window=10_000,
            max_output_tokens=1_000,
        )

    def fake_stream_factory(
        model: Model,
        config: dict[str, object],
        auth: ResolvedAuth | None,
    ) -> StreamFn:
        del model, config, auth
        return FakeStream()

    _register_provider(
        ProviderDefinition(
            id="fake",
            display_name="Fake",
            api="test-fake",
            default_model="fake",
            model_factory=fake_model_factory,
            stream_factory=fake_stream_factory,
            requires_auth=False,
        )
    )

    def recording_model_factory(model_id: str) -> Model:
        return Model(
            id=model_id,
            provider="recording",
            context_window=4_096,
            max_output_tokens=1_024,
        )

    def recording_stream_factory(
        model: Model,
        config: dict[str, object],
        auth: ResolvedAuth | None,
    ) -> StreamFn:
        del auth
        raw_response_texts = config.get("response_texts", ["ok"])
        if isinstance(raw_response_texts, list):
            response_texts = [str(text) for text in raw_response_texts]
        else:
            response_texts = ["ok"]
        raw_tool_calls = config.get("tool_calls", [])
        tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []
        scripted_messages: list[AssistantMessage] = []

        for index, text in enumerate(response_texts):
            content: list[AssistantContent]
            stop_reason: Literal["end_turn", "tool_use"] = "end_turn"
            if index < len(tool_calls):
                tc_obj = tool_calls[index]
                tc = tc_obj if isinstance(tc_obj, dict) else {}
                content = [
                    ToolCallBlock(
                        type="tool_call",
                        id=str(tc.get("id", f"call-{index + 1}")),
                        name=str(tc["name"]),
                        arguments=dict(tc.get("arguments", {})),
                    )
                ]
                stop_reason = "tool_use"
            else:
                content = [TextContent(type="text", text=text)]

            scripted_messages.append(
                AssistantMessage(
                    role="assistant",
                    content=content,
                    timestamp=float(index + 1),
                    stop_reason=stop_reason,
                )
            )

        _helpers.LAST_STREAM = _helpers.RecordingStream(scripted_messages)
        return _helpers.LAST_STREAM

    _register_provider(
        ProviderDefinition(
            id="recording",
            display_name="Recording",
            api="test-recording",
            default_model="recording-model",
            model_factory=recording_model_factory,
            stream_factory=recording_stream_factory,
            requires_auth=False,
        )
    )

    def scripted_model_factory(model_id: str) -> Model:
        return Model(
            id=model_id,
            provider="scripted-fake",
            context_window=10_000,
            max_output_tokens=1_000,
        )

    def scripted_stream_factory(
        model: Model,
        config: dict[str, object],
        auth: ResolvedAuth | None,
    ) -> StreamFn:
        del model, auth
        tool_name = str(config.get("tool_name", "echo"))
        arguments = config.get("arguments", {})
        final_text = str(config.get("final_text", "done"))
        return ScriptedStream(
            tool_name=tool_name,
            arguments=dict(arguments if isinstance(arguments, dict) else {}),
            final_text=final_text,
        )

    _register_provider(
        ProviderDefinition(
            id="scripted-fake",
            display_name="Scripted Fake",
            api="test-scripted",
            default_model="scripted-fake",
            model_factory=scripted_model_factory,
            stream_factory=scripted_stream_factory,
            requires_auth=False,
        )
    )

    yield

    unregister_api_providers(_TEST_PROVIDER_SOURCE)
