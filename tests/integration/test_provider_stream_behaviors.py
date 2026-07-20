"""Provider behavior contracts exercised through the public StreamFn boundary."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from agentm.core.abi import (
    MessageEnd,
    Model,
    OpaqueThinkingBlock,
    TextContent,
    TextDelta,
    UserMessage,
)
from agentm.extensions.builtin.llm_anthropic import AnthropicStreamFn
from agentm.extensions.builtin.llm_openai import OpenAIStreamFn

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


class _FakeAnthropicStream:
    def __init__(self, events: tuple[object, ...]) -> None:
        self._events = iter(events)
        self.closed = False

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self) -> None:
        self.closed = True


class _FakeAnthropicContext:
    def __init__(self, stream: _FakeAnthropicStream) -> None:
        self._stream = stream

    async def __aenter__(self) -> object:
        return self._stream

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> object:
        return False


class _FakeAnthropicMessages:
    def __init__(self, streams: tuple[tuple[object, ...], ...]) -> None:
        self._streams = iter(streams)
        self.requests: list[dict[str, object]] = []

    def stream(self, **body: object) -> _FakeAnthropicContext:
        self.requests.append(body)
        return _FakeAnthropicContext(_FakeAnthropicStream(next(self._streams)))


class _FakeAnthropicClient:
    def __init__(self, streams: tuple[tuple[object, ...], ...]) -> None:
        self.messages = _FakeAnthropicMessages(streams)


class _FakeOpenAIStream:
    def __init__(self, chunks: tuple[object, ...]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self) -> None:
        return None


class _FakeOpenAICompletions:
    def __init__(self, chunks: tuple[object, ...]) -> None:
        self._chunks = chunks

    async def create(self, **body: object) -> object:
        del body
        return _FakeOpenAIStream(self._chunks)


class _FakeOpenAIClient:
    def __init__(self, chunks: tuple[object, ...]) -> None:
        completions = _FakeOpenAICompletions(chunks)
        self.chat = SimpleNamespace(completions=completions)


def _event(event_type: str, **fields: object) -> object:
    return SimpleNamespace(type=event_type, **fields)


def _message_start() -> object:
    usage = SimpleNamespace(
        input_tokens=2,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return _event(
        "message_start",
        message=SimpleNamespace(usage=usage),
    )


def _message_end_events(*content_events: object) -> tuple[object, ...]:
    usage = SimpleNamespace(
        input_tokens=2,
        output_tokens=1,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return (
        _message_start(),
        *content_events,
        _event(
            "message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=usage,
        ),
        _event("message_stop"),
    )


def _model() -> Model:
    return Model(
        id="claude-test",
        provider="anthropic",
        context_window=1_000,
        max_output_tokens=100,
    )


def _user_message() -> UserMessage:
    return UserMessage(
        role="user",
        content=(TextContent(type="text", text="continue"),),
        timestamp=1.0,
    )


@pytest.mark.asyncio
async def test_anthropic_redacted_thinking_round_trips_between_requests() -> None:
    first_stream = _message_end_events(
        _event(
            "content_block_start",
            index=0,
            content_block=SimpleNamespace(
                type="redacted_thinking",
                data="encrypted-reasoning",
            ),
        ),
        _event("content_block_stop", index=0),
    )
    second_stream = _message_end_events(
        _event(
            "content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text", text=""),
        ),
        _event(
            "content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="done"),
        ),
        _event("content_block_stop", index=0),
    )
    client = _FakeAnthropicClient((first_stream, second_stream))
    provider = AnthropicStreamFn(
        client=cast("AsyncAnthropic", client),
        clock=lambda: 2.0,
    )

    first_events = [
        event
        async for event in provider(
            messages=[_user_message()],
            model=_model(),
            tools=[],
        )
    ]
    first_end = first_events[-1]
    assert isinstance(first_end, MessageEnd)
    opaque = first_end.message.content[0]
    assert isinstance(opaque, OpaqueThinkingBlock)
    assert opaque.provider == "anthropic"
    assert dict(opaque.payload) == {
        "type": "redacted_thinking",
        "data": "encrypted-reasoning",
    }

    second_events = [
        event
        async for event in provider(
            messages=[_user_message(), first_end.message],
            model=_model(),
            tools=[],
        )
    ]
    assert isinstance(second_events[-1], MessageEnd)

    request_messages = client.messages.requests[1]["messages"]
    assert isinstance(request_messages, list)
    assistant = request_messages[1]
    assert isinstance(assistant, dict)
    assert assistant["content"] == [
        {
            "type": "redacted_thinking",
            "data": "encrypted-reasoning",
        }
    ]


@pytest.mark.asyncio
async def test_anthropic_unmodeled_content_fails_instead_of_disappearing() -> None:
    stream = _message_end_events(
        _event(
            "content_block_start",
            index=0,
            content_block=SimpleNamespace(type="server_tool_use"),
        ),
    )
    client = _FakeAnthropicClient((stream,))
    provider = AnthropicStreamFn(
        client=cast("AsyncAnthropic", client),
        clock=lambda: 2.0,
    )

    with pytest.raises(
        ValueError,
        match="content block type is not modeled",
    ):
        async for _ in provider(
            messages=[_user_message()],
            model=_model(),
            tools=[],
        ):
            pass


@pytest.mark.asyncio
async def test_openai_refusal_is_preserved_as_visible_assistant_text() -> None:
    chunk = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    role="assistant",
                    refusal="I cannot help with that.",
                ),
                finish_reason="stop",
            )
        ],
    )
    provider = OpenAIStreamFn(
        client=_FakeOpenAIClient((chunk,)),
        clock=lambda: 2.0,
    )

    events = [
        event
        async for event in provider(
            messages=[_user_message()],
            model=Model(
                id="gpt-test",
                provider="openai",
                context_window=1_000,
                max_output_tokens=100,
            ),
            tools=[],
        )
    ]

    assert isinstance(events[0], TextDelta)
    assert events[0].text == "I cannot help with that."
    end = events[-1]
    assert isinstance(end, MessageEnd)
    assert end.message.content == (
        TextContent(type="text", text="I cannot help with that."),
    )


@pytest.mark.asyncio
async def test_openai_deprecated_function_call_fails_instead_of_disappearing() -> None:
    chunk = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    role="assistant",
                    function_call=SimpleNamespace(
                        name="old_tool",
                        arguments="{}",
                    ),
                ),
                finish_reason="function_call",
            )
        ],
    )
    provider = OpenAIStreamFn(
        client=_FakeOpenAIClient((chunk,)),
        clock=lambda: 2.0,
    )

    with pytest.raises(
        ValueError,
        match="deprecated function_call",
    ):
        async for _ in provider(
            messages=[_user_message()],
            model=Model(
                id="gpt-test",
                provider="openai",
                context_window=1_000,
                max_output_tokens=100,
            ),
            tools=[],
        ):
            pass
