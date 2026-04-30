"""End-to-end streaming tests for ``AnthropicStreamFn``.

Drives the StreamFn with a fake AsyncAnthropic client whose
``messages.stream(...)`` returns an async context manager that yields
SimpleNamespace events shaped like Anthropic SDK raw events. This lets us
exercise the full translation pipeline without instantiating Pydantic models
or touching the network.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from agentm.core.kernel.messages import (
    AssistantMessage,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    UserMessage,
)
from agentm.core.kernel.stream import (
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextDelta,
    ThinkingDelta,
    ToolCallArgsDelta,
    ToolCallEnd,
    ToolCallStart,
)
from agentm.llm.anthropic import AnthropicStreamFn


# --- Fake SDK plumbing ------------------------------------------------------


class _FakeStream:
    """Async-iterable replacement for ``anthropic`` ``AsyncMessageStream``."""

    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)
        self.closed = False
        # Pause-on-event lets the abort test interleave cleanly.
        self._pause_after: int | None = None
        self._paused = asyncio.Event()
        self._paused.set()

    def pause_after(self, n: int, gate: asyncio.Event) -> None:
        self._pause_after = n
        self._gate = gate

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Any]:
        for i, ev in enumerate(self._events):
            if self._pause_after is not None and i == self._pause_after:
                # Yield control so the abort signal can be set externally.
                await asyncio.sleep(0)
            yield ev


class _FakeStreamCtx:
    def __init__(self, stream: _FakeStream) -> None:
        self._stream = stream

    async def __aenter__(self) -> _FakeStream:
        return self._stream

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class _FakeMessages:
    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.last_kwargs: dict[str, Any] | None = None
        self.stream_obj = _FakeStream(events)

    def stream(self, **kwargs: Any) -> _FakeStreamCtx:
        self.last_kwargs = kwargs
        return _FakeStreamCtx(self.stream_obj)


class _FakeClient:
    def __init__(self, events: list[Any]) -> None:
        self.messages = _FakeMessages(events)


def _model() -> Model:
    return Model(
        id="claude-opus-4-7",
        provider="anthropic",
        context_window=200_000,
        max_output_tokens=32_768,
    )


def _msg_start(
    *,
    input_tokens: int = 10,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_write,
            )
        ),
    )


def _block_start_text(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=SimpleNamespace(type="text", text=""),
    )


def _block_start_tool(index: int, *, id: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=SimpleNamespace(type="tool_use", id=id, name=name),
    )


def _block_start_thinking(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=SimpleNamespace(type="thinking", thinking="", signature=None),
    )


def _delta_text(index: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _delta_input_json(index: int, partial: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="input_json_delta", partial_json=partial),
    )


def _delta_thinking(index: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="thinking_delta", thinking=text),
    )


def _delta_signature(index: int, sig: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="signature_delta", signature=sig),
    )


def _block_stop(index: int) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_stop", index=index)


def _msg_delta(stop_reason: str, *, output_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason=stop_reason),
        usage=SimpleNamespace(
            input_tokens=None,
            output_tokens=output_tokens,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        ),
    )


def _msg_stop() -> SimpleNamespace:
    return SimpleNamespace(type="message_stop")


def _user(text: str) -> UserMessage:
    return UserMessage(
        role="user",
        content=[TextContent(type="text", text=text)],
        timestamp=0.0,
    )


async def _collect(
    stream_fn: AnthropicStreamFn, **kwargs: Any
) -> list[AssistantStreamEvent]:
    events: list[AssistantStreamEvent] = []
    async for ev in stream_fn(**kwargs):
        events.append(ev)
    return events


# --- Scenarios -------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_only_stream() -> None:
    events = [
        _msg_start(input_tokens=12),
        _block_start_text(0),
        _delta_text(0, "Hello"),
        _delta_text(0, " world"),
        _block_stop(0),
        _msg_delta("end_turn", output_tokens=3),
        _msg_stop(),
    ]
    client = _FakeClient(events)
    fn = AnthropicStreamFn(client=client)  # type: ignore[arg-type]

    out = await _collect(
        fn, messages=[_user("hi")], model=_model(), tools=[]
    )

    text_deltas = [e for e in out if isinstance(e, TextDelta)]
    assert [t.text for t in text_deltas] == ["Hello", " world"]

    # The final event is a MessageEnd with the assembled assistant message.
    assert isinstance(out[-1], MessageEnd)
    msg = out[-1].message
    assert isinstance(msg, AssistantMessage)
    assert msg.stop_reason == "end_turn"
    assert len(msg.content) == 1
    block = msg.content[0]
    assert isinstance(block, TextContent)
    assert block.text == "Hello world"


@pytest.mark.asyncio
async def test_tool_use_stream() -> None:
    events = [
        _msg_start(),
        _block_start_tool(0, id="call-7", name="search"),
        _delta_input_json(0, '{"q":'),
        _delta_input_json(0, '"hi"}'),
        _block_stop(0),
        _msg_delta("tool_use"),
        _msg_stop(),
    ]
    client = _FakeClient(events)
    fn = AnthropicStreamFn(client=client)  # type: ignore[arg-type]

    out = await _collect(fn, messages=[_user("x")], model=_model(), tools=[])

    starts = [e for e in out if isinstance(e, ToolCallStart)]
    args_deltas = [e for e in out if isinstance(e, ToolCallArgsDelta)]
    ends = [e for e in out if isinstance(e, ToolCallEnd)]
    assert starts == [ToolCallStart(id="call-7", name="search")]
    assert [(d.id, d.args_json_delta) for d in args_deltas] == [
        ("call-7", '{"q":'),
        ("call-7", '"hi"}'),
    ]
    assert ends == [ToolCallEnd(id="call-7")]

    final = out[-1]
    assert isinstance(final, MessageEnd)
    assert final.message.stop_reason == "tool_use"
    assert len(final.message.content) == 1
    call = final.message.content[0]
    assert isinstance(call, ToolCallBlock)
    assert call.id == "call-7"
    assert call.name == "search"
    assert call.arguments == {"q": "hi"}


@pytest.mark.asyncio
async def test_thinking_stream_preserves_signature() -> None:
    events = [
        _msg_start(),
        _block_start_thinking(0),
        _delta_thinking(0, "let me "),
        _delta_thinking(0, "reason"),
        _delta_signature(0, "sig-abc"),
        _block_stop(0),
        _block_start_text(1),
        _delta_text(1, "answer"),
        _block_stop(1),
        _msg_delta("end_turn"),
        _msg_stop(),
    ]
    client = _FakeClient(events)
    fn = AnthropicStreamFn(client=client)  # type: ignore[arg-type]

    out = await _collect(
        fn,
        messages=[_user("x")],
        model=_model(),
        tools=[],
        thinking="medium",
    )

    thinking_deltas = [e for e in out if isinstance(e, ThinkingDelta)]
    assert [t.text for t in thinking_deltas] == ["let me ", "reason"]

    final = out[-1]
    assert isinstance(final, MessageEnd)
    blocks = final.message.content
    assert len(blocks) == 2
    assert isinstance(blocks[0], ThinkingBlock)
    assert blocks[0].text == "let me reason"
    assert blocks[0].signature == "sig-abc"
    assert isinstance(blocks[1], TextContent)
    assert blocks[1].text == "answer"

    # Thinking budget should have been forwarded to the SDK call.
    assert client.messages.last_kwargs is not None
    assert client.messages.last_kwargs.get("thinking") == {
        "type": "enabled",
        "budget_tokens": 4096,
    }


@pytest.mark.asyncio
async def test_abort_mid_stream() -> None:
    events = [
        _msg_start(),
        _block_start_text(0),
        _delta_text(0, "partial"),
        # Anything after this point should NOT be yielded as kernel events.
        _delta_text(0, "should-not-leak"),
        _block_stop(0),
        _msg_delta("end_turn"),
        _msg_stop(),
    ]
    client = _FakeClient(events)
    fn = AnthropicStreamFn(client=client)  # type: ignore[arg-type]
    signal = asyncio.Event()

    iterator = fn(
        messages=[_user("x")],
        model=_model(),
        tools=[],
        signal=signal,
    )

    collected: list[AssistantStreamEvent] = []
    saw_partial = False
    async for ev in iterator:
        collected.append(ev)
        if isinstance(ev, TextDelta) and ev.text == "partial" and not saw_partial:
            saw_partial = True
            signal.set()

    # The "should-not-leak" text must never reach the kernel.
    assert all(
        not (isinstance(e, TextDelta) and e.text == "should-not-leak")
        for e in collected
    )
    final = collected[-1]
    assert isinstance(final, MessageEnd)
    assert final.message.stop_reason == "aborted"
    assert client.messages.stream_obj.closed is True


@pytest.mark.asyncio
async def test_usage_populated_on_final_message() -> None:
    events = [
        _msg_start(
            input_tokens=42,
            output_tokens=0,
            cache_read=7,
            cache_write=3,
        ),
        _block_start_text(0),
        _delta_text(0, "ok"),
        _block_stop(0),
        _msg_delta("end_turn", output_tokens=11),
        _msg_stop(),
    ]
    client = _FakeClient(events)
    fn = AnthropicStreamFn(client=client)  # type: ignore[arg-type]

    out = await _collect(fn, messages=[_user("x")], model=_model(), tools=[])
    final = out[-1]
    assert isinstance(final, MessageEnd)
    usage = final.message.usage
    assert usage is not None
    assert usage.input_tokens == 42
    assert usage.output_tokens == 11
    assert usage.cache_read == 7
    assert usage.cache_write == 3
