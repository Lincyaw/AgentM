"""Fail-stop: reasoning_effort / extra_body must reach the provider request body.

These guard a silent-plumbing-break: if either provider stops forwarding the
configured passthrough params to the SDK ``create()`` call, the convenience
knob and the universal escape hatch become no-ops with no error. One focused
test per provider asserts the params land in the body passed to a stubbed
client.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentm.core.abi import TextContent, UserMessage
from agentm.core.abi.stream import Model
from agentm.extensions.builtin.llm_anthropic import AnthropicStreamFn
from agentm.extensions.builtin.llm_openai import OpenAIStreamFn


def _user_msg() -> UserMessage:
    return UserMessage(
        role="user",
        content=[TextContent(type="text", text="hi")],
        timestamp=0.0,
    )


def _model(provider: str) -> Model:
    return Model(
        id="m", provider=provider, context_window=1000, max_output_tokens=16
    )


class _EmptyAsyncStream:
    def __aiter__(self) -> "_EmptyAsyncStream":
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def close(self) -> None:  # pragma: no cover - not hit on clean stream
        return None


@pytest.mark.asyncio
async def test_openai_forwards_reasoning_effort_and_extra_body() -> None:
    captured: dict[str, Any] = {}

    class _Completions:
        async def create(self, **body: Any) -> _EmptyAsyncStream:
            captured.update(body)
            return _EmptyAsyncStream()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    fn = OpenAIStreamFn(
        client=_Client(),  # type: ignore[arg-type]
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )
    async for _ in fn(messages=[_user_msg()], model=_model("openai"), tools=[]):
        pass

    extra = captured["extra_body"]
    # User-supplied extra_body key is preserved, convenience knob is merged in.
    assert extra["thinking"] == {"type": "enabled"}
    assert extra["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_openai_user_extra_body_wins_over_convenience() -> None:
    captured: dict[str, Any] = {}

    class _Completions:
        async def create(self, **body: Any) -> _EmptyAsyncStream:
            captured.update(body)
            return _EmptyAsyncStream()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    fn = OpenAIStreamFn(
        client=_Client(),  # type: ignore[arg-type]
        reasoning_effort="high",
        extra_body={"reasoning_effort": "low"},
    )
    async for _ in fn(messages=[_user_msg()], model=_model("openai"), tools=[]):
        pass

    assert captured["extra_body"]["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_anthropic_maps_reasoning_effort_to_output_config() -> None:
    captured: dict[str, Any] = {}

    class _StreamCtx:
        async def __aenter__(self) -> _EmptyAsyncStream:
            return _EmptyAsyncStream()

        async def __aexit__(self, *exc: Any) -> None:
            return None

    class _Messages:
        def stream(self, **body: Any) -> _StreamCtx:
            captured.update(body)
            return _StreamCtx()

    class _Client:
        messages = _Messages()

    fn = AnthropicStreamFn(
        client=_Client(),  # type: ignore[arg-type]
        reasoning_effort="high",
        extra_body={"foo": "bar"},
    )
    async for _ in fn(messages=[_user_msg()], model=_model("anthropic"), tools=[]):
        pass

    extra = captured["extra_body"]
    assert extra["foo"] == "bar"
    assert extra["output_config"] == {"effort": "high"}
