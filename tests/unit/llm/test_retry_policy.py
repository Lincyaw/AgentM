from __future__ import annotations

import sys
import types
from typing import Any, cast

import pytest

from agentm.core.abi import EventBus, MessageEnd, Model, text_message
from agentm.core.abi.events import DiagnosticEvent
from agentm.extensions.builtin import retry_policy
from agentm.extensions.builtin.retry_policy import ExponentialBackoffRetry
from agentm.llm import anthropic, openai


class _FakeRateLimitError(Exception):
    pass


class _EmptyAsyncStream:
    def __aiter__(self) -> _EmptyAsyncStream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def close(self) -> None:
        return None


async def _collect(stream: Any) -> list[Any]:
    return [event async for event in stream]


@pytest.mark.asyncio
async def test_openai_stream_fn_retries_typed_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(RateLimitError=_FakeRateLimitError),
    )

    class _Completions:
        def __init__(self) -> None:
            self.attempts = 0

        async def create(self, **_: Any) -> _EmptyAsyncStream:
            self.attempts += 1
            if self.attempts == 1:
                raise _FakeRateLimitError("rate limit")
            return _EmptyAsyncStream()

    completions = _Completions()
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions),
    )
    class _Api:
        events = EventBus()

        def __init__(self) -> None:
            self.services: dict[str, Any] = {}
            self.provider: Any = None

        def set_service(self, name: str, service: Any) -> None:
            self.services[name] = service

        def get_service(self, name: str) -> Any:
            return self.services.get(name)

        def register_provider(self, name: str, config: Any) -> None:
            assert name == "openai"
            self.provider = config

    api = _Api()
    retry_policy.install(cast(Any, api), {"max_retries": 1, "base_delay": 0})
    openai.install(api, {"model": "fake"})
    stream_fn = api.provider.stream_fn
    stream_fn.client = cast(Any, client)
    stream_fn.clock = lambda: 0.0

    events = await _collect(
        stream_fn(
            messages=[text_message("hi")],
            model=Model(
                id="fake",
                provider="openai",
                context_window=100,
                max_output_tokens=10,
            ),
            tools=[],
        )
    )

    assert completions.attempts == 2
    assert isinstance(events[-1], MessageEnd)


@pytest.mark.asyncio
async def test_anthropic_stream_fn_retries_typed_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        types.SimpleNamespace(RateLimitError=_FakeRateLimitError),
    )

    class _StreamContext:
        def __init__(self, owner: _Messages) -> None:
            self._owner = owner

        async def __aenter__(self) -> _EmptyAsyncStream:
            self._owner.attempts += 1
            if self._owner.attempts == 1:
                raise _FakeRateLimitError("rate limit")
            return _EmptyAsyncStream()

        async def __aexit__(self, *_: Any) -> None:
            return None

    class _Messages:
        def __init__(self) -> None:
            self.attempts = 0

        def stream(self, **_: Any) -> _StreamContext:
            return _StreamContext(self)

    messages = _Messages()
    client = types.SimpleNamespace(messages=messages)
    stream_fn = anthropic.AnthropicStreamFn(
        client=cast(Any, client),
        retry_policy=ExponentialBackoffRetry(max_retries=1, base_delay=0),
        clock=lambda: 0.0,
    )

    events = await _collect(
        stream_fn(
            messages=[text_message("hi")],
            model=Model(
                id="fake",
                provider="anthropic",
                context_window=100,
                max_output_tokens=10,
            ),
            tools=[],
        )
    )

    assert messages.attempts == 2
    assert isinstance(events[-1], MessageEnd)


def test_openai_verify_ssl_false_emits_warning_once() -> None:
    bus = EventBus()
    diagnostics: list[DiagnosticEvent] = []
    bus.on(DiagnosticEvent.CHANNEL, diagnostics.append)

    class _Api:
        events = bus

        def get_service(self, name: str) -> None:
            assert name == "retry_policy"
            return None

        def register_provider(self, name: str, config: Any) -> None:
            assert name == "openai"
            assert config.stream_fn.verify_ssl is False

    openai.install(
        _Api(),
        {
            "model": "fake",
            "verify_ssl": False,
        },
    )

    warnings = [item for item in diagnostics if item.level == "warning"]
    assert len(warnings) == 1
    assert "verify_ssl=False" in warnings[0].message


@pytest.mark.asyncio
async def test_anthropic_stream_fn_uses_configurable_thinking_budgets() -> None:
    class _StreamContext:
        def __init__(self, owner: _Messages) -> None:
            self._owner = owner

        async def __aenter__(self) -> _EmptyAsyncStream:
            return _EmptyAsyncStream()

        async def __aexit__(self, *_: Any) -> None:
            return None

    class _Messages:
        def __init__(self) -> None:
            self.body: dict[str, Any] | None = None

        def stream(self, **body: Any) -> _StreamContext:
            self.body = body
            return _StreamContext(self)

    messages = _Messages()
    stream_fn = anthropic.AnthropicStreamFn(
        client=cast(Any, types.SimpleNamespace(messages=messages)),
        thinking_budgets={"high": 8192},
        clock=lambda: 0.0,
    )

    await _collect(
        stream_fn(
            messages=[text_message("hi")],
            model=Model(
                id="fake",
                provider="anthropic",
                context_window=100,
                max_output_tokens=10,
            ),
            tools=[],
            thinking="high",
        )
    )

    assert messages.body is not None
    assert messages.body["thinking"] == {"type": "enabled", "budget_tokens": 8192}


def test_anthropic_install_passes_thinking_budgets_from_config() -> None:
    class _Api:
        def get_service(self, name: str) -> None:
            assert name == "retry_policy"
            return None

        def register_provider(self, name: str, config: Any) -> None:
            assert name == "anthropic"
            assert config.stream_fn.thinking_budgets == {
                "low": 256,
                "medium": 4096,
                "high": 16384,
            }

    anthropic.install(
        _Api(),
        {"model": "fake", "thinking_budgets": {"low": 256}},
    )


@pytest.mark.asyncio
async def test_openai_drop_thinking_emits_info_diagnostic_once() -> None:
    bus = EventBus()
    diagnostics: list[DiagnosticEvent] = []
    bus.on(DiagnosticEvent.CHANNEL, diagnostics.append)

    class _Completions:
        def __init__(self) -> None:
            self.bodies: list[dict[str, Any]] = []

        async def create(self, **body: Any) -> _EmptyAsyncStream:
            self.bodies.append(body)
            return _EmptyAsyncStream()

    completions = _Completions()
    stream_fn = openai.OpenAIStreamFn(
        client=cast(Any, types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))),
        events=bus,
        clock=lambda: 0.0,
    )
    assistant = cast(
        Any,
        openai.AssistantMessage(
            role="assistant",
            content=[openai.ThinkingBlock(type="thinking", text="hidden")],
            timestamp=0.0,
        ),
    )

    for _ in range(2):
        await _collect(
            stream_fn(
                messages=[assistant],
                model=Model(
                    id="fake",
                    provider="openai",
                    context_window=100,
                    max_output_tokens=10,
                ),
                tools=[],
            )
        )

    info = [item for item in diagnostics if item.level == "info"]
    assert len(info) == 1
    assert "dropped assistant ThinkingBlock" in info[0].message
    assert completions.bodies[0]["messages"] == [{"role": "assistant", "content": ""}]


@pytest.mark.asyncio
async def test_openai_thinking_system_note_round_trip() -> None:
    class _Completions:
        def __init__(self) -> None:
            self.body: dict[str, Any] | None = None

        async def create(self, **body: Any) -> _EmptyAsyncStream:
            self.body = body
            return _EmptyAsyncStream()

    completions = _Completions()
    stream_fn = openai.OpenAIStreamFn(
        client=cast(Any, types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))),
        thinking_round_trip="system_note",
        clock=lambda: 0.0,
    )
    assistant = cast(
        Any,
        openai.AssistantMessage(
            role="assistant",
            content=[
                openai.ThinkingBlock(type="thinking", text="reason"),
                openai.TextContent(type="text", text="answer"),
            ],
            timestamp=0.0,
        ),
    )

    await _collect(
        stream_fn(
            messages=[assistant],
            model=Model(
                id="fake",
                provider="openai",
                context_window=100,
                max_output_tokens=10,
            ),
            tools=[],
        )
    )

    assert completions.body is not None
    assert completions.body["messages"] == [
        {"role": "system", "content": "Previous assistant reasoning: reason"},
        {"role": "assistant", "content": "answer"},
    ]


def test_openai_thinking_raise_strategy_errors() -> None:
    stream_fn = openai.OpenAIStreamFn(thinking_round_trip="raise")
    assistant = cast(
        Any,
        openai.AssistantMessage(
            role="assistant",
            content=[openai.ThinkingBlock(type="thinking", text="hidden")],
            timestamp=0.0,
        ),
    )

    with pytest.raises(ValueError, match="ThinkingBlock"):
        openai._to_openai_messages(
            [assistant],
            system=None,
            thinking_round_trip=stream_fn.thinking_round_trip,
        )


def test_harness_reexports_core_provider_config() -> None:
    from agentm.core.abi.provider import ProviderConfig as CoreProviderConfig
    from agentm.harness.extension import ProviderConfig as HarnessProviderConfig

    assert HarnessProviderConfig is CoreProviderConfig
