"""Default retry policy atom for provider stream functions."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Final, TypeVar

from pydantic import BaseModel as PydanticBaseModel

from agentm.core.abi import (
    ApiRegisterEvent,
    AssistantStreamEvent,
    DiagnosticEvent,
    ExtensionAPI,
    Model,
    ProviderConfig,
    RetryPolicy,
    Tool,
)
from agentm.extensions import ExtensionManifest

T = TypeVar("T")

class RetryPolicyConfig(PydanticBaseModel):
    max_retries: int = 7
    base_delay: float = 5.0
    factor: float = 2.0
    jitter: float = 0.0
    retry_streaming: bool = False
    wrap_providers: bool = False

MANIFEST = ExtensionManifest(
    name="retry_policy",
    description=(
        "Register an exponential backoff retry policy and optionally wrap "
        "provider stream functions."
    ),
    registers=("event:api_register",),
    config_schema=RetryPolicyConfig,
    requires=(),
    tier=1,
)

@dataclass(slots=True)
class ExponentialBackoffRetry:
    max_retries: int = 7
    base_delay: float = 5.0
    factor: float = 2.0
    jitter: float = 0.0

    async def run(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        is_retryable: Callable[[BaseException], bool],
    ) -> T:
        attempt = 0
        delay = self.base_delay
        while True:
            try:
                return await fn()
            except Exception as exc:
                if not is_retryable(exc) or attempt >= self.max_retries:
                    raise
                attempt += 1
                sleep_for = delay + (random.uniform(0.0, self.jitter) if self.jitter else 0.0)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                delay *= self.factor

@dataclass(slots=True)
class _RetryingStreamFn:
    inner: Any
    policy: RetryPolicy
    retry_streaming: bool

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Tool],
        system: str | None = None,
        signal: asyncio.Event | None = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        if not self.retry_streaming:
            return self.inner(
                messages=messages,
                model=model,
                tools=tools,
                system=system,
                signal=signal,
                thinking=thinking,
            )
        return self._iter(
            messages=messages,
            model=model,
            tools=tools,
            system=system,
            signal=signal,
            thinking=thinking,
        )

    async def _iter(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Tool],
        system: str | None,
        signal: asyncio.Event | None,
        thinking: str,
    ) -> AsyncIterator[AssistantStreamEvent]:
        yielded = False

        async def _collect_once() -> list[AssistantStreamEvent]:
            events: list[AssistantStreamEvent] = []
            async for event in self.inner(
                messages=messages,
                model=model,
                tools=tools,
                system=system,
                signal=signal,
                thinking=thinking,
            ):
                events.append(event)
            return events

        def _retryable_before_output(exc: BaseException) -> bool:
            return not yielded and _is_generic_retryable(exc)

        events = await self.policy.run(
            _collect_once,
            is_retryable=_retryable_before_output,
        )
        for event in events:
            yielded = True
            yield event

def _is_generic_retryable(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code == 429

def _wrap_provider(
    provider: ProviderConfig,
    *,
    policy: RetryPolicy,
    retry_streaming: bool,
) -> ProviderConfig:
    if isinstance(provider.stream_fn, _RetryingStreamFn):
        return provider
    return replace(
        provider,
        stream_fn=_RetryingStreamFn(
            inner=provider.stream_fn,
            policy=policy,
            retry_streaming=retry_streaming,
        ),
    )

def install(api: ExtensionAPI, config: RetryPolicyConfig) -> None:
    policy = ExponentialBackoffRetry(
        max_retries=max(0, config.max_retries),
        base_delay=max(0.0, config.base_delay),
        factor=max(1.0, config.factor),
        jitter=max(0.0, config.jitter),
    )
    api.set_service("retry_policy", policy)

    retry_streaming = config.retry_streaming
    wrap_providers = config.wrap_providers
    if not wrap_providers:
        return

    wrapped: set[str] = set()

    def _on_register(event: ApiRegisterEvent) -> None:
        if event.kind != "provider" or event.name in wrapped:
            return
        provider = event.payload
        if not isinstance(provider, ProviderConfig):
            return
        wrapped_provider = _wrap_provider(
            provider,
            policy=policy,
            retry_streaming=retry_streaming,
        )
        if wrapped_provider is provider:
            return
        wrapped.add(event.name)
        api.register_provider(event.name, wrapped_provider)
        api.events.emit_sync(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="info",
                source="retry_policy",
                message=f"wrapped provider {event.name!r} with retry policy",
            ),
        )

    current = api.provider
    if current is not None:
        wrapped_provider = _wrap_provider(
            current,
            policy=policy,
            retry_streaming=retry_streaming,
        )
        if wrapped_provider is not current:
            wrapped.add(current.name)
            api.register_provider(current.name, wrapped_provider)

    api.on(ApiRegisterEvent.CHANNEL, _on_register)

__all__: Final = ["ExponentialBackoffRetry", "MANIFEST", "install"]
