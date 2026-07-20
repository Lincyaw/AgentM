"""Default retry policy atom for provider stream functions."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Final, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel as PydanticBaseModel

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    RETRY_POLICY_SERVICE,
    ApiRegisterEvent,
    AgentMessage,
    AssistantStreamEvent,
    CancelSignal,
    DiagnosticEvent,
    Model,
    ProviderConfig,
    RetryPolicy,
    StreamFn,
    ThinkingLevel,
    Tool,
)
from agentm.extensions import ExtensionManifest

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

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
    priority=AtomInstallPriority.POLICY,
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
                sleep_for = delay + (
                    random.uniform(0.0, self.jitter) if self.jitter else 0.0
                )
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                delay *= self.factor


@dataclass(slots=True)
class _RetryingStreamFn:
    inner: StreamFn
    policy: RetryPolicy
    retry_streaming: bool

    def __call__(
        self,
        *,
        messages: list[AgentMessage],
        model: Model,
        tools: list[Tool],
        system: str | None = None,
        signal: CancelSignal | None = None,
        thinking: ThinkingLevel = "off",
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
        messages: list[AgentMessage],
        model: Model,
        tools: list[Tool],
        system: str | None,
        signal: CancelSignal | None,
        thinking: ThinkingLevel,
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
    if isinstance(exc, _StatusCodeError) and exc.status_code == 429:
        return True
    if httpx is not None and isinstance(exc, httpx.TransportError):
        return True
    return False


@runtime_checkable
class _StatusCodeError(Protocol):
    status_code: int


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


class _RetryPolicyRuntime:
    def __init__(self, session: AtomAPI, config: RetryPolicyConfig) -> None:
        self._session = session
        self._policy = ExponentialBackoffRetry(
            max_retries=max(0, config.max_retries),
            base_delay=max(0.0, config.base_delay),
            factor=max(1.0, config.factor),
            jitter=max(0.0, config.jitter),
        )
        self._retry_streaming = config.retry_streaming
        self._wrap_providers = config.wrap_providers
        self._wrapped: set[str] = set()

    def install(self) -> None:
        self._session.services.register(
            RETRY_POLICY_SERVICE,
            self._policy,
            scope="session",
        )
        if not self._wrap_providers:
            return
        self._wrap_current_provider()
        self._session.on(ApiRegisterEvent.CHANNEL, self.on_api_register)

    def on_api_register(self, event: ApiRegisterEvent) -> None:
        if event.kind != "provider":
            return
        provider = event.payload.get("provider")
        if not isinstance(provider, ProviderConfig):
            return
        self._wrap_provider_by_name(
            event.name,
            provider,
            emit_diagnostic=True,
        )

    def _wrap_current_provider(self) -> None:
        current = self._session.get_provider()
        if current is None:
            return
        self._wrap_provider_by_name(
            current.name,
            current,
            emit_diagnostic=False,
        )

    def _wrap_provider_by_name(
        self,
        name: str,
        provider: ProviderConfig,
        *,
        emit_diagnostic: bool,
    ) -> None:
        if name in self._wrapped:
            return
        wrapped_provider = _wrap_provider(
            provider,
            policy=self._policy,
            retry_streaming=self._retry_streaming,
        )
        if wrapped_provider is provider:
            return
        self._wrapped.add(name)
        self._session.register_provider(name, wrapped_provider, replace=True)
        if emit_diagnostic:
            self._emit_wrapped_diagnostic(name)

    def _emit_wrapped_diagnostic(self, name: str) -> None:
        self._session.bus.emit_sync(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="info",
                source="retry_policy",
                message=f"wrapped provider {name!r} with retry policy",
            ),
        )


def install(session: AtomAPI, config: RetryPolicyConfig) -> None:
    _RetryPolicyRuntime(session, config).install()


__all__: Final = ["ExponentialBackoffRetry", "MANIFEST", "install"]
