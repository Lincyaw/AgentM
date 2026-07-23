# code-health: ignore-file[AM025] -- vendor LLM adapters normalize untyped provider SDK payloads
"""Anthropic Messages API provider — native ``StreamFn`` implementation.

This module plugs into the kernel via the ``StreamFn`` Protocol and extension
composition contracts described in ``docs/refactor-abstract-inventory.md``.

Boundaries:

* The kernel layer (``agentm.core.abi``) is the only AgentM dependency
  imported at module import time. The harness layer is **not** imported here.
* Streaming is delegated to the official ``anthropic`` Python SDK
  (:class:`anthropic.AsyncAnthropic`); we never reimplement HTTP/SSE.

Conversion layout:

* ``_to_anthropic_messages``: kernel ``AgentMessage`` list → API messages dict.
* ``_to_anthropic_tools``: kernel ``Tool`` list → API tools dict.
* ``AnthropicStreamFn.__call__``: drives the SDK stream and translates each
  raw event into a kernel ``AssistantStreamEvent``.
"""

# code-health: ignore-file[AM022] -- adapts untyped Anthropic SDK request and stream objects

from __future__ import annotations

import base64
from loguru import logger
import os
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Protocol,
    runtime_checkable,
)

from agentm.core.abi import (
    Aborted,
    AgentMessage,
    AtomInstallPriority,
    AssistantContent,
    AssistantMessage,
    AssistantStreamEvent,
    CancelSignal,
    EndTurn,
    ImageContent,
    MaxTokens,
    MessageEnd,
    Model,
    OpaqueThinkingBlock,
    PauseTurn,
    ProviderConfig,
    RetryPolicy,
    TerminationHint,
    TextContent,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    Tool,
    ToolCallArgsDelta,
    ToolCallBlock,
    ToolCallEnd,
    ToolCallStart,
    ToolResultBlock,
    ToolResultMessage,
    ToolUseExpected,
    Usage,
    UserMessage,
    VendorSpecific,
)
from agentm.core.abi.messages import thaw_json
from pydantic import BaseModel, ConfigDict

from agentm.extensions import ExtensionManifest

from agentm.core.abi import RETRY_POLICY_SERVICE
from agentm.core.lib.async_cancel import (
    OperationCancelledBySignal,
    await_with_cancel_signal,
)
from agentm.core.lib import StreamAccumulator, ToolSpecAdapter, encode_tool_args

if TYPE_CHECKING:  # pragma: no cover - import only used for type hints
    from anthropic import AsyncAnthropic


class LlmAnthropicConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "claude-sonnet-4-6"
    api_key: str | None = None
    base_url: str | None = None
    name: str | None = None
    default_headers: dict[str, str] | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    thinking_budgets: dict[str, int] | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None


MANIFEST = ExtensionManifest(
    name="llm_anthropic",
    description="Register an Anthropic Messages API LLM stream provider.",
    # No provider capability is declared: the registry name is config-driven
    # (config['name'], defaulting to "anthropic") and only known at install,
    # so it cannot be a statically declarable/verifiable capability. Providers
    # are resolved through ProviderResolver, not the dependency solver.
    registers=(),
    config_schema=LlmAnthropicConfig,
    sensitive_config_fields=("api_key", "default_headers"),
    requires=(),
    priority=AtomInstallPriority.PROVIDER,
)


def _is_anthropic_retryable(exc: BaseException) -> bool:
    try:
        import anthropic
    except ImportError:  # pragma: no cover - SDK dependency is optional here
        return False
    # RateLimitError: server-side throttle. APIConnectionError /
    # APITimeoutError: transport stalls surfaced by the finite read
    # timeout set in ``_get_client`` — without retry they propagate up and
    # fail the whole firing on a single half-dead connection. Build the
    # tuple via getattr so a partial SDK / test double missing a name is
    # tolerated (mirrors ``_is_openai_retryable``).
    retryable_types = tuple(
        err_type
        for name in ("RateLimitError", "APIConnectionError", "APITimeoutError")
        if isinstance((err_type := anthropic.__dict__.get(name)), type)
    )
    return bool(retryable_types) and isinstance(exc, retryable_types)


# --- Model registry ---------------------------------------------------------


def _build_model(
    model_id: str,
    *,
    provider: str = "anthropic",
    context_window: int = 1_000_000,
    max_output_tokens: int = 64_000,
) -> Model:
    """Construct a kernel ``Model`` descriptor for the given Anthropic id.

    ``context_window`` and ``max_output_tokens`` come from caller config —
    we do not maintain a hard-coded model table, since Anthropic-compatible
    proxies (mimo, MiniMax, Doubao, …) ship arbitrary ids that no static
    table can keep up with. Defaults are sized for long tool-call
    arguments on capable models: 1M/64K. Override via ``config['context_window']`` /
    ``config['max_output_tokens']`` when the deployed model differs.
    """

    return Model(
        id=model_id,
        provider=provider,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )


# --- Message / tool serialization ------------------------------------------


def _encode_image(image: ImageContent) -> dict[str, Any]:
    """Convert kernel ``ImageContent`` to an Anthropic image content block."""

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": image.mime_type,
            "data": base64.b64encode(image.data).decode("ascii"),
        },
    }


def _encode_user_content(
    blocks: list[TextContent | ImageContent],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextContent):
            out.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            out.append(_encode_image(block))
        else:  # pragma: no cover - exhaustive over the union
            raise TypeError(f"unexpected user content type: {type(block)!r}")
    return out


def _encode_assistant_content(
    blocks: list[AssistantContent],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextContent):
            out.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingBlock):
            entry: dict[str, Any] = {"type": "thinking", "thinking": block.text}
            if block.signature is not None:
                entry["signature"] = block.signature
            out.append(entry)
        elif isinstance(block, OpaqueThinkingBlock):
            if block.provider != "anthropic":
                raise ValueError(
                    "AnthropicStreamFn cannot encode opaque reasoning owned by "
                    f"provider {block.provider!r}"
                )
            block_type = block.payload.get("type")
            data = block.payload.get("data")
            if (
                block_type != "redacted_thinking"
                or not isinstance(data, str)
                or set(block.payload) != {"type", "data"}
            ):
                raise ValueError(
                    "AnthropicStreamFn supports only a redacted_thinking "
                    "opaque reasoning payload"
                )
            out.append(dict(block.payload))
        elif isinstance(block, ToolCallBlock):
            out.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments,
                }
            )
        else:  # pragma: no cover
            raise TypeError(f"unexpected assistant content type: {type(block)!r}")
    return out


def _encode_tool_result_block(block: ToolResultBlock) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": block.tool_call_id,
        "content": _encode_user_content(list(block.content)),
        "is_error": block.is_error,
    }


def _to_anthropic_messages(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Convert kernel messages to the Anthropic Messages API request shape.

    Adjacent ``ToolResultMessage``s are packed into a single user-role message
    with multiple ``tool_result`` content blocks (Anthropic's required shape).
    """

    out: list[dict[str, Any]] = []
    previous_role: str | None = None
    was_tool_result_user = False
    for msg in messages:
        if isinstance(msg, UserMessage):
            out.append(
                {"role": "user", "content": _encode_user_content(list(msg.content))}
            )
            previous_role = "user"
            was_tool_result_user = False
        elif isinstance(msg, AssistantMessage):
            out.append(
                {
                    "role": "assistant",
                    "content": _encode_assistant_content(list(msg.content)),
                }
            )
            previous_role = "assistant"
            was_tool_result_user = False
        elif isinstance(msg, ToolResultMessage):
            blocks = [_encode_tool_result_block(b) for b in msg.content]
            if out and previous_role == "user" and was_tool_result_user:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": "user", "content": blocks})
            previous_role = "user"
            was_tool_result_user = True
        else:  # pragma: no cover - exhaustive
            raise TypeError(f"unsupported message type: {type(msg)!r}")
    return out


@dataclass(frozen=True, slots=True)
class AnthropicToolSpecAdapter(ToolSpecAdapter):
    """Convert AgentM tools to Anthropic Messages API tool specs."""

    def vendor_spec(self, tool: Tool) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def encode_tool_args(self, args: Mapping[str, Any]) -> str:
        return encode_tool_args(args)


def _to_anthropic_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    adapter = AnthropicToolSpecAdapter()
    return [adapter.vendor_spec(t) for t in tools]


# --- Streaming bridge -------------------------------------------------------


@dataclass(slots=True)
class _StreamState:
    """Provider event mapping state for one Anthropic stream."""

    accumulator: StreamAccumulator = field(default_factory=StreamAccumulator)
    scratch: dict[int, dict[str, Any]] = field(default_factory=dict)
    usage: Usage | None = None
    stop_reason: str | None = None
    termination: TerminationHint | None = None


def _map_stop_reason(raw: str | None) -> TerminationHint | None:
    """Translate Anthropic ``stop_reason`` into a kernel ``TerminationHint``."""

    if raw is None:
        return None
    if raw == "end_turn" or raw == "stop_sequence":
        return EndTurn()
    if raw == "tool_use":
        return ToolUseExpected()
    if raw == "max_tokens":
        return MaxTokens()
    if raw == "pause_turn":
        # Anthropic streams ``pause_turn`` when the response paused mid-
        # turn (e.g. extended thinking budget, server-side checkpoint).
        # Caller must resend the same input to continue.
        return PauseTurn()
    return VendorSpecific(raw=raw)


def _extract_usage(message_obj: Any) -> Usage | None:
    """Pull ``Usage`` out of an Anthropic ``Message`` (or partial)."""

    raw = _optional_sdk_attr(message_obj, "usage")
    if raw is None:
        return None
    return Usage(
        input_tokens=_nonnegative_sdk_int(
            raw,
            "input_tokens",
            default=0,
        ),
        output_tokens=_nonnegative_sdk_int(
            raw,
            "output_tokens",
            default=0,
        ),
        cache_read=_nonnegative_sdk_int(
            raw,
            "cache_read_input_tokens",
            default=0,
        ),
        cache_write=_nonnegative_sdk_int(
            raw,
            "cache_creation_input_tokens",
            default=0,
        ),
    )


def _finalize_block(state: _StreamState, index: int) -> None:
    """Flush provider scratch for one Anthropic content block."""

    scratch = state.scratch.pop(index, None)
    if scratch is None:
        return
    kind = scratch.get("kind")
    if kind == "text":
        state.accumulator.add_text(index, scratch.get("text", ""))
    elif kind == "thinking":
        state.accumulator.add_thinking(index, scratch.get("text", ""))
        state.accumulator.set_thinking_signature(index, scratch.get("signature"))
    elif kind == "opaque_thinking":
        payload = scratch.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"Anthropic opaque thinking block at index {index} has no payload"
            )
        state.accumulator.add_opaque_thinking(
            index,
            provider="anthropic",
            payload=payload,
        )
    elif kind == "tool_use":
        tool_id = scratch.get("id")
        tool_name = scratch.get("name")
        partial_json = scratch.get("partial_json")
        if not isinstance(tool_id, str) or not tool_id:
            raise ValueError(f"Anthropic tool block at index {index} has no id")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(f"Anthropic tool block at index {index} has no name")
        if not isinstance(partial_json, str):
            raise TypeError(
                f"Anthropic tool arguments at index {index} must be a string"
            )
        state.accumulator.add_tool_call(
            id=tool_id,
            name=tool_name,
            args_delta=partial_json,
            index=index,
        )
    else:
        raise ValueError(
            f"unknown Anthropic scratch block kind at index {index}: {kind!r}"
        )


_SDK_MISSING = object()


def _optional_sdk_attr(value: object, name: str) -> object | None:
    item = getattr(  # code-health: ignore[AM021] -- Anthropic SDK model boundary
        value,
        name,
        _SDK_MISSING,
    )
    return None if item is _SDK_MISSING else item


def _required_sdk_attr(value: object, name: str) -> object:
    item = _optional_sdk_attr(value, name)
    if item is None:
        raise ValueError(f"Anthropic SDK field {name!r} is required")
    return item


def _optional_sdk_string(value: object, name: str) -> str | None:
    item = _optional_sdk_attr(value, name)
    if item is None:
        return None
    if not isinstance(item, str):
        raise TypeError(f"Anthropic SDK field {name!r} must be a string or None")
    return item


def _required_sdk_string(
    value: object,
    name: str,
    *,
    allow_empty: bool = True,
) -> str:
    item = _required_sdk_attr(value, name)
    if not isinstance(item, str) or (not allow_empty and not item):
        raise TypeError(f"Anthropic SDK field {name!r} must be a string")
    return item


def _nonnegative_sdk_int(
    value: object,
    name: str,
    *,
    default: int | None = None,
) -> int:
    item = _optional_sdk_attr(value, name)
    if item is None:
        if default is None:
            raise ValueError(f"Anthropic SDK field {name!r} is required")
        return default
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise TypeError(f"Anthropic SDK field {name!r} must be a non-negative integer")
    return item


@runtime_checkable
class _AnthropicAsyncStream(Protocol):
    def __aiter__(self) -> AsyncIterator[object]: ...

    async def close(self) -> None: ...


@runtime_checkable
class _AnthropicStreamContext(Protocol):
    async def __aenter__(self) -> object: ...

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> object: ...


# --- Public callable -------------------------------------------------------


@dataclass(slots=True)
class AnthropicStreamFn:
    """Kernel-compatible ``StreamFn`` backed by the Anthropic Messages API.

    Construction parameters:

    - ``api_key``: Anthropic API key. Defaults to ``ANTHROPIC_API_KEY``.
    - ``base_url``: optional override for the API base URL.
    - ``retry_policy``: optional policy used for retrying provider calls.
    - ``client``: pre-configured :class:`anthropic.AsyncAnthropic` instance.
      Tests may inject a mock here to bypass network entirely.
    """

    api_key: str | None = None
    base_url: str | None = None
    default_headers: Mapping[str, str] | None = None
    thinking_budgets: Mapping[str, int] | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None
    retry_policy: RetryPolicy | None = None
    client: AsyncAnthropic | None = None
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        budgets = {"low": 1_024, "medium": 4_096, "high": 16_384}
        if self.thinking_budgets is not None:
            unknown = set(self.thinking_budgets) - set(budgets)
            if unknown:
                raise ValueError(
                    "AnthropicStreamFn thinking_budgets has unknown levels: "
                    f"{sorted(unknown)}"
                )
            for key, value in self.thinking_budgets.items():
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(
                        "AnthropicStreamFn thinking budgets must be positive integers"
                    )
                budgets[key] = value
        self.thinking_budgets = budgets

    def _get_client(self) -> AsyncAnthropic:
        if self.client is not None:
            return self.client
        # Imported lazily so module import doesn't require the SDK to be
        # configured (e.g. in offline test environments using injected client).
        from anthropic import AsyncAnthropic as _AsyncAnthropic

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            # The SDK merges these last, so they override its built-in headers
            # (e.g. ``User-Agent``) — used to present as a Claude-Code client to
            # Anthropic-compatible "coding" endpoints that gate on it.
            kwargs["default_headers"] = dict(self.default_headers)
        # Without an explicit read timeout, a half-dead TCP connection
        # (server accepts then stops streaming mid-response) leaves the
        # request hanging forever — no exception is raised, so the retry
        # layer never fires and a synchronous audit child wedges
        # indefinitely. Mirror llm_openai: force a finite read timeout so
        # stalls surface as anthropic.APITimeoutError (retryable above).
        import httpx

        kwargs["timeout"] = httpx.Timeout(
            connect=30.0, read=180.0, write=60.0, pool=30.0
        )
        self.client = _AsyncAnthropic(**kwargs)
        return self.client

    def __call__(
        self,
        *,
        messages: list[AgentMessage],
        model: Model,
        tools: list[Tool],
        system: str | None = None,
        signal: CancelSignal | None = None,
        thinking: Literal["off", "low", "medium", "high"] = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
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
        thinking: Literal["off", "low", "medium", "high"],
    ) -> AsyncIterator[AssistantStreamEvent]:
        client = self._get_client()
        body: dict[str, Any] = {
            "model": model.id,
            "max_tokens": model.max_output_tokens,
            "messages": _to_anthropic_messages(messages),
        }
        if system is not None:
            body["system"] = system
        if tools:
            body["tools"] = _to_anthropic_tools(tools)
        if thinking != "off":
            assert self.thinking_budgets is not None
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budgets[thinking],
            }

        extra: dict[str, Any] = thaw_json(self.extra_body) if self.extra_body else {}  # type: ignore[assignment]
        if self.reasoning_effort is not None:
            extra.setdefault("output_config", {"effort": self.reasoning_effort})
        if extra:
            body["extra_body"] = extra

        state = _StreamState()
        aborted = False

        async def _open_stream() -> tuple[
            _AnthropicStreamContext, _AnthropicAsyncStream
        ]:
            ctx = client.messages.stream(**body)
            if not isinstance(ctx, _AnthropicStreamContext):
                raise TypeError("Anthropic client must return an async stream context")
            opened = await ctx.__aenter__()
            if not isinstance(opened, _AnthropicAsyncStream):
                raise TypeError(
                    "Anthropic stream context must yield an async iterable "
                    "stream with an async close() method"
                )
            return ctx, opened

        stream_ctx: _AnthropicStreamContext | None = None
        try:
            open_operation = (
                _open_stream()
                if self.retry_policy is None
                else self.retry_policy.run(
                    _open_stream,
                    is_retryable=_is_anthropic_retryable,
                )
            )
            opened_ctx, opened_stream = await await_with_cancel_signal(
                open_operation,
                signal,
            )
            stream_ctx = opened_ctx
            iterator = opened_stream.__aiter__()
            while True:
                try:
                    event = await await_with_cancel_signal(
                        iterator.__anext__(),
                        signal,
                    )
                except StopAsyncIteration:
                    break
                except OperationCancelledBySignal:
                    aborted = True
                    try:
                        await opened_stream.close()
                    except Exception:
                        # Best-effort close; do not let cleanup mask the abort.
                        logger.opt(exception=True).debug(
                            "anthropic: error while closing aborted stream"
                        )
                    break
                async for kernel_event in _translate_event(event, state):
                    yield kernel_event
        except OperationCancelledBySignal:
            aborted = True
        finally:
            if stream_ctx is not None:
                try:
                    await stream_ctx.__aexit__(None, None, None)
                except Exception:
                    if not aborted:
                        raise
                    logger.opt(exception=True).debug(
                        "anthropic: error while closing aborted stream context"
                    )

        if aborted:
            state.stop_reason = "aborted"
            state.termination = Aborted()
            # Flush any in-flight blocks so the assembled message is consistent.
            for idx in sorted(state.scratch.keys()):
                _finalize_block(state, idx)

        assembled = state.accumulator.assemble(
            stop_reason=state.stop_reason,
            termination=state.termination,
            usage=state.usage,
            timestamp=self.clock(),
        )
        for parse_error in state.accumulator.parse_errors:
            yield parse_error
        yield MessageEnd(message=assembled)


async def _translate_event(
    event: Any,
    state: _StreamState,
) -> AsyncIterator[AssistantStreamEvent]:
    """Translate one raw Anthropic event into 0+ kernel stream events.

    Detection is by the ``type`` string + structural attribute access, so
    tests can drive this with lightweight stand-ins instead of real SDK
    Pydantic models.
    """

    etype = _required_sdk_string(event, "type", allow_empty=False)

    if etype == "message_start":
        message = _required_sdk_attr(event, "message")
        usage = _extract_usage(message)
        if usage is not None:
            state.usage = usage
        return

    if etype == "content_block_start":
        index = _nonnegative_sdk_int(event, "index")
        if index in state.scratch:
            raise ValueError(f"Anthropic content block index {index} started twice")
        block = _required_sdk_attr(event, "content_block")
        block_type = _required_sdk_string(
            block,
            "type",
            allow_empty=False,
        )
        if block_type == "text":
            state.scratch[index] = {"kind": "text", "text": ""}
        elif block_type == "thinking":
            state.scratch[index] = {
                "kind": "thinking",
                "text": _optional_sdk_string(block, "thinking") or "",
                "signature": _optional_sdk_string(block, "signature"),
            }
        elif block_type == "redacted_thinking":
            state.scratch[index] = {
                "kind": "opaque_thinking",
                "payload": {
                    "type": "redacted_thinking",
                    "data": _required_sdk_string(
                        block,
                        "data",
                        allow_empty=False,
                    ),
                },
            }
        elif block_type == "tool_use":
            tool_id = _required_sdk_string(
                block,
                "id",
                allow_empty=False,
            )
            tool_name = _required_sdk_string(
                block,
                "name",
                allow_empty=False,
            )
            state.scratch[index] = {
                "kind": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "partial_json": "",
            }
            yield ToolCallStart(id=tool_id, name=tool_name)
        else:
            raise ValueError(
                f"Anthropic content block type is not modeled by AgentM: {block_type!r}"
            )
        return

    if etype == "content_block_delta":
        index = _nonnegative_sdk_int(event, "index")
        delta = _required_sdk_attr(event, "delta")
        delta_type = _required_sdk_string(
            delta,
            "type",
            allow_empty=False,
        )
        scratch = state.scratch.get(index)
        if delta_type == "text_delta":
            if scratch is None or scratch.get("kind") != "text":
                raise ValueError(
                    f"Anthropic text delta has no text block at index {index}"
                )
            text = _required_sdk_string(delta, "text")
            scratch["text"] = scratch.get("text", "") + text
            yield TextDelta(text=text)
        elif delta_type == "input_json_delta":
            if scratch is None or scratch.get("kind") != "tool_use":
                raise ValueError(
                    f"Anthropic input JSON delta has no tool block at index {index}"
                )
            partial = _required_sdk_string(delta, "partial_json")
            scratch["partial_json"] = scratch.get("partial_json", "") + partial
            yield ToolCallArgsDelta(
                id=scratch["id"],
                args_json_delta=partial,
            )
        elif delta_type == "thinking_delta":
            if scratch is None or scratch.get("kind") != "thinking":
                raise ValueError(
                    f"Anthropic thinking delta has no thinking block at index {index}"
                )
            text = _required_sdk_string(delta, "thinking")
            scratch["text"] = scratch.get("text", "") + text
            yield ThinkingDelta(text=text, signature=None)
        elif delta_type == "signature_delta":
            if scratch is None or scratch.get("kind") != "thinking":
                raise ValueError(
                    f"Anthropic signature delta has no thinking block at index {index}"
                )
            sig = _required_sdk_string(delta, "signature")
            # Multiple signature deltas are valid; preserve stream order.
            prev = scratch.get("signature") or ""
            scratch["signature"] = prev + sig
        elif delta_type == "citations_delta":
            raise ValueError("Anthropic citation deltas are not modeled by AgentM")
        else:
            raise ValueError(f"unknown Anthropic content delta type: {delta_type!r}")
        return

    if etype == "content_block_stop":
        index = _nonnegative_sdk_int(event, "index")
        scratch = state.scratch.get(index)
        if scratch is None:
            raise ValueError(
                f"Anthropic content block stop has no start at index {index}"
            )
        kind = scratch.get("kind") if scratch is not None else None
        _finalize_block(state, index)
        if kind == "tool_use" and scratch is not None:
            yield ToolCallEnd(id=scratch["id"])
        return

    if etype == "message_delta":
        delta = _required_sdk_attr(event, "delta")
        raw_stop = _optional_sdk_string(delta, "stop_reason")
        if raw_stop is not None:
            state.stop_reason = raw_stop
            state.termination = _map_stop_reason(raw_stop)
        # Anthropic emits a final usage update on message_delta.
        raw_usage = _optional_sdk_attr(event, "usage")
        if raw_usage is not None:
            existing = state.usage
            state.usage = Usage(
                input_tokens=_nonnegative_sdk_int(
                    raw_usage,
                    "input_tokens",
                    default=existing.input_tokens if existing else 0,
                ),
                output_tokens=_nonnegative_sdk_int(
                    raw_usage,
                    "output_tokens",
                    default=existing.output_tokens if existing else 0,
                ),
                cache_read=_nonnegative_sdk_int(
                    raw_usage,
                    "cache_read_input_tokens",
                    default=existing.cache_read if existing else 0,
                ),
                cache_write=_nonnegative_sdk_int(
                    raw_usage,
                    "cache_creation_input_tokens",
                    default=existing.cache_write if existing else 0,
                ),
            )
        return

    if etype == "message_stop":
        # Some streams flush remaining blocks here; make sure scratch is empty.
        for idx in sorted(state.scratch.keys()):
            _finalize_block(state, idx)
        return

    raise ValueError(f"unknown Anthropic stream event type: {etype!r}")


# --- Extension entrypoint --------------------------------------------------


class _AnthropicProviderRuntime:
    """Install-time provider registration runtime for Anthropic-compatible models."""

    def __init__(self, session: Any, config: LlmAnthropicConfig) -> None:
        self._session = session
        self._config = config

    def install(self) -> None:
        model_id = self._model_id()
        stream_fn = self._build_stream_fn()
        name = self._provider_name()
        model = _build_model(
            model_id,
            provider=name,
            **self._model_kwargs(),
        )
        if self._session.has_provider(name):
            raise ValueError(
                "agentm.extensions.builtin.llm_anthropic.install: provider "
                f"{name!r} is already registered in this session."
            )
        self._session.register_provider(
            name,
            ProviderConfig(stream_fn=stream_fn, model=model, name=name),
        )

    def _model_id(self) -> str:
        model_id = self._config.model
        if not model_id or not isinstance(model_id, str):
            raise ValueError(
                "agentm.extensions.builtin.llm_anthropic.install: config.model is required and must "
                "be a non-empty string (e.g. 'claude-opus-4-7')."
            )
        return model_id

    def _build_stream_fn(self) -> AnthropicStreamFn:
        default_headers = self._config.default_headers
        if default_headers is not None and not isinstance(default_headers, Mapping):
            raise ValueError(
                "agentm.extensions.builtin.llm_anthropic.install: config.default_headers "
                "must be a mapping of header name to string value."
            )
        return AnthropicStreamFn(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
            default_headers=default_headers,
            thinking_budgets=self._config.thinking_budgets,
            reasoning_effort=self._config.reasoning_effort,
            extra_body=self._config.extra_body,
            retry_policy=self._session.services.get(RETRY_POLICY_SERVICE),
        )

    def _provider_name(self) -> str:
        name = self._config.name or "anthropic"
        if not isinstance(name, str) or not name:
            raise ValueError(
                "agentm.extensions.builtin.llm_anthropic.install: "
                "config.name must be a non-empty string"
            )
        return name

    def _model_kwargs(self) -> dict[str, int]:
        # Optional model-spec overrides; defaults handled in ``_build_model``.
        model_kwargs: dict[str, int] = {}
        if self._config.context_window is not None:
            model_kwargs["context_window"] = self._config.context_window
        if self._config.max_output_tokens is not None:
            model_kwargs["max_output_tokens"] = self._config.max_output_tokens
        return model_kwargs


def install(session: Any, config: LlmAnthropicConfig) -> None:
    """Provider extension entrypoint.

    Reads ``config.model`` (required), optional ``config.api_key`` and
    ``config.base_url``, then registers the resulting ``AnthropicStreamFn``
    on the given :class:`AtomAPI` under the name ``"anthropic"``.

    """

    _AnthropicProviderRuntime(session, config).install()


__all__ = ("AnthropicStreamFn", "MANIFEST", "install")
