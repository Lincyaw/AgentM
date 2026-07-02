"""Anthropic Messages API provider — native ``StreamFn`` implementation.

This module is the first concrete LLM provider for AgentM v2. It plugs into
the kernel via the ``StreamFn`` Protocol described in
``.claude/designs/pluggable-architecture.md`` §3.1 and is loaded through the
extension mechanism described in ``.claude/designs/extension-as-scenario.md``
§7 (LLM providers as extensions).

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

from __future__ import annotations

import asyncio
import base64
from loguru import logger
import os
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from agentm.core.abi import (
    Aborted,
    AgentMessage,
    AssistantContent,
    AssistantMessage,
    AssistantStreamEvent,
    EndTurn,
    ImageContent,
    MaxTokens,
    MessageEnd,
    Model,
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
from pydantic import BaseModel, ConfigDict

from agentm.extensions import ExtensionManifest

from agentm.core.abi import RETRY_POLICY_SERVICE
from agentm.core.lib import StreamAccumulator, ToolSpecAdapter, encode_tool_args

if TYPE_CHECKING:  # pragma: no cover - import only used for type hints
    from anthropic import AsyncAnthropic


class LlmAnthropicConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "claude-sonnet-4-6"
    api_key: str | None = None
    base_url: str | None = None
    default_headers: dict[str, str] | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    thinking_budgets: dict[str, int] | None = None

MANIFEST = ExtensionManifest(
    name="llm_anthropic",
    description="Register an Anthropic Messages API LLM stream provider.",
    registers=("provider:anthropic",),
    config_schema=LlmAnthropicConfig,
    requires=("retry_policy",),
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
        if isinstance((err_type := getattr(anthropic, name, None)), type)
    )
    return bool(retryable_types) and isinstance(exc, retryable_types)

class _IdentityRetryPolicy:
    async def run(
        self,
        fn: Callable[[], Any],
        *,
        is_retryable: Callable[[BaseException], bool],
    ) -> Any:
        del is_retryable
        return await fn()

# --- Model registry ---------------------------------------------------------

def _build_model(
    model_id: str,
    *,
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
        provider="anthropic",
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

@dataclass(frozen=True)
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

    raw = getattr(message_obj, "usage", None)
    if raw is None:
        return None
    return Usage(
        input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
        cache_read=int(getattr(raw, "cache_read_input_tokens", 0) or 0),
        cache_write=int(getattr(raw, "cache_creation_input_tokens", 0) or 0),
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
    elif kind == "tool_use":
        state.accumulator.add_tool_call(
            id=scratch.get("id", ""),
            name=scratch.get("name", ""),
            args_delta=scratch.get("partial_json", ""),
            index=index,
        )

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
            budgets.update(
                {
                    str(key): int(value)
                    for key, value in self.thinking_budgets.items()
                }
            )
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
        signal: asyncio.Event | None = None,
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
        signal: asyncio.Event | None,
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

        extra = dict(self.extra_body or {})
        if self.reasoning_effort is not None:
            extra.setdefault("output_config", {"effort": self.reasoning_effort})
        if extra:
            body["extra_body"] = extra

        state = _StreamState()
        aborted = False

        retry_policy = self.retry_policy or _IdentityRetryPolicy()

        async def _open_stream() -> tuple[Any, Any]:
            ctx = client.messages.stream(**body)
            stream = await ctx.__aenter__()
            return ctx, stream

        stream_ctx, stream = await retry_policy.run(
            _open_stream,
            is_retryable=_is_anthropic_retryable,
        )
        try:
            async for event in stream:
                if signal is not None and signal.is_set():
                    aborted = True
                    try:
                        await stream.close()
                    except Exception:
                        # Best-effort close; do not let cleanup mask the abort.
                        logger.opt(exception=True).debug(
                            "anthropic: error while closing aborted stream"
                        )
                    break
                async for kernel_event in _translate_event(event, state):
                    yield kernel_event
        finally:
            await stream_ctx.__aexit__(None, None, None)

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

    etype = getattr(event, "type", None)

    if etype == "message_start":
        message = getattr(event, "message", None)
        if message is not None:
            usage = _extract_usage(message)
            if usage is not None:
                state.usage = usage
        return

    if etype == "content_block_start":
        index = int(getattr(event, "index", 0))
        block = getattr(event, "content_block", None)
        block_type = getattr(block, "type", None)
        if block_type == "text":
            state.scratch[index] = {"kind": "text", "text": ""}
        elif block_type == "thinking":
            state.scratch[index] = {
                "kind": "thinking",
                "text": getattr(block, "thinking", "") or "",
                "signature": getattr(block, "signature", None),
            }
        elif block_type == "tool_use":
            tool_id = getattr(block, "id", "") or ""
            tool_name = getattr(block, "name", "") or ""
            state.scratch[index] = {
                "kind": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "partial_json": "",
            }
            yield ToolCallStart(id=tool_id, name=tool_name)
        return

    if etype == "content_block_delta":
        index = int(getattr(event, "index", 0))
        delta = getattr(event, "delta", None)
        delta_type = getattr(delta, "type", None)
        scratch = state.scratch.get(index)
        if delta_type == "text_delta":
            text = getattr(delta, "text", "") or ""
            if scratch is not None and scratch.get("kind") == "text":
                scratch["text"] = scratch.get("text", "") + text
            yield TextDelta(text=text)
        elif delta_type == "input_json_delta":
            partial = getattr(delta, "partial_json", "") or ""
            if scratch is not None and scratch.get("kind") == "tool_use":
                scratch["partial_json"] = scratch.get("partial_json", "") + partial
                yield ToolCallArgsDelta(
                    id=scratch.get("id", ""),
                    args_json_delta=partial,
                )
        elif delta_type == "thinking_delta":
            text = getattr(delta, "thinking", "") or ""
            if scratch is not None and scratch.get("kind") == "thinking":
                scratch["text"] = scratch.get("text", "") + text
            yield ThinkingDelta(text=text, signature=None)
        elif delta_type == "signature_delta":
            sig = getattr(delta, "signature", None)
            if scratch is not None and scratch.get("kind") == "thinking":
                # Anthropic delivers signatures as a single delta; concatenate
                # defensively in case multiple are sent.
                prev = scratch.get("signature") or ""
                scratch["signature"] = (prev + sig) if sig is not None else prev
        return

    if etype == "content_block_stop":
        index = int(getattr(event, "index", 0))
        scratch = state.scratch.get(index)
        kind = scratch.get("kind") if scratch is not None else None
        _finalize_block(state, index)
        if kind == "tool_use" and scratch is not None:
            yield ToolCallEnd(id=scratch.get("id", ""))
        return

    if etype == "message_delta":
        delta = getattr(event, "delta", None)
        raw_stop = getattr(delta, "stop_reason", None) if delta is not None else None
        if raw_stop is not None:
            state.stop_reason = raw_stop
            state.termination = _map_stop_reason(raw_stop)
        # Anthropic emits a final usage update on message_delta.
        usage = getattr(event, "usage", None)
        if usage is not None:
            existing = state.usage
            state.usage = Usage(
                input_tokens=int(
                    getattr(usage, "input_tokens", None)
                    or (existing.input_tokens if existing else 0)
                ),
                output_tokens=int(
                    getattr(usage, "output_tokens", None)
                    or (existing.output_tokens if existing else 0)
                ),
                cache_read=int(
                    getattr(usage, "cache_read_input_tokens", None)
                    or (existing.cache_read if existing else 0)
                ),
                cache_write=int(
                    getattr(usage, "cache_creation_input_tokens", None)
                    or (existing.cache_write if existing else 0)
                ),
            )
        return

    if etype == "message_stop":
        # Some streams flush remaining blocks here; make sure scratch is empty.
        for idx in sorted(state.scratch.keys()):
            _finalize_block(state, idx)
        return

    # Unknown events are ignored on purpose; the SDK occasionally adds new ones.
    return

# --- Extension entrypoint --------------------------------------------------

def install(api: Any, config: LlmAnthropicConfig) -> None:
    """Provider extension entrypoint.

    Reads ``config.model`` (required), optional ``config.api_key`` and
    ``config.base_url``, then registers the resulting ``AnthropicStreamFn``
    on the given :class:`ExtensionAPI` under the name ``"anthropic"``.

    """

    model_id = config.model
    if not model_id or not isinstance(model_id, str):
        raise ValueError(
            "agentm.extensions.builtin.llm_anthropic.install: config.model is required and must "
            "be a non-empty string (e.g. 'claude-opus-4-7')."
        )

    api_key = config.api_key
    base_url = config.base_url
    default_headers = config.default_headers
    if default_headers is not None and not isinstance(default_headers, Mapping):
        raise ValueError(
            "agentm.extensions.builtin.llm_anthropic.install: config.default_headers "
            "must be a mapping of header name to string value."
        )
    # Access extra fields from the Pydantic model for pass-through config
    extra = config.model_extra or {}
    stream_fn = AnthropicStreamFn(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
        thinking_budgets=config.thinking_budgets,
        reasoning_effort=extra.get("reasoning_effort"),
        extra_body=extra.get("extra_body"),
        retry_policy=api.get_service(RETRY_POLICY_SERVICE),
    )
    # Optional model-spec overrides; defaults handled in ``_build_model``.
    model_kwargs: dict[str, int] = {}
    if config.context_window is not None:
        model_kwargs["context_window"] = config.context_window
    if config.max_output_tokens is not None:
        model_kwargs["max_output_tokens"] = config.max_output_tokens
    model = _build_model(model_id, **model_kwargs)

    api.register_provider(
        "anthropic",
        ProviderConfig(stream_fn=stream_fn, model=model, name="anthropic"),
    )

__all__ = ("AnthropicStreamFn", "MANIFEST", "install")
