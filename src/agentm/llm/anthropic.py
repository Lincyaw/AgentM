"""Anthropic Messages API provider — native ``StreamFn`` implementation.

This module is the first concrete LLM provider for AgentM v2. It plugs into
the kernel via the ``StreamFn`` Protocol described in
``.claude/designs/pluggable-architecture.md`` §3.1 and is loaded through the
extension mechanism described in ``.claude/designs/extension-as-scenario.md``
§7 (LLM providers as extensions).

Boundaries:

* The kernel layer (``agentm.core.kernel``) is the only AgentM dependency
  imported at module import time. The harness layer is **not** imported here;
  ``ProviderConfig`` is resolved lazily inside :func:`install` so this module
  remains usable in isolation (e.g. notebook embedding, harness-less tests).
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
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from agentm.core.kernel.messages import (
    AgentMessage,
    AssistantContent,
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    Usage,
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
from agentm.core.kernel.tool import Tool

if TYPE_CHECKING:  # pragma: no cover - import only used for type hints
    from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


# --- Model registry ---------------------------------------------------------


# Hard-coded defaults for known Anthropic models. Unknown ids fall back to
# (200_000, 8_192) and emit a warning.
_KNOWN_MODELS: dict[str, tuple[int, int]] = {
    "claude-opus-4-7": (200_000, 32_768),
    "claude-sonnet-4-6": (200_000, 64_000),
    "claude-haiku-4-5-20251001": (200_000, 8_192),
}


def _build_model(model_id: str) -> Model:
    """Construct a kernel ``Model`` descriptor for the given Anthropic id."""

    if model_id in _KNOWN_MODELS:
        ctx, max_out = _KNOWN_MODELS[model_id]
    else:
        logger.warning(
            "anthropic: unknown model id %r; falling back to context_window=200000, "
            "max_output_tokens=8192",
            model_id,
        )
        ctx, max_out = 200_000, 8_192
    return Model(
        id=model_id,
        provider="anthropic",
        context_window=ctx,
        max_output_tokens=max_out,
    )


_THINKING_BUDGETS: dict[str, int] = {
    "low": 1_024,
    "medium": 4_096,
    "high": 16_384,
}


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
    for msg in messages:
        if isinstance(msg, UserMessage):
            out.append(
                {"role": "user", "content": _encode_user_content(list(msg.content))}
            )
        elif isinstance(msg, AssistantMessage):
            out.append(
                {
                    "role": "assistant",
                    "content": _encode_assistant_content(list(msg.content)),
                }
            )
        elif isinstance(msg, ToolResultMessage):
            blocks = [_encode_tool_result_block(b) for b in msg.content]
            # Pack into the previous message if it was also a tool-result-only
            # user message, matching Anthropic's "results from one turn live in
            # one user message" convention.
            if (
                out
                and out[-1]["role"] == "user"
                and all(
                    isinstance(c, dict) and c.get("type") == "tool_result"
                    for c in out[-1]["content"]
                )
            ):
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": "user", "content": blocks})
        else:  # pragma: no cover - exhaustive
            raise TypeError(f"unsupported message type: {type(msg)!r}")
    return out


def _to_anthropic_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert kernel ``Tool``s to Anthropic tool definitions.

    The kernel already exposes ``parameters`` as a JSON Schema dict, so this is
    a direct field rename to ``input_schema``.
    """

    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


# --- Streaming bridge -------------------------------------------------------


@dataclass
class _StreamState:
    """Mutable accumulator for one turn of streaming.

    Tracks per-content-block state so we can assemble the final
    :class:`AssistantMessage` from incremental events.
    """

    content_blocks: list[AssistantContent] = field(default_factory=list)
    # Per-block scratch keyed by Anthropic content_block index.
    scratch: dict[int, dict[str, Any]] = field(default_factory=dict)
    usage: Usage | None = None
    stop_reason: (
        Literal["end_turn", "tool_use", "max_tokens", "error", "aborted"] | None
    ) = None


_STOP_REASON_MAP: dict[str, Literal["end_turn", "tool_use", "max_tokens", "error"]] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
}


def _map_stop_reason(
    raw: str | None,
) -> Literal["end_turn", "tool_use", "max_tokens", "error"] | None:
    if raw is None:
        return None
    return _STOP_REASON_MAP.get(raw, "error")


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
    """Materialize the per-index scratch into a final ``AssistantContent``."""

    scratch = state.scratch.pop(index, None)
    if scratch is None:
        return
    kind = scratch.get("kind")
    if kind == "text":
        state.content_blocks.append(
            TextContent(type="text", text=scratch.get("text", ""))
        )
    elif kind == "thinking":
        state.content_blocks.append(
            ThinkingBlock(
                type="thinking",
                text=scratch.get("text", ""),
                signature=scratch.get("signature"),
            )
        )
    elif kind == "tool_use":
        import json

        raw_args = scratch.get("partial_json", "")
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            # Fall back to empty dict; the loop layer is responsible for
            # surfacing argument-parse errors back to the model. This keeps
            # the kernel's invariant that ``arguments`` is a parsed dict.
            args = {}
        if not isinstance(args, dict):
            args = {}
        state.content_blocks.append(
            ToolCallBlock(
                type="tool_call",
                id=scratch.get("id", ""),
                name=scratch.get("name", ""),
                arguments=args,
            )
        )


# --- Public callable -------------------------------------------------------


@dataclass
class AnthropicStreamFn:
    """Kernel-compatible ``StreamFn`` backed by the Anthropic Messages API.

    Construction parameters:

    - ``api_key``: Anthropic API key. Defaults to ``ANTHROPIC_API_KEY``.
    - ``base_url``: optional override for the API base URL.
    - ``client``: pre-configured :class:`anthropic.AsyncAnthropic` instance.
      Tests may inject a mock here to bypass network entirely.
    """

    api_key: str | None = None
    base_url: str | None = None
    client: AsyncAnthropic | None = None

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
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": _THINKING_BUDGETS[thinking],
            }

        state = _StreamState()
        aborted = False

        stream_ctx = client.messages.stream(**body)
        async with stream_ctx as stream:
            async for event in stream:
                if signal is not None and signal.is_set():
                    aborted = True
                    try:
                        await stream.close()
                    except Exception:
                        # Best-effort close; do not let cleanup mask the abort.
                        logger.debug(
                            "anthropic: error while closing aborted stream",
                            exc_info=True,
                        )
                    break
                async for kernel_event in _translate_event(event, state):
                    yield kernel_event

        if aborted:
            state.stop_reason = "aborted"
            # Flush any in-flight blocks so the assembled message is consistent.
            for idx in sorted(state.scratch.keys()):
                _finalize_block(state, idx)

        assembled = AssistantMessage(
            role="assistant",
            content=list(state.content_blocks),
            timestamp=0.0,
            stop_reason=state.stop_reason,
            usage=state.usage,
        )
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
        mapped = _map_stop_reason(raw_stop)
        if mapped is not None:
            state.stop_reason = mapped
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


__all__ = ["AnthropicStreamFn"]
