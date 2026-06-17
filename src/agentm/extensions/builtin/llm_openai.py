"""OpenAI Chat Completions API provider — native ``StreamFn`` implementation.

Sibling of :mod:`agentm.extensions.builtin.llm_anthropic` for OpenAI-compatible endpoints (the
official OpenAI API, plus proxies that speak the same protocol: LiteLLM,
DeepSeek, Doubao Ark, Together, Fireworks, vLLM, Ollama, …). Plugs into the
kernel via the ``StreamFn`` Protocol described in
``.claude/designs/pluggable-architecture.md`` §3.1 and is loaded through the
extension mechanism described in ``.claude/designs/extension-as-scenario.md``
§7 (LLM providers as extensions).

Boundaries:

* The kernel layer (``agentm.core.abi``) is the only AgentM dependency
  imported at module import time. The harness layer is **not** imported here.
* Streaming is delegated to the official ``openai`` Python SDK
  (:class:`openai.AsyncOpenAI`); we never reimplement HTTP/SSE.

Conversion layout:

* ``_to_openai_messages``: kernel ``AgentMessage`` list → API messages dict.
* ``_to_openai_tools``: kernel ``Tool`` list → API tools dict.
* ``OpenAIStreamFn.__call__``: drives the SDK stream and translates each
  raw chunk into a kernel ``AssistantStreamEvent``.

Self-signed proxies (Warpgate ticket gateway, internal LiteLLM behind a
self-signed cert): pass ``verify_ssl=False`` and/or ``default_query`` (e.g.
``{"warpgate-ticket": "..."}``) via config — both are forwarded to the
underlying ``httpx.AsyncClient`` so the OpenAI SDK never has to know.
"""

from __future__ import annotations

import asyncio
import base64
import copy
from loguru import logger
import os
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from agentm.core.abi import (
    Aborted,
    AgentMessage,
    AssistantMessage,
    AssistantStreamEvent,
    DiagnosticEvent,
    EndTurn,
    EventBus,
    ImageContent,
    MaxTokens,
    MessageEnd,
    Model,
    PauseTurn,
    ProviderConfig,
    ProviderError,
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

from agentm.core.lib import StreamAccumulator, ToolSpecAdapter, encode_tool_args
from agentm.core.lib.tool_schema import _force_strict


class LlmOpenaiConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    name: str | None = None
    default_query: dict[str, Any] | None = None
    default_headers: dict[str, Any] | None = None
    verify_ssl: bool | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    thinking_round_trip: Literal["drop", "system_note", "raise"] | None = None
    azure_endpoint: str | None = None
    api_version: str | None = None

MANIFEST = ExtensionManifest(
    name="llm_openai",
    description="Register an OpenAI Chat Completions API LLM stream provider.",
    registers=("provider:openai",),
    config_schema=LlmOpenaiConfig,
    requires=("retry_policy",),
)

def _is_openai_retryable(exc: BaseException) -> bool:
    try:
        import openai
    except ImportError:  # pragma: no cover - SDK dependency is optional here
        return False
    # APIConnectionError / APITimeoutError surface read-timeouts and
    # half-dead TCP — without retry these propagate up and waste the
    # whole rollout. Treat them like 429s.
    retryable_types = tuple(
        err_type
        for name in ("RateLimitError", "APIConnectionError", "APITimeoutError")
        if isinstance((err_type := getattr(openai, name, None)), type)
    )
    if retryable_types and isinstance(exc, retryable_types):
        return True
    # Doubao / LiteLLM XGrammar JIT compiles the tool schema for
    # constrained decoding. The compile is non-deterministic: identical
    # payloads succeed most of the time and fail ~10-30% with a 400
    # carrying ``Invalid decoding guidance syntax`` (the error string
    # also contains ``json_schema_converter``). Retrying the same
    # request typically succeeds. This is a backend issue we can't fix
    # client-side, but we can stop it from killing whole rollouts.
    bad_request = getattr(openai, "BadRequestError", None)
    if isinstance(bad_request, type) and isinstance(exc, bad_request):
        message = str(exc)
        if (
            "Invalid decoding guidance syntax" in message
            or "json_schema_converter" in message
        ):
            return True
    return False

# Keywords that XGrammar-based constrained-decoding engines (Volcengine Ark,
# vLLM, SGLang) cannot compile into an EBNF grammar. Leaving them in the
# schema triggers ``Invalid decoding guidance syntax`` 400 errors. Safe to
# strip: they are validation constraints, not structural schema description;
# the LLM generates conforming output from the property descriptions alone.
_XGRAMMAR_UNSUPPORTED = frozenset({
    "minItems", "maxItems",
    "minLength", "maxLength",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "prefixItems",
})

def _strip_validation_constraints(node: Any) -> None:
    """Remove XGrammar-unsupported keywords from a schema in-place."""
    if isinstance(node, dict):
        for key in _XGRAMMAR_UNSUPPORTED & node.keys():
            del node[key]
        for value in node.values():
            _strip_validation_constraints(value)
    elif isinstance(node, list):
        for value in node:
            _strip_validation_constraints(value)

def _default_httpx_client(*, verify: bool) -> Any:
    import httpx

    # Without explicit read timeout, a half-dead TCP connection (server
    # crash, LB drops idle conn, NAT entry expired) leaves the request
    # hanging forever — the retry layer never fires because no exception
    # is raised. Force a finite read timeout so hangs surface as errors
    # and fall into _create_with_retry.
    timeout = httpx.Timeout(
        connect=30.0,
        read=180.0,
        write=60.0,
        pool=30.0,
    )
    return httpx.AsyncClient(verify=verify, timeout=timeout)

# --- Model registry ---------------------------------------------------------

def _build_model(
    model_id: str,
    *,
    context_window: int = 128_000,
    max_output_tokens: int = 4_096,
) -> Model:
    """Construct a kernel ``Model`` descriptor for the given OpenAI-compat id.

    ``context_window`` and ``max_output_tokens`` come from caller config —
    we do not maintain a hard-coded model table since OpenAI-compatible
    proxies (LiteLLM, DeepSeek, Doubao, Kimi …) ship arbitrary ids that no
    static table can keep up with. Defaults are conservative: 128K/4096
    works for most modern models; override via ``config['context_window']``
    / ``config['max_output_tokens']`` when the deployed model differs.
    """

    return Model(
        id=model_id,
        provider="openai",
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )

# --- Message / tool serialization ------------------------------------------

def _encode_image(image: ImageContent) -> dict[str, Any]:
    """Convert kernel ``ImageContent`` to an OpenAI multimodal content part.

    OpenAI accepts data URLs in ``image_url.url``; we inline the bytes.
    """

    encoded = base64.b64encode(image.data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
    }

def _encode_user_content(
    blocks: list[TextContent | ImageContent],
) -> str | list[dict[str, Any]]:
    """Encode user-side content. Plain text collapses to a string; mixed or
    image-bearing content becomes the OpenAI list-of-parts shape.
    """

    if all(isinstance(b, TextContent) for b in blocks):
        return "".join(b.text for b in blocks if isinstance(b, TextContent))
    parts: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            parts.append(_encode_image(block))
        else:  # pragma: no cover - exhaustive over the union
            raise TypeError(f"unexpected user content type: {type(block)!r}")
    return parts

def _encode_assistant_message(
    msg: AssistantMessage,
    *,
    thinking_round_trip: Literal["drop", "system_note", "raise"] = "drop",
    on_drop: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Encode an assistant turn as a single OpenAI assistant message.

    Text blocks concatenate into ``content``; ``ToolCallBlock``s flatten
    into a ``tool_calls`` array. ``ThinkingBlock`` handling is controlled by
    ``thinking_round_trip`` because chat completions has no native slot for
    assistant reasoning.
    """

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            text_parts.append(block.text)
        elif isinstance(block, ToolCallBlock):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": OpenAIToolSpecAdapter().encode_tool_args(block.arguments),
                    },
                }
            )
        elif isinstance(block, ThinkingBlock):
            if thinking_round_trip == "raise":
                raise ValueError(
                    "OpenAIStreamFn cannot encode ThinkingBlock content with "
                    "thinking_round_trip='raise'."
                )
            if thinking_round_trip == "system_note":
                thinking_parts.append(block.text)
            else:
                if on_drop is not None:
                    on_drop()
                continue
        else:  # pragma: no cover
            raise TypeError(f"unexpected assistant content type: {type(block)!r}")
    out: dict[str, Any] = {"role": "assistant"}
    if text_parts:
        out["content"] = "".join(text_parts)
    elif not tool_calls:
        # Empty assistant turn: keep ``content`` so the API has something
        # non-null to validate against. Some servers reject role-only msgs.
        out["content"] = ""
    if tool_calls:
        out["tool_calls"] = tool_calls
    if thinking_parts:
        note = {
            "role": "system",
            "content": "Previous assistant reasoning: " + "\n".join(thinking_parts),
        }
        return [note, out]
    return [out]

def _encode_tool_result_block(block: ToolResultBlock) -> dict[str, Any]:
    """One ``ToolResultBlock`` → one OpenAI ``tool``-role message.

    OpenAI's tool result message takes a single ``content`` field per
    ``tool_call_id``. We collapse text parts and drop image parts (the
    chat completions protocol does not allow images in tool results).
    """

    text = "".join(b.text for b in block.content if isinstance(b, TextContent))
    return {
        "role": "tool",
        "tool_call_id": block.tool_call_id,
        "content": text,
    }

def _to_openai_messages(
    messages: list[AgentMessage],
    *,
    system: str | None,
    thinking_round_trip: Literal["drop", "system_note", "raise"] = "drop",
    on_thinking_drop: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Convert kernel messages to the OpenAI Chat Completions request shape.

    ``system`` is prepended as a single ``system``-role message if provided.
    ``ToolResultMessage``s are *not* packed (unlike Anthropic): OpenAI wants
    one ``tool``-role message per ``tool_call_id``.
    """

    out: list[dict[str, Any]] = []
    if system is not None and system.strip():
        out.append({"role": "system", "content": system})
    for msg in messages:
        if isinstance(msg, UserMessage):
            out.append(
                {"role": "user", "content": _encode_user_content(list(msg.content))}
            )
        elif isinstance(msg, AssistantMessage):
            out.extend(
                _encode_assistant_message(
                    msg,
                    thinking_round_trip=thinking_round_trip,
                    on_drop=on_thinking_drop,
                )
            )
        elif isinstance(msg, ToolResultMessage):
            for block in msg.content:
                out.append(_encode_tool_result_block(block))
        else:  # pragma: no cover - exhaustive
            raise TypeError(f"unsupported message type: {type(msg)!r}")
    return out

@dataclass(frozen=True)
class OpenAIToolSpecAdapter(ToolSpecAdapter):
    """Convert AgentM tools to OpenAI function-tool specs.

    ``strict=True`` (the default) adds ``additionalProperties: false``
    and forces all properties into ``required`` on every nested object —
    required by OpenAI's structured-outputs strict mode. Set to ``False``
    for OpenAI-compatible providers (doubao, DeepSeek, LiteLLM) whose
    constrained-decoding engines may reject those constraints.
    """

    strict: bool = True

    def vendor_spec(self, tool: Tool) -> dict[str, Any]:
        params = copy.deepcopy(tool.parameters) if tool.parameters else {}
        if self.strict:
            _force_strict(params)
        else:
            _strip_validation_constraints(params)
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": params,
            },
        }

    def encode_tool_args(self, args: Mapping[str, Any]) -> str:
        return encode_tool_args(args)

def _to_openai_tools(
    tools: list[Tool], *, strict: bool = True,
) -> list[dict[str, Any]]:
    adapter = OpenAIToolSpecAdapter(strict=strict)
    return [adapter.vendor_spec(t) for t in tools]

# --- Streaming bridge -------------------------------------------------------

@dataclass
class _StreamState:
    """Provider event mapping state for one OpenAI-compatible stream."""

    accumulator: StreamAccumulator = field(default_factory=StreamAccumulator)
    tool_scratch: dict[int, dict[str, Any]] = field(default_factory=dict)
    tool_order: list[int] = field(default_factory=list)
    usage: Usage | None = None
    stop_reason: str | None = None
    termination: TerminationHint | None = None

def _map_finish_reason(raw: str | None) -> TerminationHint | None:
    """Translate OpenAI ``finish_reason`` into a kernel ``TerminationHint``.

    Per `.claude/designs/pluggable-architecture.md` §3.1, providers own the
    vocabulary translation so the kernel never inspects vendor strings.
    """

    if raw is None:
        return None
    if raw == "stop":
        return EndTurn()
    if raw == "length":
        return MaxTokens()
    if raw == "tool_calls" or raw == "function_call":
        return ToolUseExpected()
    if raw == "content_filter":
        return ProviderError(detail="content_filter")
    if raw == "pause_turn":
        # OpenAI-compat backends (Doubao) reuse Anthropic's pause_turn
        # signal: the model paused mid-output and expects a resend.
        return PauseTurn()
    return VendorSpecific(raw=raw)

def _extract_usage(usage_obj: Any) -> Usage | None:
    """Pull ``Usage`` out of an OpenAI ``CompletionUsage`` (or partial)."""

    if usage_obj is None:
        return None
    # OpenAI exposes ``prompt_tokens`` / ``completion_tokens``; cached tokens
    # live under ``prompt_tokens_details.cached_tokens`` for providers that
    # support it (LiteLLM passes this through).
    cache_read = 0
    details = getattr(usage_obj, "prompt_tokens_details", None)
    if details is not None:
        cache_read = int(getattr(details, "cached_tokens", 0) or 0)
    return Usage(
        input_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
        cache_read=cache_read,
        cache_write=0,
    )

def _flush_tool_call(state: _StreamState, index: int) -> None:
    scratch = state.tool_scratch.get(index)
    if scratch is None or scratch.get("flushed"):
        return
    state.accumulator.add_tool_call(
        id=scratch.get("id", ""),
        name=scratch.get("name", ""),
        args_delta=scratch.get("arguments", ""),
    )
    scratch["flushed"] = True

# --- Public callable -------------------------------------------------------

@dataclass
class OpenAIStreamFn:
    """Kernel-compatible ``StreamFn`` backed by the OpenAI Chat Completions API.

    Construction parameters:

    - ``api_key``: API key. Defaults to ``OPENAI_API_KEY``.
    - ``base_url``: optional override for the API base URL (point at LiteLLM,
      DeepSeek, Doubao, etc.).
    - ``default_query``: query parameters forwarded on every request — e.g.
      ``{"warpgate-ticket": "<token>"}`` for self-signed Warpgate gateways.
    - ``default_headers``: extra headers forwarded on every request.
    - ``verify_ssl``: forwarded to the underlying httpx client. Set ``False``
      for self-signed cert environments. ``True`` by default.
    - ``retry_policy``: optional policy used for retrying provider calls.
    - ``httpx_client_factory``: optional transport factory used when a custom
      HTTP client is needed (for example ``verify_ssl=False``).
    - ``client``: pre-configured :class:`openai.AsyncOpenAI` instance. Tests
      may inject a stub here to bypass network entirely; if provided, all the
      preceding fields are ignored on the request path.
    """

    api_key: str | None = None
    base_url: str | None = None
    default_query: dict[str, str] | None = None
    default_headers: dict[str, str] | None = None
    verify_ssl: bool = True
    retry_policy: RetryPolicy | None = None
    httpx_client_factory: Callable[..., Any] | None = None
    client: Any = None
    clock: Callable[[], float] = time.time
    thinking_round_trip: Literal["drop", "system_note", "raise"] = "drop"
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None
    events: EventBus | None = None
    azure_endpoint: str | None = None
    api_version: str | None = None
    _reported_thinking_drop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.thinking_round_trip not in {"drop", "system_note", "raise"}:
            raise ValueError(
                "OpenAIStreamFn thinking_round_trip must be one of "
                "'drop', 'system_note', or 'raise'."
            )

    def _emit_thinking_drop_diagnostic(self) -> None:
        if self.events is None or self._reported_thinking_drop:
            return
        self._reported_thinking_drop = True
        self.events.emit_sync(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="info",
                source="openai",
                message=(
                    "OpenAI provider dropped assistant ThinkingBlock content; "
                    "set thinking_round_trip='system_note' or 'raise' to "
                    "change this behavior."
                ),
            ),
        )

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        # Imported lazily so module import doesn't require the SDK to be
        # configured (e.g. in offline test environments using injected client).

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if self.azure_endpoint is not None:
            from openai import AzureOpenAI as _AzureOpenAI

            azure_kwargs: dict[str, Any] = {
                "azure_endpoint": self.azure_endpoint,
                "api_key": api_key,
            }
            if self.api_version is not None:
                azure_kwargs["api_version"] = self.api_version
            if self.default_headers:
                azure_kwargs["default_headers"] = dict(self.default_headers)
            if not self.verify_ssl:
                factory = self.httpx_client_factory or _default_httpx_client
                azure_kwargs["http_client"] = factory(verify=False)
            self.client = _AzureOpenAI(**azure_kwargs)
            return self.client

        from openai import AsyncOpenAI as _AsyncOpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.default_query:
            kwargs["default_query"] = dict(self.default_query)
        if self.default_headers:
            kwargs["default_headers"] = dict(self.default_headers)
        if not self.verify_ssl:
            factory = self.httpx_client_factory or _default_httpx_client
            kwargs["http_client"] = factory(verify=False)
        self.client = _AsyncOpenAI(**kwargs)
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
        # ``thinking`` is intentionally not forwarded: vanilla OpenAI Chat
        # Completions has no thinking-budget knob. Reasoning that comes back
        # as ``delta.reasoning_content`` (LiteLLM Kimi-K2, DeepSeek-R1, …) is
        # surfaced regardless via ``ThinkingDelta`` events.
        del thinking

        client = self._get_client()
        body: dict[str, Any] = {
            "model": model.id,
            "max_tokens": model.max_output_tokens,
            "messages": _to_openai_messages(
                messages,
                system=system,
                thinking_round_trip=self.thinking_round_trip,
                on_thinking_drop=self._emit_thinking_drop_diagnostic,
            ),
            "stream": True,
            # Without this the upstream usage block is null on streaming.
            "stream_options": {"include_usage": True},
        }
        if tools:
            strict = self.base_url is None
            body["tools"] = _to_openai_tools(tools, strict=strict)

        extra = dict(self.extra_body or {})
        if self.reasoning_effort is not None:
            extra.setdefault("reasoning_effort", self.reasoning_effort)
        if extra:
            body["extra_body"] = extra

        state = _StreamState()
        aborted = False

        retry_policy = self.retry_policy or _IdentityRetryPolicy()

        async def _create_stream() -> Any:
            return await client.chat.completions.create(**body)

        stream = await retry_policy.run(
            _create_stream,
            is_retryable=_is_openai_retryable,
        )
        try:
            async for chunk in stream:
                if signal is not None and signal.is_set():
                    aborted = True
                    try:
                        await stream.close()
                    except Exception:
                        # Best-effort close; do not let cleanup mask the abort.
                        logger.opt(exception=True).debug(
                            "openai: error while closing aborted stream"
                        )
                    break
                async for kernel_event in _translate_chunk(chunk, state):
                    yield kernel_event
        finally:
            close = getattr(stream, "close", None)
            if close is not None and not aborted:
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.opt(exception=True).debug(
                        "openai: error while closing stream"
                    )

        # Flush any tool calls that didn't get an explicit end signal —
        # OpenAI's stream model relies on ``finish_reason`` rather than
        # per-block stop events, so we emit ``ToolCallEnd`` here for every
        # tool we started.
        for index in state.tool_order:
            scratch = state.tool_scratch.get(index)
            if scratch is not None and not scratch.get("ended"):
                yield ToolCallEnd(id=scratch.get("id", ""))
                scratch["ended"] = True
                _flush_tool_call(state, index)

        if aborted:
            state.stop_reason = "aborted"
            state.termination = Aborted()

        for index in state.tool_order:
            _flush_tool_call(state, index)
        assembled = state.accumulator.assemble(
            stop_reason=state.stop_reason,
            termination=state.termination,
            usage=state.usage,
            timestamp=self.clock(),
        )
        for parse_error in state.accumulator.parse_errors:
            yield parse_error
        yield MessageEnd(message=assembled)

class _IdentityRetryPolicy:
    async def run(
        self,
        fn: Callable[[], Any],
        *,
        is_retryable: Callable[[BaseException], bool],
    ) -> Any:
        del is_retryable
        return await fn()

async def _translate_chunk(
    chunk: Any,
    state: _StreamState,
) -> AsyncIterator[AssistantStreamEvent]:
    """Translate one raw OpenAI streaming chunk into 0+ kernel stream events.

    Detection is by structural attribute access so tests can drive this with
    lightweight stand-ins instead of real SDK Pydantic models.
    """

    # Some chunks carry only ``usage`` (the final include_usage chunk has an
    # empty ``choices`` list).
    usage_obj = getattr(chunk, "usage", None)
    if usage_obj is not None:
        usage = _extract_usage(usage_obj)
        if usage is not None:
            state.usage = usage

    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return

    choice = choices[0]
    delta = getattr(choice, "delta", None)

    if delta is not None:
        # 1) Reasoning content (OpenAI o-series, DeepSeek-R1, LiteLLM Kimi).
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            state.accumulator.add_thinking(None, reasoning)
            yield ThinkingDelta(text=reasoning, signature=None)

        # 2) Plain text content.
        content = getattr(delta, "content", None)
        if content:
            state.accumulator.add_text(None, content)
            yield TextDelta(text=content)

        # 3) Tool call deltas. Each entry is identified by ``index``; the
        # first entry for a new index carries id + function.name, later
        # entries only carry function.arguments fragments.
        tool_calls = getattr(delta, "tool_calls", None) or []
        for tc in tool_calls:
            index = int(getattr(tc, "index", 0))
            scratch = state.tool_scratch.get(index)
            tc_id = getattr(tc, "id", None)
            fn = getattr(tc, "function", None)
            fn_name = getattr(fn, "name", None) if fn is not None else None
            fn_args = getattr(fn, "arguments", None) if fn is not None else None

            if scratch is None:
                # First time we see this index — open a tool call block.
                scratch = {
                    "id": tc_id or "",
                    "name": fn_name or "",
                    "arguments": "",
                    "started": False,
                    "ended": False,
                    "flushed": False,
                }
                state.tool_scratch[index] = scratch
                state.tool_order.append(index)
            else:
                # Late-arriving id or name (some providers stream them across
                # multiple deltas); fold them in.
                if tc_id and not scratch.get("id"):
                    scratch["id"] = tc_id
                if fn_name and not scratch.get("name"):
                    scratch["name"] = fn_name

            if not scratch["started"] and scratch.get("id") and scratch.get("name"):
                yield ToolCallStart(id=scratch["id"], name=scratch["name"])
                scratch["started"] = True

            if fn_args:
                scratch["arguments"] = scratch.get("arguments", "") + fn_args
                if scratch["started"]:
                    yield ToolCallArgsDelta(
                        id=scratch["id"], args_json_delta=fn_args
                    )

    # 4) Stop reason — emit pending ToolCallEnd events for tools, then record.
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason is not None:
        for index in state.tool_order:
            scratch = state.tool_scratch.get(index)
            if scratch is not None and scratch.get("started") and not scratch.get(
                "ended"
            ):
                yield ToolCallEnd(id=scratch["id"])
                scratch["ended"] = True
                _flush_tool_call(state, index)
        if finish_reason is not None:
            state.stop_reason = finish_reason
            state.termination = _map_finish_reason(finish_reason)

# --- Extension entrypoint --------------------------------------------------

def install(api: Any, config: LlmOpenaiConfig) -> None:
    """Provider extension entrypoint.

    Reads ``config.model`` (required) and the optional fields ``api_key``,
    ``base_url``, ``default_query``, ``default_headers``, ``verify_ssl``,
    ``context_window``, ``max_output_tokens``, ``name`` (registry name —
    defaults to ``"openai"``; override when registering multiple
    OpenAI-compatible providers in the same session).

    """

    model_id = config.model
    if not model_id or not isinstance(model_id, str):
        raise ValueError(
            "agentm.extensions.builtin.llm_openai.install: config.model is required and must "
            "be a non-empty string (e.g. 'gpt-4o' or 'Kimi-K2')."
        )

    retry_policy = api.get_service("retry_policy")
    verify_ssl = config.verify_ssl if config.verify_ssl is not None else True
    if not verify_ssl:
        api.events.emit_sync(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="warning",
                source="openai",
                message=(
                    "OpenAI provider configured with verify_ssl=False; "
                    "TLS certificate verification is disabled for this session."
                ),
            ),
        )

    # Access extra fields from the Pydantic model for pass-through config
    extra = config.model_extra or {}
    stream_fn = OpenAIStreamFn(
        api_key=config.api_key,
        base_url=config.base_url,
        default_query=config.default_query,
        default_headers=config.default_headers,
        verify_ssl=verify_ssl,
        retry_policy=retry_policy,
        thinking_round_trip=config.thinking_round_trip or "drop",
        reasoning_effort=extra.get("reasoning_effort"),
        extra_body=extra.get("extra_body"),
        events=getattr(api, "events", None),
        azure_endpoint=config.azure_endpoint,
        api_version=config.api_version,
    )

    model_kwargs: dict[str, int] = {}
    if config.context_window is not None:
        model_kwargs["context_window"] = config.context_window
    if config.max_output_tokens is not None:
        model_kwargs["max_output_tokens"] = config.max_output_tokens
    model = _build_model(model_id, **model_kwargs)

    raw_name = config.name
    base_url = config.base_url
    if raw_name is None:
        if _is_non_canonical_base_url(base_url):
            raise DuplicateProviderError(
                "agentm.extensions.builtin.llm_openai.install: config['name'] is required when "
                f"base_url={base_url!r} is set to a non-canonical "
                "OpenAI-compatible endpoint. Multiple custom endpoints "
                "default to the bare 'openai' registry name and would "
                "silently overwrite each other. Pass an explicit "
                "config['name'] (e.g. 'doubao', 'litellm', 'deepseek')."
            )
        name = "openai"
    else:
        name = raw_name
    if not isinstance(name, str) or not name:
        raise ValueError(
            "agentm.extensions.builtin.llm_openai.install: config['name'] must be a non-empty string."
        )

    if api.has_provider(name):
        raise DuplicateProviderError(
            f"agentm.extensions.builtin.llm_openai.install: provider name {name!r} is already "
            "registered in this session. Choose a unique config['name'] for "
            "each OpenAI-compatible endpoint."
        )

    api.register_provider(
        name,
        ProviderConfig(stream_fn=stream_fn, model=model, name=name),
    )

# Canonical OpenAI base URLs — anything else is treated as a custom endpoint
# (LiteLLM, DeepSeek, Doubao, vLLM, Ollama, ...). When ``name`` is omitted for
# such an endpoint the provider would otherwise silently register under the
# default key ``"openai"`` and overwrite an earlier custom registration.
_CANONICAL_OPENAI_BASE_URLS: frozenset[str] = frozenset(
    {
        "https://api.openai.com/v1",
        "https://api.openai.com/v1/",
        "https://api.openai.com",
        "https://api.openai.com/",
    }
)

def _is_non_canonical_base_url(base_url: object) -> bool:
    if base_url is None:
        return False
    if not isinstance(base_url, str) or not base_url.strip():
        return False
    return base_url.rstrip("/") not in {url.rstrip("/") for url in _CANONICAL_OPENAI_BASE_URLS}

class DuplicateProviderError(ValueError):
    """Raised when an OpenAI-compatible provider would shadow an existing one.

    Two situations trigger this:

    * ``config['base_url']`` points at a non-canonical (custom) OpenAI-compatible
      endpoint and ``config['name']`` was not supplied — the install would
      otherwise default to the bare ``"openai"`` registry key and silently
      collide with another custom endpoint registered in the same session.
    * The session already has a provider registered under the requested name.
    """

__all__ = ("DuplicateProviderError", "MANIFEST", "OpenAIStreamFn", "install")
