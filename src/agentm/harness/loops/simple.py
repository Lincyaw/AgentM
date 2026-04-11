"""SimpleAgentLoop — Pure-Python ReAct agent loop.

No LangGraph dependency. Implements the cycle:
    LLM -> tool calls -> LLM -> ... -> final answer.

Middleware hooks fire at each stage. Inbox is drained before each LLM call.
stream() is the primary method. run() delegates to it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any, cast

from agentm.harness.cost_budget import BudgetExceeded
from agentm.harness.middleware import MiddlewareBase
from agentm.harness.protocols import AgentLoop, CheckpointStore
from agentm.core.tool import Tool
from agentm.harness.tool_concurrency import (
    get_max_tool_concurrency,
    partition_tool_calls,
)
from agentm.harness.types import (
    AgentEvent,
    AgentResult,
    AgentStatus,
    JsonValue,
    LoopContext,
    Message,
    ModelProtocol,
    RunConfig,
)

logger = logging.getLogger(__name__)

# Exceptions considered transient and safe to retry.
# Covers: rate limits (429), server errors (5xx), network issues.
_RETRYABLE_SUBSTRINGS = (
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "timed out",
    "timeout",
    "connection",
)


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception looks transient."""
    msg = str(exc).lower()
    return any(s in msg for s in _RETRYABLE_SUBSTRINGS)


def _truncate_for_log(value: object, limit: int = 400) -> str:
    """Return a compact log-friendly preview."""
    text = str(value).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _content_to_text(content: object) -> str:
    """Best-effort conversion for LangChain message content payloads."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _message_to_text(message: object) -> str:
    """Extract user-visible text from a message/response object."""
    if isinstance(message, dict):
        return _content_to_text(message.get("content"))
    content_text = _content_to_text(getattr(message, "content", None))
    if content_text.strip():
        return content_text
    text_attr = getattr(message, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return _content_to_text(message)


def _content_block_types(message: object) -> list[str]:
    """Return a compact list of content block types for diagnostics."""
    content_blocks = getattr(message, "content_blocks", None)
    if not isinstance(content_blocks, list):
        return []
    types: list[str] = []
    for block in content_blocks:
        if isinstance(block, dict):
            block_type = block.get("type")
            types.append(str(block_type) if block_type is not None else "<missing>")
        else:
            types.append(type(block).__name__)
    return types


def _message_diagnostic_summary(message: object) -> dict[str, JsonValue]:
    """Build a compact diagnostic summary for a response/message object."""
    if isinstance(message, dict):
        content = message.get("content")
        text_attr: object = None
        response_metadata: object = message.get("response_metadata", {})
        additional_kwargs: object = message.get("additional_kwargs", {})
        tool_calls = message.get("tool_calls", [])
    else:
        content = getattr(message, "content", None)
        text_attr = getattr(message, "text", None)
        response_metadata = getattr(message, "response_metadata", {})
        additional_kwargs = getattr(message, "additional_kwargs", {})
        tool_calls = getattr(message, "tool_calls", None) or []

    response_meta_dict = (
        cast(dict[str, JsonValue], response_metadata)
        if isinstance(response_metadata, dict)
        else {}
    )
    additional_dict = (
        cast(dict[str, JsonValue], additional_kwargs)
        if isinstance(additional_kwargs, dict)
        else {}
    )

    visible_text = _message_to_text(message)
    summary: dict[str, JsonValue] = {
        "message_type": type(message).__name__,
        "content_type": type(content).__name__,
        "visible_text": _truncate_for_log(visible_text, limit=160),
        "visible_text_len": len(visible_text),
        "text_attr": _truncate_for_log(text_attr, limit=160) if text_attr else "",
        "content_preview": _truncate_for_log(content, limit=160),
        "tool_call_count": len(cast(list[dict[str, Any]], tool_calls)),
        "content_block_types": _content_block_types(message),
        "finish_reason": cast(JsonValue, response_meta_dict.get("finish_reason")),
        "model_name": cast(JsonValue, response_meta_dict.get("model_name")),
        "response_id": cast(JsonValue, response_meta_dict.get("id")),
    }
    if "refusal" in additional_dict:
        summary["refusal"] = cast(JsonValue, additional_dict.get("refusal"))
    return summary


def _recent_message_summaries(messages: list[Message], limit: int = 3) -> list[str]:
    """Summarize the trailing conversation turns for debug logging."""
    previews: list[str] = []
    start_idx = max(len(messages) - limit, 0)
    for idx, msg in enumerate(messages[-limit:], start=start_idx):
        if isinstance(msg, dict):
            role = str(msg.get("role", ""))
            tool_calls = cast(list[dict[str, Any]], msg.get("tool_calls", []))
            content = msg.get("content")
        else:
            role = str(getattr(msg, "type", ""))
            tool_calls = cast(list[dict[str, Any]], getattr(msg, "tool_calls", None) or [])
            content = getattr(msg, "content", None)
        text = _content_to_text(content)
        previews.append(
            f"[{idx}] role={role or type(msg).__name__} "
            f"tool_calls={len(tool_calls)} text={_truncate_for_log(text, limit=120)}"
        )
    return previews


def _json_value_is_effectively_empty(value: JsonValue) -> bool:
    """Return True when a JSON-like value is semantically empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, bool | int | float):
        return False
    if isinstance(value, list):
        return len(value) == 0 or all(
            _json_value_is_effectively_empty(v) for v in value
        )
    if isinstance(value, dict):
        return len(value) == 0 or all(
            _json_value_is_effectively_empty(cast(JsonValue, v)) for v in value.values()
        )
    return False


def _json_preview(value: JsonValue) -> str:
    """Serialize a JSON-like value for logs/retry prompts."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


class EmptyStructuredSynthesisError(ValueError):
    """Structured synthesis succeeded syntactically but produced no useful content."""

    def __init__(self, output: JsonValue) -> None:
        self.output = output
        preview = _truncate_for_log(_json_preview(output))
        super().__init__(
            "Structured synthesis produced an empty payload after validation"
            f" | payload_preview={preview}"
        )


class SimpleAgentLoop(AgentLoop):
    """Pure-Python agent loop. No LangGraph dependency.

    Implements the ReAct cycle: LLM -> tool calls -> LLM -> ... -> final answer.
    Middleware hooks fire at each stage. Inbox is drained before each LLM call.

    stream() is the primary method. run() delegates to it.
    """

    def __init__(
        self,
        *,
        model: ModelProtocol,  # Protocol: async ainvoke(messages) -> response
        synthesis_model: ModelProtocol | None = None,
        tools: list[Tool],
        system_prompt: str,
        middleware: list[MiddlewareBase] | None = None,
        output_schema: type | None = None,
        output_prompt: str = "",
        synthesize_retries: int = 2,
        should_terminate: Callable[[object], bool] | None = None,
        checkpoint_store: CheckpointStore | None = None,
        retry_max_attempts: int = 3,
        retry_initial_interval: float = 1.0,
        retry_backoff_factor: float = 2.0,
        budget_excluded_tools: frozenset[str] | None = None,
    ) -> None:
        self._model = model
        self._synthesis_model = synthesis_model or model
        self._tools = {t.name: t for t in tools}
        self._system_prompt = system_prompt
        self._middleware = middleware or []
        self._output_schema = output_schema
        self._output_prompt = output_prompt
        self._synthesize_retries = synthesize_retries
        self._should_terminate = should_terminate or (
            lambda resp: not getattr(resp, "tool_calls", None)
        )
        self._checkpoint_store = checkpoint_store
        self._inbox: deque[str] = deque()
        self._retry_max_attempts = retry_max_attempts
        self._retry_initial_interval = retry_initial_interval
        self._retry_backoff_factor = retry_backoff_factor
        self._budget_excluded_tools = budget_excluded_tools or frozenset()

    def inject(self, message: str) -> None:
        """Inject a message into the agent's inbox.

        The message will be consumed before the next LLM call.
        """
        self._inbox.append(message)

    async def _invoke_with_retry(self, messages: list[Message]) -> object:
        """Call the LLM with exponential backoff retry on transient errors."""
        last_exc: Exception | None = None
        delay = self._retry_initial_interval

        for attempt in range(1, self._retry_max_attempts + 1):
            try:
                return await self._model.ainvoke(messages)
            except Exception as exc:
                last_exc = exc
                if attempt >= self._retry_max_attempts or not _is_retryable(exc):
                    raise
                logger.warning(
                    "LLM call attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt,
                    self._retry_max_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= self._retry_backoff_factor

        raise last_exc  # type: ignore[misc]  # unreachable, satisfies type checker

    async def _synthesize_output(self, messages: list[Message]) -> JsonValue:
        """Produce final output, optionally using output_schema with retries.

        When output_schema is set, tries structured output up to
        (1 + synthesize_retries) times. On validation failure, appends error
        feedback and retries. On final failure, raises instead of silently
        returning an empty or raw-text fallback.
        """
        if not self._output_schema:
            last = messages[-1] if messages else None
            last_text = _message_to_text(last)
            if last_text.strip():
                return cast(JsonValue, last_text)
            error_msg = "Synthesis produced empty non-schema output"
            diag = _message_diagnostic_summary(last)
            logger.error(
                "%s; messages=%d last=%s diag=%s",
                error_msg,
                len(messages),
                _truncate_for_log(last),
                _truncate_for_log(_json_preview(diag), limit=600),
            )
            raise RuntimeError(error_msg)

        # Use function_calling method for broader model compatibility.
        # json_schema method is not supported by many models (e.g. Doubao).
        structured_model = self._synthesis_model.with_structured_output(
            self._output_schema, method="function_calling"
        )

        # Build synthesis messages: output_prompt + conversation history + instruction
        non_system = [
            m
            for m in messages
            if not (isinstance(m, dict) and m.get("role") == "system")
        ]
        synth_messages: list[Message] = []
        if self._output_prompt:
            synth_messages.append({"role": "system", "content": self._output_prompt})
        synth_messages.extend(non_system)
        synth_messages.append(
            {"role": "human", "content": "Produce your final structured report now."}
        )
        last_raw_text = ""
        logger.info(
            "starting synthesis: messages=%d non_system=%d retries=%d schema=%s",
            len(messages),
            len(non_system),
            1 + self._synthesize_retries,
            getattr(self._output_schema, "__name__", str(self._output_schema)),
        )

        for attempt in range(1 + self._synthesize_retries):
            try:
                result = await structured_model.ainvoke(synth_messages)
                output: JsonValue = cast(
                    JsonValue,
                    result.model_dump() if hasattr(result, "model_dump") else result,
                )
                if _json_value_is_effectively_empty(output):
                    raise EmptyStructuredSynthesisError(output)
                logger.info(
                    "synthesis succeeded on attempt %d/%d",
                    attempt + 1,
                    1 + self._synthesize_retries,
                )
                return output
            except Exception as exc:
                raw_text = self._extract_raw_from_error(exc)
                if raw_text.strip():
                    last_raw_text = raw_text
                logger.warning(
                    "synthesize attempt %d/%d failed: %s | raw_preview=%s",
                    attempt + 1,
                    1 + self._synthesize_retries,
                    exc,
                    _truncate_for_log(raw_text),
                )
                if attempt < self._synthesize_retries:
                    if raw_text.strip():
                        synth_messages.append(
                            {"role": "assistant", "content": raw_text}
                        )
                    synth_messages.append(
                        {
                            "role": "human",
                            "content": (
                                f"Your JSON output had validation errors:\n{exc}\n\n"
                                f"Your previous output:\n{raw_text or '<empty>'}\n\n"
                                "Fix the errors and output valid JSON matching the schema exactly. "
                                "An all-empty graph (for example empty nodes, edges, and root_causes) "
                                "is invalid and will be rejected."
                            ),
                        }
                    )
                else:
                    error_msg = f"Synthesis failed after {attempt + 1} attempts: {exc}"
                    logger.error(
                        "%s | last_raw_preview=%s",
                        error_msg,
                        _truncate_for_log(last_raw_text),
                    )
                    raise RuntimeError(error_msg) from exc

        raise RuntimeError("Synthesis exhausted retries without returning output")

    @staticmethod
    def _extract_raw_from_error(exc: Exception) -> str:
        """Extract the raw LLM text from a structured-output failure."""
        if isinstance(exc, EmptyStructuredSynthesisError):
            return _json_preview(exc.output)
        for e in (
            exc,
            getattr(exc, "__cause__", None),
            getattr(exc, "__context__", None),
        ):
            if e is not None and hasattr(e, "llm_output") and e.llm_output:
                return str(e.llm_output)
        return str(exc)

    def _build_tool_chain(
        self, ctx: LoopContext
    ) -> Callable[[str, dict[str, Any]], Awaitable[str]]:
        """Build the middleware-wrapped tool execution chain (once per step)."""

        async def _actual_call(n: str, a: dict[str, Any]) -> str:
            tool = self._tools.get(n)
            if tool is None:
                available = ", ".join(sorted(self._tools))
                return f"Error: tool '{n}' does not exist. Available tools: {available}"
            return await tool.ainvoke(a)

        chain: Callable[[str, dict[str, Any]], Awaitable[str]] = _actual_call
        for mw in reversed(self._middleware):
            prev = chain

            async def _wrap(
                n: str,
                a: dict[str, Any],
                _mw: MiddlewareBase = mw,
                _prev: Callable[[str, dict[str, Any]], Awaitable[str]] = prev,
            ) -> str:
                return await _mw.on_tool_call(n, a, _prev, ctx)

            chain = _wrap

        return chain

    def _log_empty_response_diagnostic(
        self,
        *,
        response: object,
        ctx: LoopContext,
        prepared: list[Message],
    ) -> None:
        """Emit a detailed log when the model returns an empty final response."""
        diag = _message_diagnostic_summary(response)
        logger.error(
            "empty llm response before termination; agent_id=%s step=%d is_worker=%s "
            "tool_calls=%s finish_reason=%s model=%s response_id=%s "
            "visible_text_len=%s content_type=%s block_types=%s recent_messages=%s",
            ctx.agent_id,
            ctx.step,
            self._output_schema is None,
            diag["tool_call_count"],
            diag["finish_reason"],
            diag["model_name"],
            diag["response_id"],
            diag["visible_text_len"],
            diag["content_type"],
            diag["content_block_types"],
            _truncate_for_log(_recent_message_summaries(prepared), limit=500),
        )

    async def run(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> AgentResult:
        """Convenience: iterate stream(), return final result."""
        result: AgentResult | None = None
        async for event in self.stream(input, config=config):
            if event.type == "complete":
                raw = event.data.get("result")
                if isinstance(raw, AgentResult):
                    result = raw
        if result is None:
            raise RuntimeError("Agent loop finished without producing a result")
        return result

    def _drain_inbox(self, messages: list[Message]) -> list[str]:
        """Drain pending injected messages into conversation history."""
        drained: list[str] = []
        while self._inbox:
            injected = self._inbox.popleft()
            messages.append(
                {
                    "role": "human",
                    "content": f"[Injected message]\n{injected}",
                }
            )
            drained.append(injected)
        return drained

    async def _apply_llm_start_middleware(
        self,
        messages: list[Message],
        ctx: LoopContext,
    ) -> list[Message]:
        """Run middleware before an LLM invocation."""
        prepared = messages
        for mw in self._middleware:
            prepared = await mw.on_llm_start(prepared, ctx)
        return prepared

    async def _apply_llm_end_middleware(
        self,
        response: object,
        ctx: LoopContext,
    ) -> object:
        """Run middleware after an LLM invocation."""
        processed = response
        for mw in self._middleware:
            processed = await mw.on_llm_end(processed, ctx)
        return processed

    @staticmethod
    def _accumulate_usage(
        response: object,
        total_input_tokens: int,
        total_output_tokens: int,
    ) -> tuple[int, int]:
        """Update running token totals from model usage metadata."""
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return total_input_tokens, total_output_tokens
        return (
            total_input_tokens + (getattr(usage, "input_tokens", 0) or 0),
            total_output_tokens + (getattr(usage, "output_tokens", 0) or 0),
        )

    async def _build_termination_events(
        self,
        *,
        messages: list[Message],
        agent_id: str,
        step: int,
        tool_call_count: int,
    ) -> list[AgentEvent]:
        """Build completion events for the termination path."""
        try:
            output = await self._synthesize_output(messages)
            result = AgentResult(
                agent_id=agent_id,
                status=AgentStatus.COMPLETED,
                output=output,
                steps=step + 1,
                tool_calls=tool_call_count,
            )
            return [
                AgentEvent(
                    type="complete",
                    agent_id=agent_id,
                    step=step,
                    data={"result": result},
                )
            ]
        except Exception as exc:
            logger.error(
                "final synthesis failed at step %d: %s",
                step,
                exc,
            )
            result = AgentResult(
                agent_id=agent_id,
                status=AgentStatus.FAILED,
                error=str(exc),
                steps=step + 1,
                tool_calls=tool_call_count,
            )
            return [
                AgentEvent(
                    type="error",
                    agent_id=agent_id,
                    step=step,
                    data={"error": str(exc)},
                ),
                AgentEvent(
                    type="complete",
                    agent_id=agent_id,
                    step=step,
                    data={"result": result},
                ),
            ]

    @staticmethod
    def _build_tool_context(ctx: LoopContext, tc: dict[str, Any]) -> LoopContext:
        """Attach tool_call_id to context metadata for middleware."""
        return replace(
            ctx,
            metadata={
                **ctx.metadata,
                "tool_call_id": tc.get("id", ""),
            },
        )

    async def _execute_tool_call(
        self,
        tc: dict[str, Any],
        ctx: LoopContext,
    ) -> tuple[dict[str, Any], str, str]:
        """Execute one tool call through middleware chain."""
        name = str(tc.get("name", ""))
        args = cast(dict[str, Any], tc.get("args", {}))
        tool_ctx = self._build_tool_context(ctx, tc)
        chain = self._build_tool_chain(tool_ctx)
        result_str = await chain(name, args)
        return tc, name, result_str

    async def _execute_concurrent_batch(
        self,
        *,
        batch: list[dict[str, Any]],
        ctx: LoopContext,
        semaphore: asyncio.Semaphore,
    ) -> list[tuple[dict[str, Any], str, str]]:
        """Execute a batch concurrently, converting per-call failures to errors."""

        async def _run_one(tc: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
            async with semaphore:
                return await self._execute_tool_call(tc, ctx)

        gathered = await asyncio.gather(
            *[_run_one(tc) for tc in batch],
            return_exceptions=True,
        )

        normalized: list[tuple[dict[str, Any], str, str]] = []
        for idx, item in enumerate(gathered):
            if isinstance(item, BaseException):
                tc = batch[idx]
                name = str(tc.get("name", "?"))
                result_str = f"Error: tool '{name}' raised {item}"
                normalized.append((tc, name, result_str))
                continue
            normalized.append(item)
        return normalized

    async def _execute_serial_batch(
        self,
        *,
        batch: list[dict[str, Any]],
        ctx: LoopContext,
    ) -> list[tuple[dict[str, Any], str, str]]:
        """Execute a batch serially, propagating failures."""
        results: list[tuple[dict[str, Any], str, str]] = []
        for tc in batch:
            results.append(await self._execute_tool_call(tc, ctx))
        return results

    @staticmethod
    def _append_tool_result(
        messages: list[Message],
        tc: dict[str, Any],
        result_str: str,
    ) -> None:
        """Append a tool result message to conversation history."""
        messages.append(
            {
                "role": "tool",
                "content": result_str,
                "tool_call_id": tc.get("id", ""),
            }
        )

    async def _iter_tool_events(
        self,
        *,
        tool_calls: list[dict[str, Any]],
        ctx: LoopContext,
        messages: list[Message],
        semaphore: asyncio.Semaphore,
        agent_id: str,
        step: int,
        tool_call_count: int,
    ) -> AsyncIterator[tuple[AgentEvent, int]]:
        """Execute all tool calls for one step and yield events incrementally."""
        current_tool_call_count = tool_call_count

        for is_concurrent, batch in partition_tool_calls(tool_calls, self._tools):
            for tc in batch:
                yield (
                    AgentEvent(
                        type="tool_start",
                        agent_id=agent_id,
                        step=step,
                        data={
                            "tool": tc.get("name", ""),
                            "args": tc.get("args", {}),
                        },
                    ),
                    current_tool_call_count,
                )

            if is_concurrent and len(batch) > 1:
                batch_results = await self._execute_concurrent_batch(
                    batch=batch,
                    ctx=ctx,
                    semaphore=semaphore,
                )
            else:
                batch_results = await self._execute_serial_batch(batch=batch, ctx=ctx)

            for tc, name, result_str in batch_results:
                if name not in self._budget_excluded_tools:
                    current_tool_call_count += 1
                yield (
                    AgentEvent(
                        type="tool_end",
                        agent_id=agent_id,
                        step=step,
                        data={"tool": name, "result": result_str},
                    ),
                    current_tool_call_count,
                )
                self._append_tool_result(messages, tc, result_str)

    async def stream(
        self, input: str | list[Message], *, config: RunConfig | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent loop, yielding events as they occur.

        The last event is always type="complete" with data={"result": AgentResult}.
        """
        config = config or RunConfig()
        agent_id = str(config.metadata.get("agent_id", ""))

        # Build initial message list: system prompt + user input
        # The loop always owns the system prompt; strip any system messages
        # from caller-provided lists to avoid duplicates (DynamicContextMiddleware
        # handles this for orchestrators, but workers may not have it).
        if isinstance(input, str):
            user_messages: list[Message] = [{"role": "human", "content": input}]
        else:
            user_messages = [
                m
                for m in input
                if not (isinstance(m, dict) and m.get("role") == "system")
            ]
        messages: list[Message] = [
            {"role": "system", "content": self._system_prompt},
            *user_messages,
        ]
        step = 0
        tool_call_count = 0
        total_input_tokens = 0
        total_output_tokens = 0

        semaphore = asyncio.Semaphore(get_max_tool_concurrency())
        timeout_cm = (
            asyncio.timeout(config.timeout)
            if config.timeout
            else contextlib.nullcontext()
        )
        try:
            async with timeout_cm:
                while config.max_steps is None or step < config.max_steps:
                    # 1. Drain inbox
                    for injected in self._drain_inbox(messages):
                        yield AgentEvent(
                            type="inject",
                            agent_id=agent_id,
                            step=step,
                            data={"message": injected},
                        )

                    # 2. Middleware: on_llm_start
                    ctx = LoopContext(
                        agent_id=agent_id,
                        step=step,
                        max_steps=config.max_steps,
                        tool_call_count=tool_call_count,
                        metadata=config.metadata,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                    )
                    prepared = await self._apply_llm_start_middleware(messages, ctx)

                    yield AgentEvent(type="llm_start", agent_id=agent_id, step=step)

                    # 3. Call LLM (with retry on transient errors)
                    response = await self._invoke_with_retry(prepared)

                    # 3b. Accumulate token usage
                    total_input_tokens, total_output_tokens = self._accumulate_usage(
                        response,
                        total_input_tokens,
                        total_output_tokens,
                    )

                    # 4. Middleware: on_llm_end
                    response = await self._apply_llm_end_middleware(response, ctx)

                    response_text = _message_to_text(response)
                    tool_calls = getattr(response, "tool_calls", None) or []
                    if not tool_calls and not response_text.strip():
                        self._log_empty_response_diagnostic(
                            response=response,
                            ctx=ctx,
                            prepared=prepared,
                        )

                    messages.append(response)  # type: ignore[arg-type]
                    yield AgentEvent(
                        type="llm_end",
                        agent_id=agent_id,
                        step=step,
                        data={"content": getattr(response, "content", "")},
                    )

                    # 5. Check termination via should_terminate callback
                    terminate = self._should_terminate(response)

                    if terminate:
                        for event in await self._build_termination_events(
                            messages=messages,
                            agent_id=agent_id,
                            step=step,
                            tool_call_count=tool_call_count,
                        ):
                            yield event
                        return

                    if not tool_calls:
                        # should_terminate=False but no tool_calls: continue loop
                        step += 1
                        continue

                    async for event, tool_call_count in self._iter_tool_events(
                        tool_calls=tool_calls,
                        ctx=ctx,
                        messages=messages,
                        semaphore=semaphore,
                        agent_id=agent_id,
                        step=step,
                        tool_call_count=tool_call_count,
                    ):
                        yield event

                    step += 1

                    # 7. Checkpoint (optional)
                    if self._checkpoint_store:
                        await self._checkpoint_store.save(
                            agent_id, {"messages": messages, "step": step}
                        )

                # Max steps exhausted
                result = AgentResult(
                    agent_id=agent_id,
                    status=AgentStatus.FAILED,
                    error=f"Max steps ({config.max_steps}) reached",
                    steps=step,
                    tool_calls=tool_call_count,
                )
                yield AgentEvent(
                    type="complete",
                    agent_id=agent_id,
                    step=step,
                    data={"result": result},
                )
        except TimeoutError:
            result = AgentResult(
                agent_id=agent_id,
                status=AgentStatus.FAILED,
                error=f"Timeout after {config.timeout}s",
                steps=step,
                tool_calls=tool_call_count,
            )
            yield AgentEvent(
                type="complete",
                agent_id=agent_id,
                step=step,
                data={"result": result},
            )
        except BudgetExceeded as exc:
            yield AgentEvent(
                type="error",
                agent_id=agent_id,
                step=step,
                data={"error": str(exc)},
            )
            # Attempt output synthesis from conversation so far
            try:
                output = await self._synthesize_output(messages)
            except Exception as synth_exc:
                logger.error(
                    "budget exceeded and synthesis failed at step %d: %s",
                    step,
                    synth_exc,
                )
                output = None
            result = AgentResult(
                agent_id=agent_id,
                status=AgentStatus.FAILED,
                output=output,
                error=str(exc),
                steps=step,
                tool_calls=tool_call_count,
            )
            yield AgentEvent(
                type="complete",
                agent_id=agent_id,
                step=step,
                data={"result": result},
            )
