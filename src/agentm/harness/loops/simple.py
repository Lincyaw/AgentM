"""SimpleAgentLoop — Pure-Python ReAct agent loop.

No LangGraph dependency. Implements the cycle:
    LLM -> tool calls -> LLM -> ... -> final answer.

Middleware hooks fire at each stage. Inbox is drained before each LLM call.
stream() is the primary method. run() delegates to it.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from agentm.harness.tool import Tool
from agentm.harness.types import (
    AgentEvent,
    AgentResult,
    AgentStatus,
    LoopContext,
    RunConfig,
)

logger = logging.getLogger(__name__)


class SimpleAgentLoop:
    """Pure-Python agent loop. No LangGraph dependency.

    Implements the ReAct cycle: LLM -> tool calls -> LLM -> ... -> final answer.
    Middleware hooks fire at each stage. Inbox is drained before each LLM call.

    stream() is the primary method. run() delegates to it.
    """

    def __init__(
        self,
        *,
        model: Any,  # Protocol: async ainvoke(messages) -> response
        tools: list[Tool],
        system_prompt: str,
        middleware: list[Any] | None = None,
        output_schema: type | None = None,
        output_prompt: str = "",
        synthesize_retries: int = 2,
        should_terminate: Callable[[Any], bool] | None = None,
        checkpoint_store: Any | None = None,
    ) -> None:
        self._model = model
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
        self._inbox: list[str] = []

    def inject(self, message: str) -> None:
        """Inject a message into the agent's inbox.

        The message will be consumed before the next LLM call.
        """
        self._inbox.append(message)

    async def _synthesize_output(self, messages: list[Any]) -> Any:
        """Produce final output, optionally using output_schema with retry/fallback.

        When output_schema is set, tries structured output up to
        (1 + synthesize_retries) times. On validation failure, appends error
        feedback and retries. On final failure, falls back to plain LLM and
        returns {"raw_text": ...}.
        """
        if not self._output_schema:
            # No schema — return the last AI message content
            last = messages[-1] if messages else None
            return getattr(last, "content", str(last))

        structured_model = self._model.with_structured_output(self._output_schema)

        # Build synthesis messages: output_prompt + conversation history + instruction
        non_system = [
            m for m in messages if not (isinstance(m, dict) and m.get("role") == "system")
        ]
        synth_messages: list[Any] = []
        if self._output_prompt:
            synth_messages.append({"role": "system", "content": self._output_prompt})
        synth_messages.extend(non_system)
        synth_messages.append(
            {"role": "human", "content": "Produce your final structured report now."}
        )

        for attempt in range(1 + self._synthesize_retries):
            try:
                result = await structured_model.ainvoke(synth_messages)
                output = result.model_dump() if hasattr(result, "model_dump") else result
                logger.info(
                    "synthesize attempt %d/%d succeeded",
                    attempt + 1,
                    1 + self._synthesize_retries,
                )
                return output
            except Exception as exc:
                raw_text = self._extract_raw_from_error(exc)
                if attempt < self._synthesize_retries:
                    logger.warning(
                        "synthesize attempt %d/%d failed (%s), retrying",
                        attempt + 1,
                        1 + self._synthesize_retries,
                        exc,
                    )
                    synth_messages.append({"role": "assistant", "content": raw_text})
                    synth_messages.append(
                        {
                            "role": "human",
                            "content": (
                                f"Your JSON output had validation errors:\n{exc}\n\n"
                                f"Your previous output:\n{raw_text}\n\n"
                                "Fix the errors and output valid JSON matching the schema exactly."
                            ),
                        }
                    )
                else:
                    logger.error(
                        "synthesize FAILED all %d attempts, falling back to plain LLM",
                        attempt + 1,
                    )
                    try:
                        raw = await self._model.ainvoke(synth_messages)
                        return {"raw_text": str(getattr(raw, "content", raw))}
                    except Exception as exc2:
                        logger.error("synthesize fallback also FAILED: %s", exc2)
                        return {"raw_text": ""}

        return {"raw_text": ""}  # unreachable, but satisfies type checker

    @staticmethod
    def _extract_raw_from_error(exc: Exception) -> str:
        """Extract the raw LLM text from a structured-output failure."""
        for e in (exc, getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
            if e is not None and hasattr(e, "llm_output") and e.llm_output:
                return str(e.llm_output)
        return str(exc)

    async def run(
        self, input: str, *, config: RunConfig | None = None
    ) -> AgentResult:
        """Convenience: iterate stream(), return final result."""
        result = None
        async for event in self.stream(input, config=config):
            if event.type == "complete":
                result = event.data.get("result")
        return result  # type: ignore[return-value]

    async def stream(
        self, input: str, *, config: RunConfig | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent loop, yielding events as they occur.

        The last event is always type="complete" with data={"result": AgentResult}.
        """
        config = config or RunConfig()
        agent_id = config.metadata.get("agent_id", "")

        # Build initial message list: system prompt + user input
        messages: list[Any] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "human", "content": input},
        ]
        step = 0
        tool_call_count = 0

        while config.max_steps is None or step < config.max_steps:
            # 1. Drain inbox
            while self._inbox:
                injected = self._inbox.pop(0)
                messages.append(
                    {"role": "human", "content": f"[Injected message]\n{injected}"}
                )
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
            )
            prepared = messages
            for mw in self._middleware:
                prepared = await mw.on_llm_start(prepared, ctx)

            yield AgentEvent(type="llm_start", agent_id=agent_id, step=step)

            # 3. Call LLM
            response = await self._model.ainvoke(prepared)

            # 4. Middleware: on_llm_end
            for mw in self._middleware:
                response = await mw.on_llm_end(response, ctx)

            messages.append(response)
            yield AgentEvent(
                type="llm_end",
                agent_id=agent_id,
                step=step,
                data={"content": getattr(response, "content", "")},
            )

            # 5. Check termination via should_terminate callback
            tool_calls = getattr(response, "tool_calls", None) or []
            terminate = self._should_terminate(response)

            if terminate:
                # Terminate: skip tool execution even if tool_calls present
                output = await self._synthesize_output(messages)

                result = AgentResult(
                    agent_id=agent_id,
                    status=AgentStatus.COMPLETED,
                    output=output,
                    steps=step + 1,
                    tool_calls=tool_call_count,
                )
                yield AgentEvent(
                    type="complete",
                    agent_id=agent_id,
                    step=step,
                    data={"result": result},
                )
                return

            if not tool_calls:
                # should_terminate=False but no tool_calls: continue loop
                step += 1
                continue

            # 6. Execute tools (through middleware wrapping chain)
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})

                # Build the call chain: mw_N( ... mw_1( actual_tool ) ... )
                async def _actual_call(n: str, a: dict[str, Any]) -> str:
                    return await self._tools[n].ainvoke(a)

                chain = _actual_call
                for mw in reversed(self._middleware):
                    prev = chain

                    async def _wrap(
                        n: str,
                        a: dict[str, Any],
                        _mw: Any = mw,
                        _prev: Any = prev,
                    ) -> str:
                        return await _mw.on_tool_call(n, a, _prev, ctx)

                    chain = _wrap

                yield AgentEvent(
                    type="tool_start",
                    agent_id=agent_id,
                    step=step,
                    data={"tool": name, "args": args},
                )
                result_str = await chain(name, args)
                tool_call_count += 1
                yield AgentEvent(
                    type="tool_end",
                    agent_id=agent_id,
                    step=step,
                    data={"tool": name, "result": result_str},
                )

                messages.append(
                    {
                        "role": "tool",
                        "content": result_str,
                        "tool_call_id": tc.get("id", ""),
                    }
                )

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
