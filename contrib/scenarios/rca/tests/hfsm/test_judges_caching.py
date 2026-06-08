"""Per-session LRU caching tests for the LLM-mode judges.

Same ``canonical_cache_key(ctx)`` → same Verdict, no second provider call.
Different ``ctx`` → fresh provider call. The fake stream_fn counts its
invocations so the test asserts the cache hit / miss pattern by counting.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantMessage,
    AssistantStreamEvent,
    MessageEnd,
    Model,
    ToolCallBlock,
)
from agentm.core.abi.extension import ProviderConfig

from agentm_rca.hfsm.atoms import (
    judge_coverage,
    judge_falsified_genuinely,
    judge_independence,
    judge_satisfied,
)
from agentm_rca.hfsm.judges import JudgeContext


_CWD = str(Path(__file__).resolve().parents[5])


class _CountingStreamFn:
    """Provider double that emits the same canned ``submit_verdict`` payload
    every time and counts invocations.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del messages, model, tools, system, signal, thinking
        self.calls += 1
        return self._iter(self.calls)

    async def _iter(self, n: int) -> AsyncIterator[AssistantStreamEvent]:
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id=f"tc-{n}",
                        name="submit_verdict",
                        arguments={
                            "verdict": f"v-{n}",
                            "reason": f"r-{n}",
                            "confidence": "low",
                        },
                    )
                ],
                timestamp=0.0,
                stop_reason="tool_use",
            )
        )


class _LlmAPI:
    def __init__(self, *, stream_fn: _CountingStreamFn, cwd: str = _CWD) -> None:
        self._services: dict[str, Any] = {}
        self.cwd = cwd
        self._provider = ProviderConfig(
            stream_fn=stream_fn,  # type: ignore[arg-type]
            model=Model(
                id="fake-judge",
                provider="fake",
                context_window=8000,
                max_output_tokens=512,
            ),
            name="fake-judge",
        )

    @property
    def provider(self) -> ProviderConfig:
        return self._provider

    def set_service(self, name: str, obj: Any) -> None:
        self._services[name] = obj

    def get_service(self, name: str) -> Any:
        return self._services.get(name)


_JUDGES = [
    (judge_satisfied, "rca.judge.satisfied"),
    (judge_coverage, "rca.judge.coverage"),
    (judge_independence, "rca.judge.independence"),
    (judge_falsified_genuinely, "rca.judge.falsified_genuinely"),
]


@pytest.mark.parametrize(("module", "service_name"), _JUDGES)
def test_same_context_hits_cache(module: Any, service_name: str) -> None:
    stream = _CountingStreamFn()
    api = _LlmAPI(stream_fn=stream)
    module.install(api, {"mode": "llm"})
    impl = api.get_service(service_name)

    ctx = JudgeContext(graph_slice={"x": 1}, operands={})
    first = impl.judge(ctx)
    second = impl.judge(ctx)

    assert first == second
    # Cache hit on the second call — the provider was only invoked once.
    assert stream.calls == 1


