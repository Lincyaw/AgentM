"""LLM-mode contract tests for ``judge_investigation_genuine``.

Mirrors ``test_judges_llm_contract.py``'s pattern: a fake provider whose
``stream_fn`` emits one ``MessageEnd`` carrying an ``AssistantMessage``
with a ``ToolCallBlock`` for ``submit_verdict``.

Three cases:

1. Happy path — provider returns a well-formed payload. ``judge.judge``
   returns exactly that ``Verdict``.
2. Malformed payload once, valid on retry — the impl's one-retry contract
   (design §3.4) returns the valid ``Verdict``; the provider was invoked
   twice.
3. Provider raises twice — the impl returns the canonical ``unclear``
   verdict via :func:`make_unclear` and never falls back to structural
   rules.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from agentm.core.abi import (
    AssistantMessage,
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
)
from agentm.core.abi.extension import ProviderConfig

from agentm_rca_hfsm.atoms import judge_investigation_genuine
from agentm_rca_hfsm.judges import JudgeContext


_CWD = str(Path(__file__).resolve().parents[4])


class _FakeStreamFn:
    """Provider double whose behaviour is driven by a script."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
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
        if not self._script:
            raise AssertionError("_FakeStreamFn: script exhausted")
        entry = self._script.pop(0)
        return self._iter(entry)

    async def _iter(self, entry: Any) -> AsyncIterator[AssistantStreamEvent]:
        if isinstance(entry, Exception):
            raise entry
        if entry == "malformed":
            yield MessageEnd(
                message=AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="no tool call here")],
                    timestamp=0.0,
                    stop_reason="end_turn",
                )
            )
            return
        assert isinstance(entry, dict)
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id="tc-1",
                        name="submit_verdict",
                        arguments=entry,
                    )
                ],
                timestamp=0.0,
                stop_reason="tool_use",
            )
        )


class _LlmAPI:
    """Minimal ExtensionAPI shim covering the surface the judge atom touches."""

    def __init__(self, *, stream_fn: _FakeStreamFn, cwd: str = _CWD) -> None:
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


def _ctx(tag: str) -> JudgeContext:
    return JudgeContext(graph_slice={"tag": tag}, operands={})


def test_llm_happy_path_returns_payload() -> None:
    stream = _FakeStreamFn(
        [
            {
                "verdict": "genuine_investigation",
                "reason": "symptoms recorded, hypotheses verified",
                "confidence": "high",
            }
        ]
    )
    api = _LlmAPI(stream_fn=stream)
    judge_investigation_genuine.install(api, {"mode": "llm"})
    impl = api.get_service("rca.judge.investigation_genuine")

    verdict = impl.judge(_ctx("happy"))

    assert verdict.verdict == "genuine_investigation"
    assert verdict.reason == "symptoms recorded, hypotheses verified"
    assert verdict.confidence == "high"
    assert stream.calls == 1


def test_llm_retries_once_on_malformed() -> None:
    stream = _FakeStreamFn(
        [
            "malformed",
            {
                "verdict": "speculation",
                "reason": "after retry",
                "confidence": "medium",
            },
        ]
    )
    api = _LlmAPI(stream_fn=stream)
    judge_investigation_genuine.install(api, {"mode": "llm"})
    impl = api.get_service("rca.judge.investigation_genuine")

    verdict = impl.judge(_ctx("retry"))

    assert verdict.verdict == "speculation"
    assert verdict.reason == "after retry"
    assert stream.calls == 2


def test_llm_falls_back_to_unclear_on_double_failure() -> None:
    stream = _FakeStreamFn(
        [
            RuntimeError("first provider failure"),
            RuntimeError("second provider failure"),
        ]
    )
    api = _LlmAPI(stream_fn=stream)
    judge_investigation_genuine.install(api, {"mode": "llm"})
    impl = api.get_service("rca.judge.investigation_genuine")

    verdict = impl.judge(_ctx("double-fail"))

    assert verdict.verdict == "unclear"
    assert "judge LLM unreachable" in verdict.reason
    assert "second provider failure" in verdict.reason
    assert verdict.confidence == "none"
    assert stream.calls == 2
