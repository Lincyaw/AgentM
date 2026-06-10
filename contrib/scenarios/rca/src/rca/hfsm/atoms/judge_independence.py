"""``judge_independence`` — ``rca.judge.independence`` service atom.

Phase 2 C1 of the rca_hfsm scenario. Registers a single ``Judge``
implementation under the service name ``rca.judge.independence`` and
toggles between an LLM-backed and a scripted (stub) backing
implementation via ``config.mode`` (default: ``"llm"``).

Replaces the Phase 1 ``worker_session_id`` literal-equality check (see
``updates.independent_positive_workers``). Catches: same source data,
identical observations across different session IDs, brief copy-paste.
C1 only mounts the judge as a service; the gate continues to use its
Phase 1 rules. The gate refactor is C2's job.

JudgeContext shape: ``graph_slice = {"check_a", "check_b"}`` with
``operands = {}``. Canonical verdict strings per design §4.3:
``"independent" | "redundant" | "unclear"``.

§11 single-file contract: stdlib + ``agentm.core.abi.*`` +
``agentm.extensions`` + scenario-local ``judges`` module only. Failure
mode: one retry on provider error or malformed ``submit_verdict``
payload, then :func:`make_unclear`. No regex anywhere.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    AssistantStreamEvent,
    FunctionTool,
    MessageEnd,
    Model,
    TextContent,
    Tool,
    ToolCallBlock,
    ToolResult,
    UserMessage,
)
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from rca.hfsm.judges import (
    JudgeContext,
    SUBMIT_VERDICT_TOOL_NAME,
    Verdict,
    build_submit_verdict_tool_schema,
    canonical_cache_key,
    make_unclear,
)


_KIND = "independence"
_SERVICE_NAME = f"rca.judge.{_KIND}"
_PROMPT_RELPATH = f"contrib/scenarios/rca/prompts/hfsm/judges/{_KIND}.md"
_LRU_MAX = 256


class _ScriptedVerdict(BaseModel):
    verdict: str
    reason: str
    confidence: str


class JudgeIndependenceConfig(BaseModel):
    mode: str = "llm"
    model: str | None = None
    scripted: list[_ScriptedVerdict] = []


MANIFEST = ExtensionManifest(
    name="judge_independence",
    description=(
        "Registers the rca.judge.independence service. LLM-backed by default; "
        "scripted stub mode available via config.mode='stub' for tests."
    ),
    registers=(),
    config_schema=JudgeIndependenceConfig,
    requires=(),
)


async def _inert_execute(args: dict[str, Any]) -> ToolResult:
    del args
    return ToolResult(content=[TextContent(type="text", text="ok")])


def _build_tool() -> Tool:
    schema = build_submit_verdict_tool_schema(_KIND)
    return FunctionTool(
        name=schema.name,
        description=schema.description,
        parameters=schema.parameters,
        fn=_inert_execute,
    )


def _format_user_message(ctx: JudgeContext) -> str:
    return (
        "graph_slice:\n"
        + json.dumps(ctx.graph_slice, indent=2, sort_keys=True, default=str)
        + "\n\noperands:\n"
        + json.dumps(ctx.operands, indent=2, sort_keys=True, default=str)
    )


def _load_prompt(cwd: str) -> str:
    for candidate in (
        Path(cwd) / _PROMPT_RELPATH,
        Path(__file__).resolve().parents[4] / "prompts" / "hfsm" / "judges" / f"{_KIND}.md",
    ):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"judge_{_KIND}: prompt file not found")


def _parse_submit_verdict(message: AssistantMessage) -> Verdict:
    for block in message.content:
        if not isinstance(block, ToolCallBlock) or block.name != SUBMIT_VERDICT_TOOL_NAME:
            continue
        args = block.arguments
        if not isinstance(args, dict):
            raise ValueError(f"submit_verdict arguments not a dict: {args!r}")
        verdict = args.get("verdict")
        reason = args.get("reason")
        confidence = args.get("confidence")
        if not isinstance(verdict, str) or not verdict.strip():
            raise ValueError(f"submit_verdict.verdict missing/empty: {args!r}")
        if not isinstance(reason, str):
            raise ValueError(f"submit_verdict.reason not a string: {args!r}")
        if not isinstance(confidence, str):
            raise ValueError(f"submit_verdict.confidence not a string: {args!r}")
        return Verdict(verdict=verdict, reason=reason, confidence=confidence)
    raise ValueError("assistant message contained no submit_verdict tool call")


def _run_coro(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


class _StubJudge:
    def __init__(self, kind: str, scripted: list[dict[str, Any]]) -> None:
        self.kind = kind
        self._scripted = list(scripted)
        self._cursor = 0
        self._cache: "OrderedDict[str, Verdict]" = OrderedDict()

    def judge(self, context: JudgeContext) -> Verdict:
        key = canonical_cache_key(context)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        if self._cursor >= len(self._scripted):
            raise IndexError(
                f"_StubJudge({self.kind}): scripted verdicts exhausted "
                f"after {self._cursor} calls"
            )
        entry = self._scripted[self._cursor]
        self._cursor += 1
        verdict = Verdict(
            verdict=str(entry["verdict"]),
            reason=str(entry["reason"]),
            confidence=str(entry["confidence"]),
        )
        self._cache[key] = verdict
        if len(self._cache) > _LRU_MAX:
            self._cache.popitem(last=False)
        return verdict


class _LlmJudge:
    def __init__(
        self, *, kind: str, api: ExtensionAPI, model_override: str | None
    ) -> None:
        self.kind = kind
        self._api = api
        self._model_override = model_override
        self._tool = _build_tool()
        self._cache: "OrderedDict[str, Verdict]" = OrderedDict()

    def judge(self, context: JudgeContext) -> Verdict:
        key = canonical_cache_key(context)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        verdict = _run_coro(self._judge_async(context))
        self._cache[key] = verdict
        if len(self._cache) > _LRU_MAX:
            self._cache.popitem(last=False)
        return verdict

    async def _judge_async(self, context: JudgeContext) -> Verdict:
        last_err: str = ""
        for attempt in range(2):
            try:
                return await self._call_provider(context)
            except Exception as exc:  # noqa: BLE001 — design §3.4 catch-all
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt == 1:
                    break
        return make_unclear(reason=f"judge LLM unreachable: {last_err}")

    async def _call_provider(self, context: JudgeContext) -> Verdict:
        provider = self._api.provider
        if provider is None:
            raise RuntimeError("no active provider registered")
        model = self._select_model(provider.model)
        messages: list[AgentMessage] = [
            UserMessage(
                role="user",
                content=[TextContent(type="text", text=_format_user_message(context))],
                timestamp=0.0,
            )
        ]
        stream: AsyncIterator[AssistantStreamEvent] = provider.stream_fn(
            messages=messages,
            model=model,
            tools=[self._tool],
            system=_load_prompt(self._api.cwd),
            signal=None,
            thinking="off",
        )
        final_message: AssistantMessage | None = None
        async for event in stream:
            if isinstance(event, MessageEnd):
                final_message = event.message
        if final_message is None:
            raise RuntimeError("provider stream ended without MessageEnd")
        return _parse_submit_verdict(final_message)

    def _select_model(self, default: Model) -> Model:
        if self._model_override is None:
            return default
        return Model(
            id=self._model_override,
            provider=default.provider,
            context_window=default.context_window,
            max_output_tokens=default.max_output_tokens,
            metadata=dict(getattr(default, "metadata", {})),
        )


def install(api: ExtensionAPI, config: JudgeIndependenceConfig) -> None:
    mode = config.mode.lower()
    if mode not in {"llm", "stub"}:
        raise ValueError(
            f"judge_{_KIND}: config.mode must be 'llm' or 'stub'; got {mode!r}"
        )
    impl: Any
    if mode == "stub":
        impl = _StubJudge(
            kind=_KIND,
            scripted=[v.model_dump() for v in config.scripted],
        )
    else:
        impl = _LlmJudge(kind=_KIND, api=api, model_override=config.model)
    api.set_service(_SERVICE_NAME, impl)
