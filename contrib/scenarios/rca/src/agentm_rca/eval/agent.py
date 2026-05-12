"""``rcabench-platform`` BaseAgent adapter for the AgentM RCA scenario.

Discovered by ``rcabench-platform``'s ``llm_eval.agents`` entry point and
invoked via ``rca llm-eval run --agent agentm``. Bridges the
``incident + data_dir -> CausalGraph JSON`` contract to an in-process
``AgentSession`` running the local ``rca`` scenario.

Two pieces of glue do the work:

* The orchestrator's ``submit_final_report`` tool in
  :mod:`agentm_rca.tools.finalize` validates against the rcabench-platform
  ``AgentRCAOutput`` contract and emits the model's
  ``model_dump_json(by_alias=True)`` as its tool result. The adapter
  subscribes to ``tool_result`` on the session bus and parses that
  authoritative payload via :meth:`AgentRCAOutput.parse_str` — never the
  unvalidated ``tool_call`` args. A failed validation re-runs the model
  without polluting captured state.
* The session's final message list is walked once after ``prompt`` returns
  and translated into a ``rcabench-platform`` :class:`Trajectory`. The
  system prompt is captured separately from the first
  ``before_send_to_llm`` event because :meth:`AgentSession.prompt` does not
  return it.
"""

from __future__ import annotations

import json
import os
from typing import Any

from rcabench_platform.v3.sdk.llm_eval.agents.base_agent import (
    AgentResult,
    BaseAgent,
    RunContext,
)
from rcabench_platform.v3.sdk.llm_eval.trajectory.schema import (
    AgentTrajectory,
    Message,
    ToolCall,
    Trajectory,
    Turn,
)

_DEFAULT_MODEL = "claude-sonnet-4-6"
# RCA investigations dispatch workers, poll, and run many SQL queries; the
# kernel's stock 32-turn cap exhausts before ``submit_final_report`` is
# reached. Bump the default and let the framework's ``--max-steps`` (or
# ``--ak max_turns=N``) override.
_DEFAULT_MAX_TURNS = 128
# Empty AgentRCAOutput-shaped fallback when the orchestrator never reaches
# ``submit_final_report``. Keeps the wire shape consistent with successful
# runs so the platform's parsers don't choke.
_EMPTY_AGENT_RCA_OUTPUT: dict[str, list[Any]] = {
    "root_causes": [],
    "propagation": [],
}


def _provider_name_from_base_url(base_url: str) -> str:
    """Derive a stable provider registry slug from a base URL.

    PR #95 requires every non-canonical OpenAI-compatible endpoint to
    register under a unique name. The host segment is unique per gateway
    in practice (LiteLLM, Doubao Ark, DeepSeek, etc.) and stays stable
    across runs, which keeps observability traces comparable.
    """

    from urllib.parse import urlparse

    host = urlparse(base_url).hostname or "openai-compat"
    return host.replace(".", "-")


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"false", "0", "no", "off", ""}


def _build_provider(
    provider: str, model: str
) -> tuple[str, dict[str, Any]]:
    """Same env-var convention as ``agentm.cli._build_provider``.

    Duplicated here (rather than imported) because the eval adapter is
    discovered via ``llm_eval.agents`` entry point and must not assume
    the CLI module has been imported.
    """
    if provider == "anthropic":
        cfg: dict[str, Any] = {"model": model}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            cfg["base_url"] = base_url
        return ("agentm.extensions.builtin.llm_anthropic", cfg)

    if provider == "openai":
        cfg = {"model": model}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            cfg["base_url"] = base_url
            # PR #95 (8b8231e) made ``name`` mandatory whenever
            # ``base_url`` is non-canonical, to prevent two custom
            # endpoints from clobbering each other under the bare
            # ``openai`` registry slot. Honor an explicit override via
            # ``AGENTM_PROVIDER_NAME`` and otherwise derive a stable
            # slug from the base host so eval rollouts don't crash.
            cfg["name"] = os.environ.get(
                "AGENTM_PROVIDER_NAME"
            ) or _provider_name_from_base_url(base_url)
        ticket = os.environ.get("WARPGATE_TICKET")
        if ticket:
            cfg["default_query"] = {"warpgate-ticket": ticket}
        if not _env_bool("OPENAI_VERIFY_SSL", default=True):
            cfg["verify_ssl"] = False
        return ("agentm.extensions.builtin.llm_openai", cfg)

    raise ValueError(
        f"unknown provider {provider!r}; expected 'anthropic' or 'openai'"
    )


def _coerce_max_turns(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        result = int(value)
    except (TypeError, ValueError):
        return fallback
    return result if result > 0 else fallback


class AgentMAgent(BaseAgent):
    """Run an RCA investigation through an in-process AgentM session."""

    def __init__(
        self,
        *,
        scenario: str = "rca",
        model: str | None = None,
        provider: str | None = None,
        exp_id: str | None = None,
        max_turns: Any = None,
        **_extra: Any,
    ) -> None:
        # ``rcabench-platform`` passes ``exp_id`` plus any ``--ak key=value``
        # kwargs to the agent ctor. We accept and ignore the unknowns so a
        # stale flag in someone's eval YAML does not crash the rollout.
        # ``--ak`` values arrive as strings, so coerce explicitly.
        self._scenario = scenario
        self._model = model or os.environ.get("AGENTM_MODEL", _DEFAULT_MODEL)
        self._provider = (
            provider
            or os.environ.get("AGENTM_PROVIDER")
            or "anthropic"
        )
        self._exp_id = exp_id
        self._max_turns = _coerce_max_turns(max_turns, _DEFAULT_MAX_TURNS)

    @staticmethod
    def name() -> str:
        return "agentm"

    def model_name(self) -> str | None:
        return self._model

    async def run(
        self,
        incident: str,
        data_dir: str,
        **kwargs: Any,
    ) -> AgentResult:
        from agentm.core.abi import (
            BeforeSendToLlmEvent,
            EventBus,
            ToolResultEvent,
        )
        from agentm.core.abi.messages import TextContent as _TextContent
        from agentm.core.abi.loop import LoopConfig
        from agentm.core.abi.session_config import AgentSessionConfig
        from agentm.core.runtime.session import AgentSession
        from agentm.core.runtime.session_factory import create_agent_session
        from rcabench_platform.v3.sdk.evaluation.v2 import AgentRCAOutput

        ctx: RunContext | None = kwargs.get("ctx")
        max_turns = _coerce_max_turns(kwargs.get("max_steps"), self._max_turns)
        os.environ["AGENTM_RCA_DATA_DIR"] = data_dir

        captured: dict[str, Any] = {
            "submission": None,
            "system_prompt": "",
        }

        def _on_tool_result(event: ToolResultEvent) -> None:
            if event.tool_name != "submit_final_report":
                return
            if event.result.is_error:
                return
            text = "".join(
                block.text
                for block in event.result.content
                if isinstance(block, _TextContent)
            )
            try:
                output = AgentRCAOutput.parse_str(text)
            except Exception:
                # Tool returned a non-conforming payload; treat as missing.
                return
            captured["submission"] = output
            if ctx is not None:
                ctx.emit(
                    {
                        "type": "progress",
                        "message": "submit_final_report accepted",
                    }
                )

        def _on_before_llm(event: BeforeSendToLlmEvent) -> None:
            if not captured["system_prompt"] and event.system:
                captured["system_prompt"] = event.system

        bus = EventBus()
        bus.on("tool_result", _on_tool_result)
        bus.on("before_send_to_llm", _on_before_llm)

        # Mirror ``agentm.cli._build_provider`` so the eval adapter honors
        # the same ``AGENTM_PROVIDER`` / ``OPENAI_*`` / ``ANTHROPIC_*``
        # convention as ``uv run agentm``. Previously we always pinned
        # the anthropic provider, which silently routed Doubao-Seed
        # requests through whatever ``ANTHROPIC_BASE_URL`` happened to
        # point at (e.g. the Kimi anthropic-compat gateway) instead of
        # the intended ``OPENAI_BASE_URL`` LiteLLM endpoint.
        provider_module, provider_config = _build_provider(
            self._provider, self._model
        )

        config = AgentSessionConfig(
            cwd=os.getcwd(),
            provider=(provider_module, provider_config),
            scenario=self._scenario,
            bus=bus,
            loop_config=LoopConfig(max_turns=max_turns),
        )

        session = await create_agent_session(AgentSession, config)
        try:
            if ctx is not None:
                ctx.emit({"type": "running", "run_id": session.session_id})
            final_messages = await session.prompt(incident)
        finally:
            await session.shutdown()

        submission: AgentRCAOutput | None = captured["submission"]
        if submission is None:
            response = json.dumps(_EMPTY_AGENT_RCA_OUTPUT)
            submission_dump: Any = None
        else:
            response = submission.model_dump_json(by_alias=True)
            submission_dump = submission.model_dump(
                mode="json", by_alias=True
            )

        trajectory = _build_trajectory(
            agent_name=f"agentm:{self._scenario}",
            system_prompt=captured["system_prompt"],
            final_messages=final_messages,
        )

        # OTel-correct trace_id (= ``root_session_id``, shared by all
        # sessions in the rollout tree) goes onto
        # ``AgentResult.trace_id``; rcabench-platform >= 0.4.44 forwards
        # it via ``RolloutResult`` into ``evaluation_data.trace_id``.
        # A single ``trace_id =`` filter then recovers the parent plus
        # every spawned extractor / auditor child JSONL. ``session_id``
        # is kept in metadata for completeness — it identifies the
        # parent's session-root span specifically.
        return AgentResult(
            response=response,
            trajectory=trajectory,
            trace_id=session.root_session_id,
            metadata={
                "model": self._model,
                "scenario": self._scenario,
                "max_turns": max_turns,
                "submit_final_report_seen": submission is not None,
                "submission": submission_dump,
                "session_id": session.session_id,
            },
        )


def _build_trajectory(
    *,
    agent_name: str,
    system_prompt: str,
    final_messages: list[Any],
) -> Trajectory:
    """Translate AgentM's session messages to rcabench's Message schema."""
    from agentm.core.abi.messages import (
        AssistantMessage,
        TextContent,
        ToolCallBlock,
        ToolResultMessage,
        UserMessage,
    )

    messages: list[Message] = []
    for msg in final_messages:
        if isinstance(msg, UserMessage):
            text = "".join(
                c.text for c in msg.content if isinstance(c, TextContent)
            )
            messages.append(Message(role="user", content=text))
        elif isinstance(msg, AssistantMessage):
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ToolCallBlock):
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=json.dumps(
                                block.arguments, ensure_ascii=False
                            ),
                        )
                    )
            messages.append(
                Message(
                    role="assistant",
                    content="\n".join(text_parts),
                    tool_calls=tool_calls or None,
                )
            )
        elif isinstance(msg, ToolResultMessage):
            for result_block in msg.content:
                text = "".join(
                    c.text
                    for c in result_block.content
                    if isinstance(c, TextContent)
                )
                messages.append(
                    Message(
                        role="tool",
                        content=text,
                        tool_call_id=result_block.tool_call_id,
                    )
                )

    return Trajectory(
        agent_trajectories=[
            AgentTrajectory(
                agent_name=agent_name,
                system_prompt=system_prompt,
                turns=[Turn(messages=messages)],
            )
        ]
    )


__all__ = ["AgentMAgent"]
