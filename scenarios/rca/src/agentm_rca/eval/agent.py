"""``rcabench-platform`` BaseAgent adapter for the AgentM RCA scenario.

Discovered by ``rcabench-platform``'s ``llm_eval.agents`` entry point and
invoked via ``rca llm-eval run --agent agentm``. Bridges the
``incident + data_dir -> CausalGraph JSON`` contract to an in-process
``AgentSession`` running the local ``rca`` scenario.

Two pieces of glue do the work:

* The orchestrator's ``submit_final_report`` tool grew a required
  ``causal_graph`` field in :mod:`agentm_rca.tools.finalize`. The adapter
  subscribes to ``tool_call`` on the session bus and snatches the argument
  the moment the model emits it.
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
_EMPTY_CAUSAL_GRAPH: dict[str, list[Any]] = {
    "nodes": [],
    "edges": [],
    "root_causes": [],
}


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
            ToolCallEvent,
        )
        from agentm.core.abi.loop import LoopConfig
        from agentm.harness import AgentSession, AgentSessionConfig

        ctx: RunContext | None = kwargs.get("ctx")
        max_turns = _coerce_max_turns(kwargs.get("max_steps"), self._max_turns)
        os.environ["AGENTM_RCA_DATA_DIR"] = data_dir

        captured: dict[str, Any] = {
            "causal_graph": None,
            "system_prompt": "",
        }

        def _on_tool_call(event: ToolCallEvent) -> None:
            if event.tool_name != "submit_final_report":
                return
            cg = event.args.get("causal_graph")
            if cg is None:
                return
            captured["causal_graph"] = cg
            if ctx is not None:
                ctx.emit(
                    {
                        "type": "progress",
                        "message": "submit_final_report received",
                    }
                )

        def _on_before_llm(event: BeforeSendToLlmEvent) -> None:
            if not captured["system_prompt"] and event.system:
                captured["system_prompt"] = event.system

        bus = EventBus()
        bus.on("tool_call", _on_tool_call)
        bus.on("before_send_to_llm", _on_before_llm)

        provider_config: dict[str, str] = {"model": self._model}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            provider_config["base_url"] = base_url

        config = AgentSessionConfig(
            cwd=os.getcwd(),
            provider=("agentm.llm.anthropic", provider_config),
            scenario=self._scenario,
            bus=bus,
            loop_config=LoopConfig(max_turns=max_turns),
        )

        session = await AgentSession.create(config)
        try:
            if ctx is not None:
                ctx.emit({"type": "running", "run_id": session.session_id})
            final_messages = await session.prompt(incident)
        finally:
            await session.shutdown()

        cg = captured["causal_graph"]
        if cg is None:
            response = json.dumps(_EMPTY_CAUSAL_GRAPH)
        elif isinstance(cg, str):
            response = cg
        else:
            response = json.dumps(cg, ensure_ascii=False)

        trajectory = _build_trajectory(
            agent_name=f"agentm:{self._scenario}",
            system_prompt=captured["system_prompt"],
            final_messages=final_messages,
        )

        return AgentResult(
            response=response,
            trajectory=trajectory,
            metadata={
                "model": self._model,
                "scenario": self._scenario,
                "max_turns": max_turns,
                "submit_final_report_seen": cg is not None,
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
