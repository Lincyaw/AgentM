"""tau2_gym — atom that embeds tau2-bench's Environment and UserSimulator
directly into an AgentM session.

No Gym interface, no background orchestrator thread. AgentM's own loop
drives the conversation:

- Domain tools → registered as AgentM tools, each calls
  ``environment.get_response(ToolCall)``.
- Text output → intercepted, routed to ``user_sim.generate_next_message()``,
  response injected as a UserMessage.
- ``###STOP###`` from user sim → session terminates.

One session = one task. The harness sets ``purpose="tau2-eval:{model}:{domain}:{task_id}"``
so the atom can resolve which task to load.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Final

from loguru import logger

from agentm_eval.benchmarks.tau2 import ensure_tau2_path

from agentm.core.abi import (
    BeforeAgentStartEvent,
    DecideTurnActionEvent,
    FunctionTool,
    Inject,
    ModelEndTurn,
    Stop,
    TextContent,
    ToolResult,
    ToolTerminate,
    UserMessage,
)
from agentm.extensions import ExtensionManifest

MANIFEST: Final = ExtensionManifest(
    name="tau2_gym",
    description="Embed tau2-bench Environment + UserSimulator into AgentM",
)

SYSTEM_PROMPT_TEMPLATE = """\
<instructions>
{agent_instruction}
</instructions>
<policy>
{policy}
</policy>"""

AGENT_INSTRUCTION = """\
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""


async def install(api, config: dict[str, Any]):
    ensure_tau2_path()

    parts = api.purpose.split(":")
    if len(parts) < 4 or parts[0] != "tau2-eval":
        raise ValueError(
            f"tau2_gym: expected purpose='tau2-eval:{{model}}:{{domain}}:{{task_id}}', got {api.purpose!r}"
        )
    domain = parts[2]
    task_id = parts[3]
    max_steps = config.get("max_steps", 100)

    from agentm.core.lib import resolve_model_profile
    from tau2.runner.build import build_environment, build_user
    from tau2.runner.helpers import get_tasks

    # Resolve user simulator LLM from a dedicated config.toml profile
    user_sim_profile_name = config.get("user_sim_model", "tau2-user-sim")
    user_llm_args: dict[str, Any] = {}
    profile = resolve_model_profile(user_sim_profile_name)
    if profile:
        user_llm = f"openai/{profile.model}"
        if profile.base_url:
            user_llm_args["api_base"] = profile.base_url
        if profile.api_key:
            user_llm_args["api_key"] = profile.api_key
    else:
        raise ValueError(
            f"tau2_gym: model profile {user_sim_profile_name!r} not found in config.toml"
        )

    # Build tau2 components directly — no orchestrator, no Gym
    tasks = get_tasks(domain, task_ids=[str(task_id)])
    if not tasks:
        raise ValueError(f"tau2_gym: task {task_id} not found in domain {domain}")
    task = tasks[0]

    environment = await asyncio.to_thread(build_environment, domain)
    environment.set_state(
        initialization_data=task.initial_state.initialization_data if task.initial_state else None,
        initialization_actions=task.initial_state.initialization_actions if task.initial_state else None,
        message_history=[],
    )

    user_sim = build_user(
        "user_simulator", environment, task,
        llm=user_llm, llm_args=user_llm_args,
    )
    user_state = user_sim.get_init_state()

    policy = environment.get_policy()
    tau2_tools = environment.get_tools()

    logger.info(
        "tau2_gym: domain={}, task={}, tools={}, policy={}chars",
        domain, task_id, len(tau2_tools), len(policy),
    )

    bridge = _Tau2Bridge(
        environment=environment,
        user_sim=user_sim,
        user_state=user_state,
        task=task,
        domain=domain,
        max_steps=max_steps,
    )

    # System prompt — identical to tau2's native LLMAgent
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        agent_instruction=AGENT_INSTRUCTION,
        policy=policy,
    )
    api.on(BeforeAgentStartEvent.CHANNEL, lambda evt: setattr(evt, "system", system_prompt))

    # Register domain tools
    for t in tau2_tools:
        schema = t.openai_schema
        func_schema = schema.get("function", schema)
        tool_name = func_schema["name"]
        params = func_schema.get("parameters", {"type": "object", "properties": {}})

        api.register_tool(FunctionTool(
            name=tool_name,
            description=func_schema.get("description", tool_name),
            parameters=params,
            fn=bridge.make_tool_handler(tool_name),
        ))

    # Text interceptor — routes model text to user simulator
    api.on(DecideTurnActionEvent.CHANNEL, bridge.make_turn_interceptor())

    api.set_service("tau2_bridge", bridge)


class _Tau2Bridge:
    __slots__ = (
        "environment", "user_sim", "user_state", "task", "domain",
        "trajectory",
        "terminated", "reward", "reward_info",
        "_text_intercept_count", "_max_text_intercepts",
    )

    def __init__(self, environment, user_sim, user_state, task, domain: str, max_steps: int):
        self.environment = environment
        self.user_sim = user_sim
        self.user_state = user_state
        self.task = task
        self.domain = domain
        self.trajectory: list[Any] = []
        self.terminated = False
        self.reward = 0.0
        self.reward_info: dict[str, Any] = {}
        self._text_intercept_count = 0
        self._max_text_intercepts = max_steps

    def make_tool_handler(self, tool_name: str):
        """Create an async handler for a tau2 domain tool."""
        bridge = self

        async def handler(args: dict[str, Any]) -> ToolResult | ToolTerminate:
            return await asyncio.to_thread(bridge._execute_tool, tool_name, args)

        return handler

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResult | ToolTerminate:
        from tau2.data_model.message import (
            AssistantMessage as Tau2AssistantMessage,
            ToolCall as Tau2ToolCall,
        )

        tool_call = Tau2ToolCall(id="", name=tool_name, arguments=args, requestor="assistant")

        agent_msg = Tau2AssistantMessage(
            role="assistant", content=None,
            tool_calls=[tool_call],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self.trajectory.append(agent_msg)

        tool_msg = self.environment.get_response(tool_call)
        self.trajectory.append(tool_msg)

        content = tool_msg.content or ""
        result = ToolResult(content=[TextContent(type="text", text=content)])

        if self.terminated:
            return ToolTerminate(result=result, reason="tau2:done")
        return result

    def _send_text_to_user(self, text: str) -> str | None:
        """Send agent text to user simulator, return user response or None if stopped."""
        from tau2.data_model.message import AssistantMessage as Tau2AssistantMessage
        from tau2.user.user_simulator import UserSimulator

        agent_msg = Tau2AssistantMessage(
            role="assistant",
            content=text,
            tool_calls=None,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self.trajectory.append(agent_msg)

        user_msg, self.user_state = self.user_sim.generate_next_message(
            agent_msg, self.user_state,
        )
        self.trajectory.append(user_msg)

        if UserSimulator.is_stop(user_msg):
            self.terminated = True
            self._evaluate()
            return None

        return user_msg.content

    def _evaluate(self) -> None:
        """Run tau2 evaluation and capture reward."""
        try:
            from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
            from tau2.data_model.simulation import SimulationRun
            from tau2.orchestrator.orchestrator import TerminationReason
            from tau2.utils.utils import get_now

            all_messages = list(self.trajectory)
            sim_run = SimulationRun(
                id="eval",
                task_id=str(self.task.id),
                timestamp=get_now(),
                start_time=get_now(),
                end_time=get_now(),
                duration=0.0,
                termination_reason=TerminationReason.USER_STOP,
                messages=all_messages,
            )

            result = evaluate_simulation(
                simulation=sim_run,
                task=self.task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain=self.domain,
            )
            self.reward = result.reward
            self.reward_info = json.loads(result.model_dump_json())
            logger.info("tau2_gym: evaluation reward={}", self.reward)
        except Exception:
            logger.opt(exception=True).error("tau2_gym: evaluation failed")

    def finalize(self) -> None:
        """Ensure evaluation runs if not already done."""
        if not self.reward_info:
            self._evaluate()

    def make_turn_interceptor(self):
        bridge = self

        def handler(event: DecideTurnActionEvent):
            obs = event.observation
            if obs.assistant_message is None:
                return None
            if obs.tool_outcomes:
                return None
            if not isinstance(obs.default_action, Stop):
                return None
            if not isinstance(obs.default_action.cause, ModelEndTurn):
                return None

            text = ""
            for block in obs.assistant_message.content:
                if isinstance(block, TextContent):
                    text += block.text
            if not text.strip():
                return None

            bridge._text_intercept_count += 1
            if bridge._text_intercept_count > bridge._max_text_intercepts:
                logger.warning("tau2_gym: text intercept limit reached")
                return None

            user_response = bridge._send_text_to_user(text)

            if user_response is None:
                return Stop(cause=ModelEndTurn())

            return Inject(messages=[UserMessage(
                role="user",
                timestamp=time.time(),
                content=[TextContent(type="text", text=user_response)],
            )])

        return handler
