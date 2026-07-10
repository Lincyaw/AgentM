"""tau2_gym — atom that bridges tau2-bench's Gym environment into an AgentM session.

Creates AgentM tools from a tau2 domain's tool schemas so the LLM calls them
natively through AgentM's tool-use loop. Tool execution routes through
``AgentGymEnv.step()``, which runs the tau2 orchestrator (including user
simulator and environment state) in a background thread.

The atom is designed for eval harnesses, not interactive use: one session =
one task. The harness creates a session per task, calls ``session.prompt()``,
collects the result, and shuts down.

Requires tau2-bench installed (``pip install -e /path/to/tau2-bench[gym]``) or
importable via ``TAU2_BENCH_DIR`` env var pointing at the tau2-bench checkout.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from typing import Any, Final

from loguru import logger

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
    description="Bridge tau2-bench Gym environment tools into AgentM",
)

AGENT_INSTRUCTION = """\
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""

SYSTEM_PROMPT_TEMPLATE = """\
<instructions>
{agent_instruction}
</instructions>
<policy>
{policy}
</policy>"""


def _ensure_tau2():
    tau2_dir = os.environ.get(
        "TAU2_BENCH_DIR",
        os.path.expanduser("~/AoyangSpace/tau2-bench"),
    )
    tau2_src = os.path.join(tau2_dir, "src")
    if tau2_src not in sys.path:
        sys.path.insert(0, tau2_src)


async def install(api, config: dict[str, Any]):
    _ensure_tau2()

    from tau2.gym.gym_agent import AgentGymEnv, register_gym_agent

    try:
        register_gym_agent()
    except Exception:
        logger.debug("register_gym_agent already registered or unavailable")

    domain = config["domain"]
    task_id = config["task_id"]
    user_llm = config.get("user_llm", "openai/gpt-4.1-mini")
    user_llm_args = config.get("user_llm_args", {})
    max_steps = config.get("max_steps", 100)

    env = AgentGymEnv(
        domain=domain,
        task_id=str(task_id),
        max_steps=max_steps,
        user_llm=user_llm,
        user_llm_args=user_llm_args,
        all_messages_as_observation=False,
    )

    obs, info = await asyncio.to_thread(env.reset)
    tau2_tools = info.get("tools", [])
    policy = info.get("policy", "")

    if not tau2_tools:
        logger.error(
            "tau2_gym: env.reset() returned 0 tools for domain={}, task={}. "
            "Check that OPENAI_API_BASE/KEY are set before session creation.",
            domain, task_id,
        )

    logger.info(
        "tau2_gym: domain={}, task={}, tools={}, policy={}chars, obs={}chars",
        domain, task_id, len(tau2_tools), len(policy), len(obs),
    )

    bridge = _Tau2Bridge(env=env, initial_obs=obs)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        agent_instruction=AGENT_INSTRUCTION,
        policy=policy,
    )
    api.on(BeforeAgentStartEvent.CHANNEL, lambda evt: setattr(evt, "system", system_prompt))

    def _fmt_tool(name):
        def fmt(args):
            parts = []
            for k, v in args.items():
                parts.append(f"{k}='{v}'" if isinstance(v, str) else f"{k}={json.dumps(v)}")
            return f"{name}({', '.join(parts)})"
        return fmt

    for t in tau2_tools:
        schema = t.openai_schema
        func_schema = schema.get("function", schema)
        tool_name = func_schema["name"]
        params = func_schema.get("parameters", {"type": "object", "properties": {}})

        tool = FunctionTool(
            name=tool_name,
            description=func_schema.get("description", tool_name),
            parameters=params,
            fn=bridge._make_handler(_fmt_tool(tool_name)),
        )
        api.register_tool(tool)

    api.on(DecideTurnActionEvent.CHANNEL, bridge.make_turn_interceptor())

    api.set_service("tau2_bridge", bridge)


class _Tau2Bridge:
    """Holds the gym env and routes tool calls through env.step()."""

    __slots__ = ("env", "last_obs", "terminated", "reward", "reward_info",
                 "_text_intercept_count", "_max_text_intercepts", "_lock")

    def __init__(self, env, initial_obs: str):
        self.env = env
        self.last_obs = initial_obs
        self._lock = threading.Lock()
        self._text_intercept_count = 0
        self._max_text_intercepts = 10
        self.terminated = False
        self.reward = 0.0
        self.reward_info: dict[str, Any] = {}

    def _step_sync(self, action_str: str) -> ToolResult | ToolTerminate:
        with self._lock:
            return self._step_locked(action_str)

    def _step_locked(self, action_str: str) -> ToolResult | ToolTerminate:
        if self.terminated:
            return ToolTerminate(
                result=ToolResult(content=[TextContent(type="text", text="Conversation already ended.")]),
                reason="tau2:done",
            )

        obs, reward, terminated, truncated, info = self.env.step(action_str)
        self.last_obs = obs
        self.terminated = terminated
        if reward > 0 or self.reward == 0.0:
            self.reward = reward
        if info.get("reward_info"):
            try:
                self.reward_info = json.loads(info["reward_info"]) if isinstance(info["reward_info"], str) else info["reward_info"]
            except (json.JSONDecodeError, TypeError):
                pass

        clean_obs = obs
        if clean_obs.startswith("tool: "):
            clean_obs = clean_obs[6:]
        elif clean_obs.startswith("user: "):
            clean_obs = clean_obs[6:]
        result = ToolResult(
            content=[TextContent(type="text", text=clean_obs if clean_obs else "(no response)")],
        )

        if terminated:
            return ToolTerminate(result=result, reason="tau2:done")
        return result

    async def _step(self, action_str: str) -> ToolResult | ToolTerminate:
        return await asyncio.to_thread(self._step_sync, action_str)

    def finalize(self) -> None:
        """Ensure reward is evaluated even if env.step missed it."""
        if self.reward_info:
            return
        try:
            reward, reward_info_str = self.env._get_reward()
            if reward > 0 or self.reward == 0.0:
                self.reward = reward
            if reward_info_str and reward_info_str != "{}":
                self.reward_info = json.loads(reward_info_str) if isinstance(reward_info_str, str) else reward_info_str
        except Exception:
            logger.debug("tau2_gym: finalize _get_reward failed")

    def _make_handler(self, fmt):
        async def handler(args: dict[str, Any]) -> ToolResult | ToolTerminate:
            return await self._step(fmt(args))
        return handler

    def make_turn_interceptor(self):
        """Catch plain-text model outputs and route them through env.step()
        as user-facing replies — matching tau2's native flow where text
        content is routed to the user simulator."""
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
                logger.warning("tau2_gym: text intercept limit reached, stopping")
                return None

            result = bridge._step_sync(text)

            if isinstance(result, ToolTerminate):
                return Stop(cause=ModelEndTurn())

            obs_text = result.content[0].text if result.content else ""
            if obs_text.startswith("user: "):
                obs_text = obs_text[6:]
            inject_msg = UserMessage(
                role="user",
                timestamp=time.time(),
                content=[TextContent(type="text", text=obs_text)],
            )
            return Inject(messages=[inject_msg])

        return handler
