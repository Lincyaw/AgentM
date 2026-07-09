"""AgentM adapter for tau2-bench.

Bridges AgentM model profiles (config.toml) into tau2's HalfDuplexAgent
interface.  Two agent flavors:

  agentm_agent      — standard conversational agent (text ↔ tool calls)
  agentm_agent_solo — solo/ticket mode (tool calls only, no user dialog)

Model resolution: reads ~/.agentm/config.toml (or $AGENTM_HOME/config.toml),
resolves the named profile into (model, api_base, api_key, extra_body), and
calls litellm the same way tau2's built-in LLMAgent does.

Usage (from tau2-bench repo root):

    # register and run programmatically
    python contrib/evals/tau2/run_eval.py \\
        --model litellm-dsv4flash \\
        --domain airline \\
        --num-tasks 5

    # or import the factory and register yourself
    from contrib.evals.tau2.agentm_agent import create_agentm_agent
    registry.register_agent_factory(create_agentm_agent, "agentm_agent")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import toml
from loguru import logger


# ---------------------------------------------------------------------------
# AgentM config.toml resolution
# ---------------------------------------------------------------------------

def _agentm_config_path() -> Path:
    home = os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm"))
    return Path(home) / "config.toml"


def load_agentm_profiles() -> dict[str, dict[str, Any]]:
    cfg_path = _agentm_config_path()
    if not cfg_path.exists():
        logger.warning(f"AgentM config not found at {cfg_path}")
        return {}
    cfg = toml.load(cfg_path)
    return cfg.get("models", {})


def resolve_profile(profile_name: str) -> dict[str, Any]:
    """Resolve an AgentM model profile into litellm-compatible kwargs.

    Returns dict with keys: model, api_base, api_key, extra_body (optional).
    """
    profiles = load_agentm_profiles()
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise KeyError(
            f"AgentM profile {profile_name!r} not found. "
            f"Available: {available}"
        )
    p = profiles[profile_name]
    result: dict[str, Any] = {
        "model": f"openai/{p['model']}",
        "api_base": p.get("base_url"),
        "api_key": p.get("api_key"),
    }
    if p.get("max_output_tokens"):
        result["max_tokens"] = p["max_output_tokens"]
    if p.get("extra_body"):
        result["extra_body"] = p["extra_body"]
    return {k: v for k, v in result.items() if v is not None}


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

from tau2.agent.base_agent import (  # noqa: E402
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (  # noqa: E402
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
)
from tau2.environment.tool import Tool  # noqa: E402
from tau2.utils.llm_utils import generate  # noqa: E402

AGENT_INSTRUCTION = """\
You are a customer service agent that helps the user according to the policy below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Follow the policy. Generate valid JSON for tool arguments."""

SYSTEM_PROMPT_TEMPLATE = """\
# Instructions

{agent_instruction}

# Policy

{domain_policy}"""


class AgentMAgentState:
    __slots__ = ("system_messages", "messages")

    def __init__(
        self,
        system_messages: list[SystemMessage],
        messages: list[APICompatibleMessage],
    ):
        self.system_messages = system_messages
        self.messages = messages


class AgentMAgent(HalfDuplexAgent[AgentMAgentState]):
    """tau2 agent backed by an AgentM model profile."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        agentm_profile: Optional[str] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm_args: dict[str, Any] = dict(llm_args or {})

        if agentm_profile:
            resolved = resolve_profile(agentm_profile)
            self.llm = resolved.pop("model")
            self.llm_args.update(resolved)
            logger.info(
                "AgentM profile {!r} → model={}, api_base={}",
                agentm_profile,
                self.llm,
                self.llm_args.get("api_base", "(default)"),
            )
        else:
            self.llm = llm

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> AgentMAgentState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history)
        return AgentMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: AgentMAgentState,
    ) -> tuple[AssistantMessage, AgentMAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=state.system_messages + state.messages,
            call_name="agentm_agent_response",
            **self.llm_args,
        )
        # tau2 requires either text content OR tool calls, never both.
        # Many models (DeepSeek, GLM, etc.) return both; prefer tool calls.
        if assistant_message.tool_calls and assistant_message.content:
            assistant_message.content = None
        state.messages.append(assistant_message)
        return assistant_message, state


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_agentm_agent(tools, domain_policy, **kwargs):
    """Factory for the registry.  Accepts ``agentm_profile`` in kwargs."""
    return AgentMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm", "openai/gpt-4.1-mini"),
        llm_args=kwargs.get("llm_args"),
        agentm_profile=kwargs.get("agentm_profile"),
    )
