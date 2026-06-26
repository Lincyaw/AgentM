"""Builtin ``goal`` atom — condition-driven session continuation.

Sets a completion condition; after every turn an independent evaluator
checks whether the condition holds. If not, the evaluator's reason is
injected as a user message and the loop continues. When the condition
is met the goal clears and the session terminates normally.

Evaluator: a child agent session with ``trace_query`` tools that can
inspect the parent session's trajectory via ClickHouse. The checker
decides what to query, judges the evidence, and outputs a verdict.

Hard stops (``MaxTurnsExhausted``, ``BudgetExhausted``, ``SignalAborted``)
are ``final=True`` and cannot be overridden — they are the safety net.

Interface:

* ``/goal <condition>`` — set (or replace) the active goal.
* ``/goal`` — show status.
* ``/goal clear`` — remove the goal before it's met.

§11: single file; ``MANIFEST`` + ``install(api, config)``; no atom-to-atom
imports; ``core.abi`` only; no ``core.runtime.*`` / ``core._internal``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Final

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    AssistantMessage,
    BeforeAgentStartEvent,
    CommandSpec,
    DecideTurnActionEvent,
    ExtensionAPI,
    Inject,
    LoopConfig,
    ModelEndTurn,
    Stop,
    TextContent,
    ToolCallBlock,
    ToolTerminated,
    UserMessage,
    text_message,
)
from agentm.core.lib import assistant_text
from agentm.extensions import ExtensionManifest

_CLEAR_ALIASES: Final[frozenset[str]] = frozenset({
    "clear", "stop", "off", "reset", "none", "cancel",
})

_CHECKER_PROMPT_TEMPLATE: Final[str] = (
    "You are an independent goal-completion checker. Your job is to verify "
    "whether the parent agent's work satisfies a specific condition.\n\n"
    "## Condition to verify\n{condition}\n\n"
    "You have tools to query the parent session's trajectory:\n"
    "- `list_turns` — overview of all turns (tools called, token counts)\n"
    "- `read_turn` — read actual messages (filter by role, paginate)\n"
    "- `get_tool_calls` — query specific tool calls with args and results\n\n"
    "Investigate the trajectory to determine whether the condition is met. "
    "Focus on the agent's actual output and deliverables, not procedural steps.\n\n"
    "When done, call `submit_verdict` with your structured verdict. "
    "Do NOT output your verdict as text — you MUST call the tool."
)

_AUTO_INIT_PROMPT: Final[str] = (
    "You are a goal-condition formulator. Your job is to derive a completion "
    "condition, not to execute the task itself.\n\n"
    "You may use tools to do lightweight exploration — list files, read "
    "schemas, run simple queries to understand what data or constraints "
    "exist. But keep it shallow: just enough to discover the implicit "
    "requirements, not to solve the problem.\n\n"
    "## System prompt\n{system}\n\n"
    "## User request\n{user_text}\n\n"
    "A good condition goes beyond surface-level deliverables ('called tool X'). "
    "Look for implicit quality constraints hidden in the task and its domain. "
    "Examples of what to look for in different domains:\n\n"
    "- **Investigation/analysis tasks**: Does the domain imply exhaustive "
    "coverage (all anomalies explained, not just the first)? Are there "
    "multiple independent phenomena that each need separate explanation? "
    "Is step-by-step causal reasoning with multi-source evidence required?\n"
    "- **Coding tasks**: Are there existing tests, linters, type checkers, "
    "or CI that must pass? Does the codebase have conventions (naming, "
    "architecture, module boundaries) the change must respect? Is there a "
    "spec, issue, or acceptance criteria the implementation must satisfy?\n"
    "- **Any task with a structural output contract**: Must the deliverable "
    "conform to a schema, format, or protocol?\n\n"
    "These are examples — derive the actual constraints from the specific "
    "task context. Keep exploration lightweight — just enough to find the "
    "constraints, then formulate the condition.\n\n"
    "When ready, respond with EXACTLY one line starting with:\n"
    "CONDITION: <the completion condition>"
)

_TRACE_QUERY_EXT: Final[tuple[str, dict[str, Any]]] = (
    "agentm.extensions.builtin.trace_query", {},
)


class GoalConfig(BaseModel):
    condition: str | None = None
    checker_scenario: str = "local"
    checker_max_turns: int = 10
    checker_prompt: str | None = None
    auto_init: bool = False
    auto_init_scenario: str | None = None
    auto_init_max_turns: int = 10


@dataclass
class _GoalState:
    condition: str
    started_at: float = field(default_factory=time.monotonic)
    turns_evaluated: int = 0
    last_reason: str = ""
    achieved: bool = False


MANIFEST = ExtensionManifest(
    name="goal",
    description=(
        "Condition-driven session continuation: set a completion condition "
        "via /goal; an independent checker agent inspects the parent "
        "session's trajectory after every turn and keeps the session "
        "running until the condition is met."
    ),
    registers=(
        "event:decide_turn_action",
        "event:before_agent_start",
        "command:goal",
    ),
    config_schema=GoalConfig,
    requires=(),
)


# ---------------------------------------------------------------------------
# Child-session helpers
# ---------------------------------------------------------------------------

async def _prompt_child_session_messages(
    api: ExtensionAPI,
    scenario: str,
    max_turns: int,
    prompt: str,
    purpose: str,
    extra_extensions: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[AgentMessage] | None:
    config = AgentSessionConfig(
        cwd=api.cwd,
        scenario=scenario,
        extra_extensions=extra_extensions or [],
        purpose=purpose,
        lineage={
            "kind": purpose,
            "parent_session_id": api.session_id,
            "root_session_id": api.root_session_id,
            "purpose": purpose,
        },
        loop_config=LoopConfig(max_turns=max_turns),
    )
    try:
        child = await api.spawn_child_session(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("goal: {} spawn failed: {}", purpose, exc)
        return None

    try:
        return await child.prompt(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("goal: {} failed: {}", purpose, exc)
        return None
    finally:
        try:
            await child.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.debug("goal: {} shutdown (ignored): {}", purpose, exc)


# ---------------------------------------------------------------------------
# Checker evaluation (child agent with trace_query tools)
# ---------------------------------------------------------------------------

async def _evaluate_checker(
    api: ExtensionAPI,
    scenario: str,
    max_turns: int,
    condition: str,
    checker_prompt_override: str | None = None,
) -> tuple[bool, str]:
    if checker_prompt_override:
        prompt = checker_prompt_override.format(condition=condition)
    else:
        prompt = _CHECKER_PROMPT_TEMPLATE.format(condition=condition)
    messages = await _prompt_child_session_messages(
        api, scenario, max_turns, prompt, "goal_checker",
        extra_extensions=[_TRACE_QUERY_EXT],
    )
    if messages is None:
        return False, "checker produced no response"
    return _parse_verdict_from_tool_call(messages)


# ---------------------------------------------------------------------------
# Verdict parsing — structured tool call
# ---------------------------------------------------------------------------

def _parse_verdict_from_tool_call(
    messages: list[AgentMessage],
) -> tuple[bool, str]:
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name == "submit_verdict":
                args = block.arguments
                met = bool(args.get("met", False))
                reason = str(args.get("reason", ""))
                unexplained = args.get("unexplained", [])
                if not met and unexplained:
                    reason = f"{reason} (unexplained: {', '.join(unexplained)})"
                return met, reason
    return False, "checker did not call submit_verdict"


def _parse_condition(msg: AssistantMessage) -> str | None:
    text = assistant_text(msg).strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("CONDITION:"):
            cond = stripped[10:].strip()
            return cond or None
    return None


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def install(api: ExtensionAPI, config: GoalConfig) -> None:
    checker_scenario = config.checker_scenario
    checker_max_turns = config.checker_max_turns
    checker_prompt_override = config.checker_prompt
    state: _GoalState | None = None

    if config.condition:
        state = _GoalState(condition=config.condition)
        logger.info("goal: static condition — {}", config.condition)

    # -- slash command: /goal -----------------------------------------------

    async def _goal_command(args: str, cmd_api: ExtensionAPI) -> None:
        nonlocal state
        stripped = args.strip()

        if not stripped:
            if state is None:
                cmd_api.send_user_message("No active goal.")
                return
            elapsed = time.monotonic() - state.started_at
            if state.achieved:
                cmd_api.send_user_message(
                    f"Goal achieved in {state.turns_evaluated} turn(s) "
                    f"({elapsed:.0f}s): {state.condition}"
                )
            else:
                msg = (
                    f"Active goal ({state.turns_evaluated} turn(s), "
                    f"{elapsed:.0f}s): {state.condition}"
                )
                if state.last_reason:
                    msg += f"\nLast evaluation: {state.last_reason}"
                msg += f"\nChecker: scenario={checker_scenario}"
                cmd_api.send_user_message(msg)
            return

        if stripped.lower() in _CLEAR_ALIASES:
            if state is not None and not state.achieved:
                state = None
                cmd_api.send_user_message("Goal cleared.")
            else:
                cmd_api.send_user_message("No active goal to clear.")
            return

        state = _GoalState(condition=stripped)
        cmd_api.send_user_message(
            f"Goal set (checker scenario={checker_scenario}): {stripped}"
        )

    api.register_command(
        "goal",
        CommandSpec(
            description="Set a completion condition that keeps the session running until met.",
            handler=_goal_command,
        ),
    )

    # -- decide_turn_action: evaluate after each turn -----------------------

    _init_task: asyncio.Task[None] | None = None

    async def _on_decide(event: DecideTurnActionEvent) -> Any:
        nonlocal state
        if _init_task is not None and not _init_task.done():
            await _init_task
        if state is None or state.achieved:
            return None

        default = event.observation.default_action
        if not isinstance(default, Stop):
            return None
        cause = default.cause
        if cause.final:
            return None
        if not isinstance(cause, (ModelEndTurn, ToolTerminated)):
            return None

        is_met, reason = await _evaluate_checker(
            api, checker_scenario, checker_max_turns, state.condition,
            checker_prompt_override,
        )
        state.turns_evaluated += 1
        state.last_reason = reason

        if is_met:
            state.achieved = True
            logger.info("goal: condition met — {}", reason)
            return None

        logger.info("goal: not met — {}", reason)
        return Inject(messages=[
            text_message(
                f"[Goal not met] {reason}\n\n"
                f"Continue working toward: {state.condition}"
            ),
        ])

    api.on(DecideTurnActionEvent.CHANNEL, _on_decide)

    # -- auto-init: derive goal from user prompt at session start -----------

    if config.auto_init:
        init_scenario = config.auto_init_scenario or api.scenario
        init_max_turns = config.auto_init_max_turns
        if init_scenario is None:
            logger.warning("goal: auto_init requires a scenario but none is set — skipping")

        if init_scenario is not None and api.parent_session_id is None:

            async def _derive_goal(system: str, user_text: str) -> None:
                nonlocal state
                prompt = _AUTO_INIT_PROMPT.format(system=system, user_text=user_text)
                messages = await _prompt_child_session_messages(
                    api, init_scenario, init_max_turns, prompt, "goal_derivation",
                )
                reply: AssistantMessage | None = None
                if messages:
                    for msg in reversed(messages):
                        if isinstance(msg, AssistantMessage):
                            reply = msg
                            break
                condition = _parse_condition(reply) if reply else None
                if condition:
                    state = _GoalState(condition=condition)
                    logger.info("goal: auto-initialized — {}", condition)

            def _on_before_start(event: BeforeAgentStartEvent) -> None:
                if state is not None:
                    return
                system = event.system or ""
                if not system:
                    return

                user_parts: list[str] = []
                for msg in event.messages:
                    if isinstance(msg, UserMessage):
                        for block in msg.content:
                            if isinstance(block, TextContent):
                                user_parts.append(block.text)
                user_text = "\n".join(user_parts) if user_parts else "(see system prompt)"

                nonlocal _init_task
                _init_task = asyncio.create_task(_derive_goal(system, user_text))

            api.on(BeforeAgentStartEvent.CHANNEL, _on_before_start)
