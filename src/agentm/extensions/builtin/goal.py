"""Builtin ``goal`` atom — condition-driven session continuation.

Sets a completion condition; after every turn an independent evaluator
checks whether the condition holds. If not, the evaluator's reason is
injected as a user message and the loop continues. When the condition
is met the goal clears and the session terminates normally.

Two evaluator modes:

1. **Inline** (default): a single LLM call using the session's own
   provider. Fast, no tools — judges only from conversation evidence.
2. **Checker scenario**: spawn a child agent session with its own
   scenario (tools, extensions). The checker can run tests, read files,
   inspect state — anything the scenario equips it with.

   Configure via the atom's ``checker_scenario`` config key::

       - module: agentm.extensions.builtin.goal
         config:
           checker_scenario: goal_checker

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
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Final

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    BeforeAgentStartEvent,
    CommandSpec,
    DecideTurnActionEvent,
    ExtensionAPI,
    Inject,
    LoopConfig,
    MessageEnd,
    Model,
    ModelEndTurn,
    ProviderConfig,
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

_EVAL_SYSTEM: Final[str] = (
    "You are a goal-completion evaluator. You will receive a goal condition "
    "and recent conversation context. Determine whether the condition is met.\n\n"
    "Respond with EXACTLY one of these two formats:\n"
    "MET: <one-line reason>\n"
    "NOT_MET: <one-line reason explaining what remains>\n\n"
    "Be strict: the condition must be clearly satisfied by evidence in the "
    "conversation. If uncertain, respond NOT_MET."
)

_CHECKER_PROMPT_TEMPLATE: Final[str] = (
    "You are an independent goal-completion checker. Your job is to verify "
    "whether a specific condition has been met.\n\n"
    "## Condition to verify\n{condition}\n\n"
    "## Context from the working agent\n{context}\n\n"
    "Use your tools to independently verify the condition. When done, state "
    "your verdict on the FIRST line of your final response in one of these "
    "two formats:\n"
    "MET: <reason>\n"
    "NOT_MET: <reason explaining what remains>"
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


class GoalConfig(BaseModel):
    checker_scenario: str | None = None
    checker_max_turns: int = 10
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
        "via /goal; an independent evaluator checks it after every turn and "
        "keeps the session running until the condition is met. Evaluator "
        "can be an inline LLM call (default) or a full checker agent "
        "session with its own scenario and tools."
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
# Evaluator — inline LLM call (default, no tools)
# ---------------------------------------------------------------------------

async def _evaluate_inline(
    provider: ProviderConfig,
    model: Model,
    condition: str,
    conversation_summary: str,
) -> tuple[bool, str]:
    prompt = (
        f"## Goal condition\n{condition}\n\n"
        f"## Recent conversation\n{conversation_summary}\n\n"
        "Is the goal condition met? Respond MET or NOT_MET with a reason."
    )
    eval_model = replace(model, max_output_tokens=min(512, model.max_output_tokens))
    messages: list[AgentMessage] = [
        UserMessage(
            role="user",
            content=[TextContent(type="text", text=prompt)],
            timestamp=0.0,
        ),
    ]
    final: AssistantMessage | None = None
    try:
        async for event in provider.stream_fn(
            messages=messages,
            model=eval_model,
            tools=[],
            system=_EVAL_SYSTEM,
            signal=None,
            thinking="off",
        ):
            if isinstance(event, MessageEnd):
                final = event.message
    except Exception as exc:  # noqa: BLE001
        logger.warning("goal: inline evaluator failed: {}", exc)
        return False, f"evaluator error: {exc}"

    if final is None:
        return False, "evaluator returned no response"

    return _parse_verdict(final)


# ---------------------------------------------------------------------------
# Shared child-session lifecycle
# ---------------------------------------------------------------------------

async def _prompt_child_session(
    api: ExtensionAPI,
    scenario: str,
    max_turns: int,
    prompt: str,
    purpose: str,
) -> AssistantMessage | None:
    try:
        child = await api.spawn_child_session(
            cwd=api.cwd,
            scenario=scenario,
            provider=None,
            loop_config=LoopConfig(max_turns=max_turns),
            purpose=purpose,
            lineage={
                "kind": purpose,
                "parent_session_id": api.session_id,
                "root_session_id": api.root_session_id,
                "purpose": purpose,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("goal: {} spawn failed: {}", purpose, exc)
        return None

    try:
        final_messages = await child.prompt(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("goal: {} failed: {}", purpose, exc)
        return None
    finally:
        try:
            await child.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.debug("goal: {} shutdown (ignored): {}", purpose, exc)

    if not final_messages:
        return None
    for msg in reversed(final_messages):
        if isinstance(msg, AssistantMessage):
            return msg
    return None


# ---------------------------------------------------------------------------
# Evaluator — checker agent session (configurable scenario)
# ---------------------------------------------------------------------------

async def _evaluate_checker(
    api: ExtensionAPI,
    scenario: str,
    max_turns: int,
    condition: str,
    conversation_summary: str,
) -> tuple[bool, str]:
    prompt = _CHECKER_PROMPT_TEMPLATE.format(
        condition=condition,
        context=conversation_summary,
    )
    msg = await _prompt_child_session(api, scenario, max_turns, prompt, "goal_checker")
    if msg is None:
        return False, "checker produced no response"
    return _parse_verdict(msg)


# ---------------------------------------------------------------------------
# Shared verdict parsing
# ---------------------------------------------------------------------------

def _parse_verdict(msg: AssistantMessage) -> tuple[bool, str]:
    text = assistant_text(msg).strip()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("MET:"):
            return True, stripped[4:].strip()
        if stripped.upper().startswith("NOT_MET:"):
            return False, stripped[8:].strip()

    return False, text if text else "evaluator gave ambiguous response"


def _parse_condition(msg: AssistantMessage) -> str | None:
    text = assistant_text(msg).strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("CONDITION:"):
            cond = stripped[10:].strip()
            return cond or None
    return None


def _summarize_recent_turns(
    observation: DecideTurnActionEvent,
) -> str:
    msg = observation.observation.assistant_message
    if msg is None:
        return "(no assistant output this turn)"
    parts: list[str] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolCallBlock):
            parts.append(
                f"[tool_call: {block.name}("
                f"{json.dumps(block.arguments, ensure_ascii=False)})]"
            )
    return "\n".join(parts) or "(no text output)"


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def install(api: ExtensionAPI, config: GoalConfig) -> None:
    checker_scenario = config.checker_scenario
    checker_max_turns = config.checker_max_turns
    state: _GoalState | None = None

    async def _run_evaluation(conversation_text: str, condition: str) -> tuple[bool, str]:
        if checker_scenario is not None:
            return await _evaluate_checker(
                api, checker_scenario, checker_max_turns,
                condition, conversation_text,
            )
        provider = api.provider
        model = api.model
        if provider is None or model is None:
            logger.warning("goal: no provider/model for inline evaluation")
            return False, "no provider available"
        return await _evaluate_inline(
            provider, model, condition, conversation_text,
        )

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
                mode = f"checker={checker_scenario}" if checker_scenario else "inline"
                msg += f"\nEvaluator: {mode}"
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
        mode = f"checker={checker_scenario}" if checker_scenario else "inline"
        cmd_api.send_user_message(f"Goal set ({mode}): {stripped}")

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

        conversation_text = _summarize_recent_turns(event)
        is_met, reason = await _run_evaluation(conversation_text, state.condition)
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
                reply = await _prompt_child_session(
                    api, init_scenario, init_max_turns, prompt, "goal_derivation",
                )
                condition = _parse_condition(reply) if reply else None
                if condition:
                    state = _GoalState(condition=condition)
                    mode = f"checker={checker_scenario}" if checker_scenario else "inline"
                    logger.info("goal: auto-initialized ({}) — {}", mode, condition)

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
