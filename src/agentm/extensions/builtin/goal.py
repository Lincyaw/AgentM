# code-health: ignore-file[AM025] -- atom tools validate untyped tool, config, and service payloads
"""Builtin ``goal`` atom -- condition-driven session continuation.

Sets a completion condition; after every turn an independent evaluator
checks whether the condition holds.  If not, the evaluator's reason is
injected as a user message and the loop continues.  When the condition
is met the goal clears and the session terminates normally.

Evaluator: a child agent session with ``trace_query`` tools that can
inspect the parent session's trajectory.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    AssistantMessage,
    AtomAPI,
    AtomInstallPriority,
    BeforeSendEvent,
    DecideEvent,
    FunctionTool,
    Inject,
    JsonValue,
    LoopConfig,
    ModelEndTurn,
    Stop,
    TextContent,
    Tool,
    ToolCallBlock,
    ToolResult,
    ToolTerminate,
    ToolTerminated,
    UserMessage,
    text_message,
)
from agentm.extensions import ExtensionManifest

GOAL_CONDITION_SERVICE: Final = "goal.condition"

_ORACLE_PRINCIPLES: Final[str] = (
    "Shared acceptance discipline:\n"
    "- The task's stated requirements are the sole authority.\n"
    "- An in-repo test that encodes behavior the task is changing is stale; "
    "its failure is not a defect.\n"
    "- Verification rests on evidence that demonstrates the requirement and "
    "that this environment can actually run."
)

_CHECKER_PROMPT_TEMPLATE: Final[str] = (
    "You are an independent goal-completion checker. Verify whether the "
    "parent agent's work satisfies the condition below.\n\n"
    "## Condition to verify\n{condition}\n\n"
    "You have tools to query the parent session's trajectory:\n"
    "- `list_turns` -- overview of all turns\n"
    "- `read_turn` -- read actual messages\n"
    "- `get_tool_calls` -- query specific tool calls\n\n"
    "Investigate the trajectory to determine whether the condition is met.\n\n"
    + _ORACLE_PRINCIPLES
    + "\n\nRequire real, positive evidence for each part of the condition.\n\n"
    "When done, call `submit_verdict` with your structured verdict. "
    "Do NOT output your verdict as text."
)

_AUTO_INIT_PROMPT: Final[str] = (
    "You are a goal-condition formulator. Derive a compound completion "
    "condition -- not to execute the task itself.\n\n" + _ORACLE_PRINCIPLES + "\n\n"
    "## Parent system prompt (background only)\n{system}\n\n"
    "## Original user request\n{user_text}\n\n"
    "When ready, call the `submit_result` tool with your structured output."
)

_TRACE_QUERY_EXT: Final[tuple[str, dict[str, JsonValue]]] = (
    "agentm.extensions.builtin.trace_query",
    {},
)


class _CheckerVerdictArgs(BaseModel):
    met: bool = Field(description="True only if the condition is fully satisfied.")
    reason: str = Field(description="Concrete justification citing the trajectory.")
    unexplained: list[str] = Field(
        default_factory=list,
        description="Condition sub-requirements you could not confirm.",
    )


async def _checker_submit_verdict(args: dict[str, JsonValue]) -> ToolTerminate:
    parsed = _CheckerVerdictArgs.model_validate(args)
    return ToolTerminate(
        result=ToolResult(
            content=[TextContent(type="text", text=parsed.model_dump_json())]
        ),
        reason="goal_checker:verdict_submitted",
    )


_CHECKER_VERDICT_TOOL: Final = FunctionTool(
    name="submit_verdict",
    description=(
        "Submit your final verdict on whether the goal condition is met. "
        "This terminates the checker session."
    ),
    parameters=_CheckerVerdictArgs,
    fn=_checker_submit_verdict,
)

_CHECKER_TURN_REMINDER_EXT: Final[tuple[str, dict[str, JsonValue]]] = (
    "agentm.extensions.builtin.turn_reminder",
    {"warn_within": 4, "finalize_tool": "submit_verdict"},
)

_DERIVER_TURN_REMINDER_EXT: Final[tuple[str, dict[str, JsonValue]]] = (
    "agentm.extensions.builtin.turn_reminder",
    {"warn_within": 6, "finalize_tool": "submit_result"},
)


class _ConditionSchema(BaseModel):
    goal: str = Field(description="The final acceptance criterion.")
    verification_method: str = Field(description="How to verify the goal is met.")
    invariants: list[str] = Field(description="Correctness constraints.")
    checklist: list[str] = Field(description="Yes/no verification questions.")


_STRUCTURED_OUTPUT_EXT: Final[tuple[str, dict[str, JsonValue]]] = (
    "agentm.extensions.builtin.structured_output",
    {"result_schema": _ConditionSchema.model_json_schema()},
)

_AGENT_ENV_SESSION_SERVICE: Final[str] = "agent_env.session_id"


class GoalConfig(BaseModel):
    condition: str | None = None
    checker_scenario: str = "local"
    checker_max_turns: int = 10
    checker_prompt: str | None = None
    checker_retries: int = 1
    auto_init: bool = False
    auto_init_scenario: str | None = None
    auto_init_max_turns: int = 16
    auto_init_retries: int = 3
    max_rejects: int = 5


class _ConditionPayload(BaseModel):
    goal: str
    verification_method: str = ""
    invariants: list[str] = []
    checklist: list[str] = []


@dataclass(slots=True)
class _GoalState:
    condition: str
    goal_summary: str = ""
    verification_method: str = ""
    started_at: float = field(default_factory=time.monotonic)
    turns_evaluated: int = 0
    consecutive_rejects: int = 0
    last_reason: str = ""
    achieved: bool = False
    released: bool = False


MANIFEST = ExtensionManifest(
    name="goal",
    description=(
        "Condition-driven session continuation: set a completion condition "
        "via atom config; an independent checker agent inspects the parent "
        "session's trajectory after every turn and keeps the session "
        "running until the condition is met."
    ),
    registers=(
        "event:decide",
        "event:before_send",
    ),
    config_schema=GoalConfig,
    requires=(),
    priority=AtomInstallPriority.NORMAL,
)


# ---------------------------------------------------------------------------
# Child-session helpers
# ---------------------------------------------------------------------------


async def _prompt_child(
    api: AtomAPI,
    scenario: str,
    max_turns: int,
    prompt: str,
    purpose: str,
    extra_extensions: list[tuple[str, dict[str, JsonValue]]] | None = None,
    extra_tools: Sequence[Tool] | None = None,
    atom_config_overrides: dict[str, dict[str, JsonValue]] | None = None,
) -> list[AgentMessage] | None:
    config = AgentSessionConfig(
        cwd=api.ctx.cwd,
        scenario=scenario,
        extra_extensions=list(extra_extensions) if extra_extensions else [],  # type: ignore[arg-type]
        extra_tools=list(extra_tools) if extra_tools else [],
        atom_config_overrides=atom_config_overrides or {},
        purpose=purpose,
        parent_session_id=api.ctx.session_id,
        root_session_id=api.ctx.root_session_id,
        loop_config=LoopConfig(max_turns=max_turns),
    )
    try:
        child = await api.spawn_child_session(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("goal: {} spawn failed: {}", purpose, exc)
        return None
    try:
        return await child.run(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("goal: {} failed: {}", purpose, exc)
        return None
    finally:
        try:
            await child.shutdown()
        except Exception:  # noqa: BLE001, S110
            pass


def _env_attach_overrides(api: AtomAPI) -> dict[str, dict[str, JsonValue]]:
    session_id = api.services.get(_AGENT_ENV_SESSION_SERVICE)
    if not isinstance(session_id, str) or not session_id:
        return {}
    return {"operations": {"attach_session": session_id}}


# ---------------------------------------------------------------------------
# Checker evaluation
# ---------------------------------------------------------------------------


async def _evaluate_checker(
    api: AtomAPI,
    scenario: str,
    max_turns: int,
    condition: str,
    checker_prompt_override: str | None = None,
) -> tuple[bool | None, str]:
    if checker_prompt_override:
        prompt = checker_prompt_override.format(condition=condition)
    else:
        prompt = _CHECKER_PROMPT_TEMPLATE.format(condition=condition)
    messages = await _prompt_child(
        api,
        scenario,
        max_turns,
        prompt,
        "goal_checker",
        extra_extensions=[_TRACE_QUERY_EXT, _CHECKER_TURN_REMINDER_EXT],
        extra_tools=[_CHECKER_VERDICT_TOOL],
        atom_config_overrides=_env_attach_overrides(api),
    )
    if messages is None:
        return None, "checker produced no response"
    return _parse_verdict(messages)


def _parse_verdict(messages: list[AgentMessage]) -> tuple[bool | None, str]:
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name == "submit_verdict":
                args = block.arguments
                met = bool(args.get("met", False))
                reason = str(args.get("reason", ""))
                unexplained: list[str] = list(args.get("unexplained", []))  # type: ignore[arg-type]
                if not met and unexplained:
                    reason = f"{reason} (unexplained: {', '.join(unexplained)})"
                return met, reason
    return None, "checker did not call submit_verdict"


def _parse_structured_payload(
    messages: list[AgentMessage],
) -> _ConditionPayload:
    last_error: Exception | None = None
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name == "submit_result":
                result = block.arguments.get("result", block.arguments)
                if isinstance(result, str):
                    try:
                        decoded = json.loads(result)
                    except json.JSONDecodeError:
                        decoded = None
                    if isinstance(decoded, dict):
                        result = decoded
                if not isinstance(result, dict):
                    last_error = ValueError(f"bad submit_result: {result!r}")
                    continue
                try:
                    return _ConditionPayload.model_validate(result)
                except ValidationError as exc:
                    last_error = exc
                    continue
    if last_error is not None:
        raise last_error
    raise ValueError("no submit_result tool call found")


def _format_condition(payload: _ConditionPayload) -> str:
    parts = [f"## Goal\n{payload.goal}"]
    if payload.verification_method:
        parts.append(f"## Verification method\n{payload.verification_method}")
    if payload.invariants:
        inv = "\n".join(f"{i + 1}. {v}" for i, v in enumerate(payload.invariants))
        parts.append(f"## Invariants\n{inv}")
    if payload.checklist:
        chk = "\n".join(f"- {c}" for c in payload.checklist)
        parts.append(f"## Checklist\n{chk}")
    return "\n\n".join(parts)


def _collect_user_text(messages: list[AgentMessage]) -> str:
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, UserMessage):
            for block in msg.content:
                if isinstance(block, TextContent):
                    parts.append(block.text)
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class _GoalRuntime:
    def __init__(self, api: AtomAPI, config: GoalConfig) -> None:
        self._api = api
        self._checker_scenario = config.checker_scenario
        self._checker_max_turns = config.checker_max_turns
        self._checker_prompt_override = config.checker_prompt
        self._checker_retries = max(config.checker_retries, 0)
        self._auto_init = config.auto_init
        self._auto_init_scenario = config.auto_init_scenario or api.ctx.scenario
        self._auto_init_max_turns = config.auto_init_max_turns
        self._auto_init_retries = max(config.auto_init_retries, 0)
        self._max_rejects = max(config.max_rejects, 1)
        self._auto_init_started = False
        self._state: _GoalState | None = None

        if config.condition:
            self._state = _GoalState(condition=config.condition)
            api.services.register(
                GOAL_CONDITION_SERVICE, config.condition, scope="session"
            )

    def install(self) -> None:
        if self._auto_init and self._auto_init_scenario is None:
            logger.warning("goal: auto_init requires a scenario — disabled")
            self._auto_init = False
        self._api.on(BeforeSendEvent.CHANNEL, self._on_before_send)
        self._api.on(DecideEvent.CHANNEL, self._on_decide)

    async def _on_before_send(self, event: BeforeSendEvent) -> dict[str, str] | None:
        if (
            self._auto_init
            and not self._auto_init_started
            and self._state is None
            and self._api.ctx.parent_session_id is None
        ):
            user_text = _collect_user_text(list(event.messages))
            self._auto_init_started = True
            if user_text:
                await self._derive_goal(event.system or "", user_text)

        state = self._state
        if state is None or state.achieved or state.released:
            return None
        system = (
            (event.system or "") + "\n\n## Completion criteria\n"
            "An independent checker will verify your work against the "
            "condition below before the session can end.\n\n" + state.condition
        )
        return {"system": system}

    async def _derive_goal(self, system: str, user_text: str) -> None:
        if self._auto_init_scenario is None:
            return
        prompt = _AUTO_INIT_PROMPT.format(
            system=system,
            user_text=user_text,
            max_turns=self._auto_init_max_turns,
        )
        for attempt in range(self._auto_init_retries + 1):
            messages = await _prompt_child(
                self._api,
                self._auto_init_scenario,
                self._auto_init_max_turns,
                prompt,
                "goal_derivation",
                extra_extensions=[_STRUCTURED_OUTPUT_EXT, _DERIVER_TURN_REMINDER_EXT],
                atom_config_overrides=_env_attach_overrides(self._api),
            )
            if messages is None:
                continue
            try:
                payload = _parse_structured_payload(messages)
            except (TypeError, ValueError, ValidationError) as exc:
                logger.warning(
                    "goal: auto_init attempt {} failed: {}", attempt + 1, exc
                )
                continue
            if self._state is not None:
                return
            condition = _format_condition(payload)
            self._state = _GoalState(
                condition=condition,
                goal_summary=payload.goal,
                verification_method=payload.verification_method,
            )
            self._api.services.register(
                GOAL_CONDITION_SERVICE, condition, scope="session"
            )
            logger.info("goal: auto-initialized — {}", condition[:200])
            return
        logger.warning("goal: auto_init exhausted all retries")

    async def _on_decide(self, event: DecideEvent) -> Inject | None:
        state = self._state
        if state is None or state.achieved or state.released:
            return None

        default = event.observation.default_action
        if not isinstance(default, Stop):
            return None
        cause = default.cause
        if cause.session_terminal:
            return None
        if not isinstance(cause, (ModelEndTurn, ToolTerminated)):
            return None

        is_met: bool | None = None
        reason = ""
        for attempt in range(self._checker_retries + 1):
            is_met, reason = await _evaluate_checker(
                self._api,
                self._checker_scenario,
                self._checker_max_turns,
                state.condition,
                self._checker_prompt_override,
            )
            if is_met is not None:
                break
            logger.warning(
                "goal: checker infra failure (attempt {}/{}) — {}",
                attempt + 1,
                self._checker_retries + 1,
                reason,
            )
        state.turns_evaluated += 1
        state.last_reason = reason

        if is_met is None:
            logger.warning("goal: checker unavailable — allowing stop")
            return None

        if is_met:
            state.achieved = True
            state.consecutive_rejects = 0
            logger.info("goal: condition met — {}", reason)
            return None

        state.consecutive_rejects += 1
        if state.consecutive_rejects >= self._max_rejects:
            state.released = True
            logger.warning(
                "goal: releasing after {} rejects — {}",
                state.consecutive_rejects,
                reason,
            )
            return None

        logger.info(
            "goal: not met ({}/{}) — {}",
            state.consecutive_rejects,
            self._max_rejects,
            reason,
        )
        inject_text = (
            f"[Goal not met ({state.consecutive_rejects}/{self._max_rejects})] {reason}"
        )
        if state.verification_method:
            inject_text += f"\n\nVerification method: {state.verification_method}"
        else:
            inject_text += (
                f"\n\nContinue working toward: {state.goal_summary or state.condition}"
            )
        return Inject(messages=(text_message(inject_text),))


def install(api: AtomAPI, config: GoalConfig) -> None:
    _GoalRuntime(api, config).install()
