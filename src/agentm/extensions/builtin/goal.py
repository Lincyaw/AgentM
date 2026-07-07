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

import time
from dataclasses import dataclass, field
from typing import Any, Final

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    AssistantMessage,
    BeforeSendToLlmEvent,
    CommandSpec,
    DecideTurnActionEvent,
    ExtensionAPI,
    FunctionTool,
    Inject,
    LoopConfig,
    ModelEndTurn,
    Stop,
    TextContent,
    ToolCallBlock,
    ToolResult,
    ToolTerminate,
    ToolTerminated,
    UserMessage,
    text_message,
)
from agentm.extensions import ExtensionManifest

GOAL_CONDITION_SERVICE: Final = "goal.condition"

_CLEAR_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "clear",
        "stop",
        "off",
        "reset",
        "none",
        "cancel",
    }
)

_CHECKER_PROMPT_TEMPLATE: Final[str] = (
    "You are an independent goal-completion checker. Your job is to verify "
    "whether the parent agent's work satisfies a specific condition.\n\n"
    "## Condition to verify\n{condition}\n\n"
    "You have tools to query the parent session's trajectory:\n"
    "- `list_turns` — overview of all turns (tools called, token counts)\n"
    "- `read_turn` — read actual messages (filter by role, paginate)\n"
    "- `get_tool_calls` — query specific tool calls with args and results\n\n"
    "Investigate the trajectory to determine whether the condition is met.\n\n"
    "You are checking the final result, not the process. But the result must "
    "be backed by evidence in the trajectory:\n"
    "- If the condition requires tests to pass, the agent must have actually "
    "executed those tests AND the output must show they passed. Use "
    "`get_tool_calls` to find the execution and check the output.\n"
    "- A timeout, crash, or inconclusive test run is NOT a pass — do not "
    "infer that code 'would pass' based on reading the source. Only actual "
    "passing output counts.\n"
    "- If the condition requires specific output or artifacts, verify they "
    "exist in the trajectory.\n\n"
    "Judge pragmatically:\n"
    "- Focus on the primary acceptance criteria (tests pass, correct output). "
    "Ancillary modifications that help achieve the goal (e.g., fixing a "
    "dependency to unblock tests) are acceptable unless the spec explicitly "
    "forbids them.\n"
    "- If all required tests pass and the implementation is correct, do not "
    "reject solely because the agent touched files beyond the minimum set. "
    "Only reject for scope violations that are explicitly stated in the "
    "original specification.\n"
    "- Distinguish between 'the spec says not to do X' (reject) and 'the "
    "spec does not mention X' (allow).\n\n"
    "When done, call `submit_verdict` with your structured verdict. "
    "Do NOT output your verdict as text — you MUST call the tool."
)

_AUTO_INIT_PROMPT: Final[str] = (
    "You are a goal-condition formulator. Your job is to derive a compound "
    "completion condition — not to execute the task itself.\n\n"
    "Derive the condition for the ORIGINAL user request only. Do not make "
    "`submit_result`, structured output, or this meta-task itself part of the "
    "completion goal unless the original user request explicitly requires it.\n\n"
    "Treat the system prompt as background policy/context, not as the task. "
    "Do not derive the goal from runtime-context instructions such as "
    "working-directory guidance unless the original user request is itself "
    "about that runtime context.\n\n"
    "You may use tools to do lightweight exploration — list files, read "
    "schemas, run simple queries to understand what data or constraints "
    "exist. But keep it shallow: just enough to discover the implicit "
    "requirements, not to solve the problem.\n\n"
    "## Parent system prompt (background only)\n{system}\n\n"
    "## Original user request\n{user_text}\n\n"
    "Your output has four parts:\n\n"
    "### 1. Completion goal\n"
    "The final acceptance criterion (tests pass, output matches spec, etc.).\n\n"
    "### 2. Verification method\n"
    "How the goal should be verified. Two steps:\n"
    "a) Read the source material (e.g., INSTRUCTION.md) and quote the "
    "specific passages that describe testing or validation procedures.\n"
    "b) **Check what actually exists in the environment**: list the test "
    "directories, check whether the test files or scripts mentioned in the "
    "spec are present. If they are missing, the verification method must "
    "acknowledge this and describe what the agent CAN do instead (e.g., "
    "write its own tests, verify via build + code review, run available "
    "build targets). Never require the agent to run tests or scripts that "
    "do not exist in the workspace — and never encourage the agent to "
    "download them from external sources.\n\n"
    "### 3. Implementation invariants\n"
    "Specific correctness constraints that the implementation must satisfy. "
    "These are the tricky edge cases where a developer is most likely to "
    "make mistakes. Derive them from the task structure:\n"
    "- **Initialization paths**: are there functions called during boot/init "
    "with different preconditions than normal runtime?\n"
    "- **Concurrency**: are there locks that could be re-acquired through "
    "callbacks or interrupt handlers?\n"
    "- **API surface**: are there test files or specs that define the exact "
    "function signatures, names, or return types required?\n"
    "- **Build system**: are there Makefiles, CMakeLists, or configs that "
    "must be updated for new source files or targets?\n"
    "- **Edge cases**: what happens at boundaries (empty input, max size, "
    "zero refcount, null pointer)?\n\n"
    "IMPORTANT: every invariant must be traceable to an explicit statement "
    "in the source material. Do NOT invent restrictions that are not stated "
    "— e.g. do not add 'only modify these files' unless the spec explicitly "
    "says so. If the spec lists files to implement, that is guidance, not a "
    "prohibition on touching other files.\n\n"
    "### 4. Verification checklist\n"
    "Concrete checks the reviewer should perform on the code, each phrased "
    "as a yes/no question. Focus on the invariants above.\n\n"
    "When ready, call the `submit_result` tool with your structured output. "
    "Do NOT output the condition as text — you MUST call the tool."
)

_TRACE_QUERY_EXT: Final[tuple[str, dict[str, Any]]] = (
    "agentm.extensions.builtin.trace_query",
    {},
)


class _CheckerVerdictArgs(BaseModel):
    met: bool = Field(
        description=(
            "True only if the condition is fully satisfied with trajectory "
            "evidence."
        ),
    )
    reason: str = Field(
        description="Concrete justification citing the trajectory.",
    )
    unexplained: list[str] = Field(
        default_factory=list,
        description=(
            "Condition sub-requirements you could not confirm from the "
            "trajectory."
        ),
    )


async def _checker_submit_verdict(args: dict[str, Any]) -> ToolTerminate:
    parsed = _CheckerVerdictArgs.model_validate(args)
    return ToolTerminate(
        result=ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=parsed.model_dump_json(),
                )
            ]
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


class _ConditionSchema(BaseModel):
    goal: str = Field(description="The final acceptance criterion.")
    verification_method: str = Field(
        description=(
            "How to verify the goal is met. Quote specific passages from the "
            "task specification that describe testing or validation procedures, "
            "including commands to run, companion processes, and expected outputs."
        ),
    )
    invariants: list[str] = Field(
        description="Correctness constraints the implementation must satisfy. Focus on tricky edge cases: init paths, concurrency, API surface, build system."
    )
    checklist: list[str] = Field(
        description="Yes/no verification questions the reviewer should check against the code."
    )


_STRUCTURED_OUTPUT_EXT: Final[tuple[str, dict[str, Any]]] = (
    "agentm.extensions.builtin.structured_output",
    {"result_schema": _ConditionSchema.model_json_schema()},
)
_AGENT_ENV_SESSION_SERVICE: Final[str] = "agent_env.session_id"
_OPERATIONS_ATOM: Final[str] = "".join(("oper", "ations"))


class GoalConfig(BaseModel):
    condition: str | None = None
    checker_scenario: str = "local"
    checker_max_turns: int = 10
    checker_prompt: str | None = None
    auto_init: bool = False
    auto_init_scenario: str | None = None
    auto_init_max_turns: int = 10
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
        "via /goal; an independent checker agent inspects the parent "
        "session's trajectory after every turn and keeps the session "
        "running until the condition is met."
    ),
    registers=(
        "event:decide_turn_action",
        "event:before_send_to_llm",
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
    extra_tools: list[Any] | None = None,
    atom_config_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[AgentMessage] | None:
    config = AgentSessionConfig(
        cwd=api.cwd,
        scenario=scenario,
        extra_extensions=extra_extensions or [],
        extra_tools=extra_tools or [],
        atom_config_overrides=atom_config_overrides or {},
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


def _agent_env_attach_overrides(api: ExtensionAPI) -> dict[str, dict[str, Any]]:
    session_id = api.get_service(_AGENT_ENV_SESSION_SERVICE)
    if not isinstance(session_id, str) or not session_id:
        return {}
    return {_OPERATIONS_ATOM: {"attach_session": session_id}}


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
        api,
        scenario,
        max_turns,
        prompt,
        "goal_checker",
        extra_extensions=[_TRACE_QUERY_EXT],
        extra_tools=[_CHECKER_VERDICT_TOOL],
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
                if not isinstance(result, dict):
                    last_error = ValueError(
                        f"submit_result arguments did not contain a result object: {result!r}"
                    )
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
        inv_lines = "\n".join(
            f"{i + 1}. {v}" for i, v in enumerate(payload.invariants)
        )
        parts.append(f"## Invariants\n{inv_lines}")
    if payload.checklist:
        chk_lines = "\n".join(f"- {c}" for c in payload.checklist)
        parts.append(f"## Checklist\n{chk_lines}")
    return "\n\n".join(parts)


def _structured_retry_prompt(original: str, exc: Exception, attempt: int) -> str:
    return (
        f"{original}\n\n"
        f"## Structured-output retry #{attempt}\n"
        "Your previous response did not produce a valid structured result.\n"
        f"Error: {str(exc)[:2000]}\n\n"
        "You MUST call the submit_result tool exactly once with a valid "
        "`result` object conforming to the required JSON schema."
    )


def _collect_user_text(messages: list[AgentMessage]) -> str:
    user_parts: list[str] = []
    for msg in messages:
        if isinstance(msg, UserMessage):
            for block in msg.content:
                if isinstance(block, TextContent):
                    user_parts.append(block.text)
    return "\n".join(user_parts).strip()


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


class _GoalRuntime:
    def __init__(self, api: ExtensionAPI, config: GoalConfig) -> None:
        self._api = api
        self._checker_scenario = config.checker_scenario
        self._checker_max_turns = config.checker_max_turns
        self._checker_prompt_override = config.checker_prompt
        self._auto_init_enabled = config.auto_init
        self._auto_init_scenario = config.auto_init_scenario or api.scenario
        self._auto_init_max_turns = config.auto_init_max_turns
        self._auto_init_retries = max(config.auto_init_retries, 0)
        self._max_rejects = max(config.max_rejects, 1)
        self._auto_init_started = False
        self._state: _GoalState | None = None

        if config.condition:
            self._state = _GoalState(condition=config.condition)
            api.set_service(GOAL_CONDITION_SERVICE, config.condition)
            logger.info("goal: static condition — {}", config.condition)

    def install(self) -> None:
        self._register_goal_command()
        self._register_auto_init()
        self._api.on(DecideTurnActionEvent.CHANNEL, self._on_decide)

    def _register_goal_command(self) -> None:
        self._api.register_command(
            "goal",
            CommandSpec(
                description=(
                    "Set a completion condition that keeps the session running "
                    "until met."
                ),
                handler=self._goal_command,
            ),
        )

    async def _goal_command(self, args: str, cmd_api: ExtensionAPI) -> None:
        stripped = args.strip()

        if not stripped:
            cmd_api.send_user_message(self._goal_status_message())
            return

        if stripped.lower() in _CLEAR_ALIASES:
            if self._state is not None and not self._state.achieved:
                self._state = None
                cmd_api.send_user_message("Goal cleared.")
            else:
                cmd_api.send_user_message("No active goal to clear.")
            return

        self._state = _GoalState(condition=stripped)
        cmd_api.send_user_message(
            f"Goal set (checker scenario={self._checker_scenario}): {stripped}"
        )

    def _goal_status_message(self) -> str:
        state = self._state
        if state is None:
            return "No active goal."

        elapsed = time.monotonic() - state.started_at
        if state.released:
            return (
                f"Goal released after {state.consecutive_rejects} consecutive "
                f"rejects ({state.turns_evaluated} turn(s), {elapsed:.0f}s, "
                f"not achieved): {state.condition}"
            )
        if state.achieved:
            return (
                f"Goal achieved in {state.turns_evaluated} turn(s) "
                f"({elapsed:.0f}s): {state.condition}"
            )

        msg = (
            f"Active goal ({state.turns_evaluated} turn(s), "
            f"{elapsed:.0f}s): {state.condition}"
        )
        if state.last_reason:
            msg += f"\nLast evaluation: {state.last_reason}"
        return msg + f"\nChecker: scenario={self._checker_scenario}"

    def _register_auto_init(self) -> None:
        if not self._auto_init_enabled:
            return
        if self._auto_init_scenario is None:
            logger.warning(
                "goal: auto_init requires a scenario but none is set — skipping"
            )
            return
        if self._api.parent_session_id is not None:
            return
        self._api.on(BeforeSendToLlmEvent.CHANNEL, self._on_before_send)

    async def _on_before_send(self, event: BeforeSendToLlmEvent) -> None:
        if self._state is not None:
            return
        if self._auto_init_started:
            return
        user_text = _collect_user_text(event.messages)
        if not user_text:
            logger.warning(
                "goal: auto_init skipped; no user message available "
                "before first LLM request"
            )
            self._auto_init_started = True
            return

        self._auto_init_started = True
        await self._derive_goal(event.system or "", user_text)

    async def _derive_goal(self, system: str, user_text: str) -> None:
        if self._auto_init_scenario is None:
            return

        base_prompt = _AUTO_INIT_PROMPT.format(
            system=system,
            user_text=user_text,
        )
        prompt = base_prompt
        last_error: Exception | None = None
        for attempt in range(self._auto_init_retries + 1):
            messages = await _prompt_child_session_messages(
                self._api,
                self._auto_init_scenario,
                self._auto_init_max_turns,
                prompt,
                "goal_derivation",
                extra_extensions=[_STRUCTURED_OUTPUT_EXT],
                atom_config_overrides=_agent_env_attach_overrides(self._api),
            )
            if messages is None:
                last_error = RuntimeError("goal derivation child produced no response")
            else:
                try:
                    payload = _parse_structured_payload(messages)
                except (TypeError, ValueError, ValidationError) as exc:
                    last_error = exc
                else:
                    if self._state is not None:
                        return
                    condition = _format_condition(payload)
                    self._state = _GoalState(
                        condition=condition,
                        goal_summary=payload.goal,
                        verification_method=payload.verification_method,
                    )
                    self._api.set_service(GOAL_CONDITION_SERVICE, condition)
                    logger.info(
                        "goal: auto-initialized after {} attempt(s) — {}",
                        attempt + 1,
                        condition,
                    )
                    return

            if attempt < self._auto_init_retries:
                assert last_error is not None
                logger.warning(
                    "goal: auto_init structured output invalid "
                    "(attempt {}, retrying {} left): {}",
                    attempt + 1,
                    self._auto_init_retries - attempt,
                    last_error,
                )
                prompt = _structured_retry_prompt(
                    base_prompt,
                    last_error,
                    attempt + 1,
                )
                continue

            if last_error is not None:
                logger.warning(
                    "goal: auto_init failed after {} attempt(s); "
                    "leaving goal unset: {}",
                    attempt + 1,
                    last_error,
                )
            return

    async def _on_decide(self, event: DecideTurnActionEvent) -> Any:
        state = self._state
        if state is None or state.achieved or state.released:
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
            self._api,
            self._checker_scenario,
            self._checker_max_turns,
            state.condition,
            self._checker_prompt_override,
        )
        state.turns_evaluated += 1
        state.last_reason = reason

        if is_met:
            state.achieved = True
            state.consecutive_rejects = 0
            logger.info("goal: condition met — {}", reason)
            return None

        state.consecutive_rejects += 1
        if state.consecutive_rejects >= self._max_rejects:
            state.released = True
            logger.warning(
                "goal: releasing after {} consecutive rejects — {}",
                state.consecutive_rejects, reason,
            )
            return None

        logger.info(
            "goal: not met ({}/{} rejects) — {}",
            state.consecutive_rejects, self._max_rejects, reason,
        )
        if state.verification_method:
            inject_text = (
                f"[Goal not met ({state.consecutive_rejects}/{self._max_rejects})] {reason}\n\n"
                f"Verification method (from task specification): {state.verification_method}"
            )
        else:
            inject_text = (
                f"[Goal not met ({state.consecutive_rejects}/{self._max_rejects})] {reason}\n\n"
                f"Continue working toward: {state.goal_summary or state.condition}"
            )
        return Inject(messages=[text_message(inject_text)])


def install(api: ExtensionAPI, config: GoalConfig) -> None:
    _GoalRuntime(api, config).install()
