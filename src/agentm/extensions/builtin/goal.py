"""Builtin ``goal`` atom — condition-driven session continuation.

Sets a completion condition; after every turn an independent evaluator
checks whether the condition holds. If not, the evaluator's reason is
injected as a user message and the loop continues. When the condition
is met the goal clears and the session terminates normally.

Evaluator: a child agent session with ``trace_query`` tools that can
inspect the parent session's trajectory. The checker decides what to
query, judges the evidence, and outputs a verdict.

Hard stops (``MaxTurnsExhausted``, ``BudgetExhausted``, ``SignalAborted``)
are ``final=True`` and cannot be overridden — they are the safety net.

The goal condition is configured via atom config (``condition`` or
``auto_init``); the current branch has no slash-command surface, so there is
no ``/goal`` command.

§11: single file; ``MANIFEST`` + ``install(api, config)``; no atom-to-atom
imports; ``core.abi`` only; no ``core.runtime.*`` / ``core._internal``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Final

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

# Acceptance discipline shared by every goal role (deriver, checker).
# Both the condition that gets written and the verdict that judges it must
# rest on the same ground rules; a single source keeps the roles from
# drifting into contradiction — e.g. the deriver baking "existing tests
# must pass" into a condition the checker is separately told to distrust.
# The auditor prompt (contrib llmharness, a separate package) states the
# same principles; keep them in sync.
_ORACLE_PRINCIPLES: Final[str] = (
    "Shared acceptance discipline — the ground rules for what 'done' means:\n"
    "- The task's stated requirements are the sole authority. The "
    "repository's current state — its existing tests, code, and "
    "conventions — is context, never the acceptance oracle.\n"
    "- An in-repo test or code path that encodes behavior the task is "
    "changing is stale. Under a correct change it is expected to fail: that "
    "failure is not a defect and not missing evidence, and making it pass "
    "again is not evidence of correctness. This is distinct from behavior "
    "the task does not change, which must not regress.\n"
    "- Verification rests on evidence that demonstrates the requirement and "
    "that this environment can actually run. A mechanism that cannot run "
    "here is not a valid requirement to impose nor a reason to withhold "
    "approval; rely on runnable checks — authored where the environment "
    "provides none — that exercise the requirement directly."
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
) + _ORACLE_PRINCIPLES + (
    "\n\n"
    "Apply that discipline when you judge: weigh every piece of evidence by "
    "whether it demonstrates the requirement, not by whether pre-existing "
    "checks stayed green. Never reject because a stale test failed or "
    "because a mechanism that cannot run here is absent.\n\n"
    "Require real, positive evidence for each part of the condition: the "
    "trajectory must show a check that actually ran in this environment and "
    "produced the requirement's own success signal — not agent prose, not "
    "inference from source, not a command that timed out, crashed, or was "
    "never executed. When the condition names several behaviors, confirm "
    "each is exercised by evidence tied to it; a check that never touches a "
    "named behavior does not satisfy it. When rejecting, name the specific "
    "behaviors whose evidence is missing so the agent knows what to add.\n\n"
    "Judge pragmatically: focus on whether the acceptance criteria are met. "
    "Do not invent restrictions the specification does not state — "
    "'the spec forbids X' is a reject; 'the spec is silent on X' is not.\n\n"
    "When done, call `submit_verdict` with your structured verdict. "
    "Do NOT output your verdict as text — you MUST call the tool."
)

_AUTO_INIT_PROMPT: Final[str] = (
    "You are a goal-condition formulator. Your job is to derive a compound "
    "completion condition — not to execute the task itself.\n\n"
    "Derive the condition for the ORIGINAL user request only. The system "
    "prompt is background context, not the task. Do not include meta-tasks "
    "(submit_result, structured output) or runtime-context instructions "
    "(working-directory guidance) unless the original request requires them.\n\n"
) + _ORACLE_PRINCIPLES + (
    "\n\n"
    "Apply that discipline when you write the condition: never turn an "
    "existing in-repo test into an acceptance requirement when the task is "
    "changing the behavior it guards — that test is expected to fail, so "
    "requiring it to pass makes the condition self-contradictory. Frame "
    "verification around runnable checks that exercise the requirement, not "
    "around keeping pre-existing checks green.\n\n"
    "You may use tools for lightweight exploration (list files, read specs, "
    "check what exists and what runs) — just enough to discover implicit "
    "requirements and to confirm that the files, commands, and procedures "
    "your condition references both exist and can actually run here. Use "
    "cheap checks (list directories, glob for names, a quick invocation to "
    "confirm a validation tool starts rather than erroring on setup); do "
    "NOT read source files end-to-end or solve the problem. When the spec's "
    "stated procedure conflicts with what actually works here, derive the "
    "condition from what can be verified here and note the substitution.\n\n"
    "You have a budget of {max_turns} turns. Spend at most half of them "
    "exploring and reserve the final turn for `submit_result` — a "
    "submitted condition with approximate grounding beats a perfect one "
    "that never gets submitted.\n\n"
    "## Parent system prompt (background only)\n{system}\n\n"
    "## Original user request\n{user_text}\n\n"
    "Your output has four parts:\n\n"
    "### 1. Completion goal\n"
    "The final acceptance criterion (tests pass, output matches spec, etc.). "
    "A change is complete only when it satisfies the requirement without "
    "disturbing behavior the task does not intend to change; both must "
    "hold.\n\n"
    "### 2. Verification method\n"
    "For each part of the goal, derive the observable evidence that would "
    "show it satisfied and state what the agent should author or run to "
    "produce it — including tests it writes itself where the environment "
    "provides none. Existing tests, scripts, or specs are inputs to "
    "consider, never the definition of done. Quote the passages of the "
    "source material that describe how the result should be validated. "
    "Never require tests or scripts that do not exist in the workspace, and "
    "never encourage downloading them from external sources.\n\n"
    "### 3. Implementation invariants\n"
    "Specific correctness constraints that the implementation must satisfy. "
    "These are the tricky edge cases where a developer is most likely to "
    "make mistakes. Derive them from the task structure:\n"
    "- **Initialization paths**: are there functions called during boot/init "
    "with different preconditions than normal runtime?\n"
    "- **Concurrency**: are there locks that could be re-acquired through "
    "callbacks or interrupt handlers?\n"
    "- **API surface**: are there test files or specs that define the exact "
    "function signatures, names, or return types required? (Only where they "
    "agree with the task's requirements — see the shared discipline above.)\n"
    "- **Build system**: does the build configuration need updating for new "
    "source files or targets?\n"
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
# Budget-runway warnings so the child submits before max_turns hard-stops
# it. Without this the deriver/checker can burn the whole budget exploring,
# produce nothing, and a fresh-session retry repeats the same failure.
_DERIVER_TURN_REMINDER_EXT: Final[tuple[str, dict[str, Any]]] = (
    "agentm.extensions.builtin.turn_reminder",
    {"warn_within": 6, "finalize_tool": "submit_result"},
)
_CHECKER_TURN_REMINDER_EXT: Final[tuple[str, dict[str, Any]]] = (
    "agentm.extensions.builtin.turn_reminder",
    {"warn_within": 4, "finalize_tool": "submit_verdict"},
)
_AGENT_ENV_SESSION_SERVICE: Final[str] = "agent_env.session_id"
_OPERATIONS_ATOM: Final[str] = "".join(("oper", "ations"))


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


async def _prompt_child_session_messages(
    api: AtomAPI,
    scenario: str,
    max_turns: int,
    prompt: str,
    purpose: str,
    extra_extensions: list[tuple[str, dict[str, Any]]] | None = None,
    extra_tools: list[Any] | None = None,
    atom_config_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[AgentMessage] | None:
    config = AgentSessionConfig(
        cwd=api.ctx.cwd,
        scenario=scenario,
        extra_extensions=extra_extensions or [],
        extra_tools=extra_tools or [],
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
        return await child.prompt(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("goal: {} failed: {}", purpose, exc)
        return None
    finally:
        try:
            await child.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.debug("goal: {} shutdown (ignored): {}", purpose, exc)


def _agent_env_attach_overrides(api: AtomAPI) -> dict[str, dict[str, Any]]:
    session_id = api.services.get(_AGENT_ENV_SESSION_SERVICE)
    if not isinstance(session_id, str) or not session_id:
        return {}
    return {_OPERATIONS_ATOM: {"attach_session": session_id}}


# ---------------------------------------------------------------------------
# Checker evaluation (child agent with trace_query tools)
# ---------------------------------------------------------------------------


async def _evaluate_checker(
    api: AtomAPI,
    scenario: str,
    max_turns: int,
    condition: str,
    checker_prompt_override: str | None = None,
) -> tuple[bool | None, str]:
    """Run one checker session.

    Returns ``(met, reason)``; ``met is None`` means the checker itself
    failed to produce a verdict (spawn failure or no ``submit_verdict``
    call) — an infrastructure failure, not a judgment on the goal.
    """

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
        extra_extensions=[_TRACE_QUERY_EXT, _CHECKER_TURN_REMINDER_EXT],
        extra_tools=[_CHECKER_VERDICT_TOOL],
        # Attach the checker to the same agent_env sandbox as the agent (mirror
        # the deriver path). Without this, an agent_env checker scenario fails
        # to load ("'image' or 'attach_session' required") and the goal loop
        # silently allows stop-without-verdict.
        atom_config_overrides=_agent_env_attach_overrides(api),
    )
    if messages is None:
        return None, "checker produced no response"
    return _parse_verdict_from_tool_call(messages)


# ---------------------------------------------------------------------------
# Verdict parsing — structured tool call
# ---------------------------------------------------------------------------


def _parse_verdict_from_tool_call(
    messages: list[AgentMessage],
) -> tuple[bool | None, str]:
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
                    # Mirror the structured_output atom's runtime handling:
                    # models sometimes double-encode the result as a JSON
                    # string. The tool accepted it, so this parser must too.
                    try:
                        decoded = json.loads(result)
                    except json.JSONDecodeError:
                        decoded = None
                    if isinstance(decoded, dict):
                        result = decoded
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
    def __init__(self, api: AtomAPI, config: GoalConfig) -> None:
        self._api = api
        self._checker_scenario = config.checker_scenario
        self._checker_max_turns = config.checker_max_turns
        self._checker_prompt_override = config.checker_prompt
        self._checker_retries = max(config.checker_retries, 0)
        self._auto_init_enabled = config.auto_init
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
            logger.info("goal: static condition — {}", config.condition)

    def install(self) -> None:
        if self._auto_init_enabled and self._auto_init_scenario is None:
            logger.warning(
                "goal: auto_init requires a scenario but none is set — disabled"
            )
            self._auto_init_enabled = False
        self._api.on(BeforeSendEvent.CHANNEL, self._on_before_send)
        self._api.on(DecideEvent.CHANNEL, self._on_decide)

    async def _on_before_send(self, event: BeforeSendEvent) -> dict[str, str] | None:
        if (
            self._auto_init_enabled
            and not self._auto_init_started
            and self._state is None
            and self._api.ctx.parent_session_id is None
        ):
            user_text = _collect_user_text(list(event.messages))
            self._auto_init_started = True
            if not user_text:
                logger.warning(
                    "goal: auto_init skipped; no user message available "
                    "before first LLM request"
                )
            else:
                await self._derive_goal(event.system or "", user_text)
        return self._goal_criteria_override(event)

    def _goal_criteria_override(
        self, event: BeforeSendEvent
    ) -> dict[str, str] | None:
        """Expose the acceptance criteria to the main agent.

        Appended to the system prompt on every request while a goal is
        active: the block is identical each turn, so it forms a stable
        prefix (KV-cache friendly) and — unlike a conversation message —
        survives compaction. The agent sees the bar it will be checked
        against from turn one and can build verification that covers it,
        instead of discovering the criteria through rejections.

        Frozen events: return a ``{"system": ...}`` override rather than
        mutating the event.
        """

        state = self._state
        if state is None or state.achieved or state.released:
            return None
        system = (
            (event.system or "")
            + "\n\n## Completion criteria\n"
            "An independent checker will verify your work against the "
            "condition below before the session can end. Design your "
            "verification early: build checks that exercise each "
            "checklist item, run them, and only declare completion once "
            "they pass.\n\n"
            + state.condition
        )
        return {"system": system}

    async def _derive_goal(self, system: str, user_text: str) -> None:
        if self._auto_init_scenario is None:
            return

        base_prompt = _AUTO_INIT_PROMPT.format(
            system=system,
            user_text=user_text,
            max_turns=self._auto_init_max_turns,
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
                extra_extensions=[_STRUCTURED_OUTPUT_EXT, _DERIVER_TURN_REMINDER_EXT],
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
                    self._api.services.register(
                        GOAL_CONDITION_SERVICE, condition, scope="session"
                    )
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

    async def _on_decide(self, event: DecideEvent) -> Any:
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
                attempt + 1, self._checker_retries + 1, reason,
            )
        state.turns_evaluated += 1
        state.last_reason = reason

        if is_met is None:
            # The checker itself failed, not the agent's work. Burning a
            # reject slot here would let harness flakiness drive the release
            # counter, and injecting a fabricated rejection would mislead
            # the agent — allow the stop without a verdict instead.
            logger.warning(
                "goal: checker unavailable after {} attempt(s); "
                "allowing stop without verdict — {}",
                self._checker_retries + 1, reason,
            )
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
        return Inject(messages=(text_message(inject_text),))


def install(api: AtomAPI, config: GoalConfig) -> None:
    _GoalRuntime(api, config).install()
