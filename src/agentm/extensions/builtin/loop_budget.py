# code-health: ignore-file[AM025] -- atom tools validate untyped tool, config, and service payloads
"""Builtin ``loop_budget`` atom -- sets the agent-loop turn / tool budget and,
optionally, warns the model as it approaches that budget.

The loop budget is a policy, so it lives as an atom rather than a privileged
manifest field: a scenario that wants a hard ceiling lists this atom with
``config``, exactly like any other capability.

The optional ``reminder`` sub-config turns on budget-aware runway warnings.
When it is omitted the atom only sets the budget and stays silent. When
present, the atom appends a short reminder as the agent nears the
``max_turns`` / ``max_tool_calls`` cap so the model can wrap up instead of
being hard-stopped mid-thought.

Cache discipline: the reminder is a new message at the end of the send-list,
never an edit to the system prompt or to a message already in it. Touching
either would invalidate the KV prefix that everything before it shares, while
a fresh trailing message leaves that prefix byte-identical. It also keeps the
reminder a message in its own right, marked synthetic, rather than text
smuggled into somebody else's turn.
"""

from __future__ import annotations

import time as _time

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    LOOP_BUDGET_SERVICE,
    AgentMessage,
    AtomAPI,
    BeforeRunEvent,
    BeforeSendEvent,
    LoopConfig,
    MessageMeta,
    TextContent,
    ToolResultEvent,
    TurnBeginEvent,
    UserMessage,
)
from agentm.extensions import ExtensionManifest


class ReminderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warn_within: int = Field(default=5, ge=0)
    finalize_tool: str = ""


class LoopBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_turns: int | None = None
    max_tool_calls: int | None = None
    reminder: ReminderConfig | None = None


MANIFEST = ExtensionManifest(
    name="loop_budget",
    description=(
        "Sets the agent-loop turn / tool-call budget and, when a reminder is "
        "configured, warns the model as it approaches the cap."
    ),
    registers=(
        "service:loop_budget",
        "event:before_run",
        "event:turn_begin",
        "event:tool_result",
        "event:before_send",
    ),
    config_schema=LoopBudgetConfig,
    requires=(),
)


def install(api: AtomAPI, config: LoopBudgetConfig) -> None:
    loop_config = LoopConfig(
        max_turns=_positive_or_none(config.max_turns, "max_turns"),
        max_tool_calls=_positive_or_none(config.max_tool_calls, "max_tool_calls"),
    )
    api.services.register(LOOP_BUDGET_SERVICE, loop_config, scope="session")
    if config.reminder is not None:
        _TurnReminderRuntime(api, config.reminder).install()


def _positive_or_none(value: int | None, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"loop_budget: '{key}' must be a positive integer or null; got {value!r}"
        )
    return value


class _TurnReminderRuntime:
    def __init__(self, api: AtomAPI, config: ReminderConfig) -> None:
        self._api = api
        self._warn_within = config.warn_within
        self._finalize_tool = config.finalize_tool
        self._turn_index = 0
        self._tool_calls_used = 0

    def install(self) -> None:
        self._api.on(BeforeRunEvent.CHANNEL, self._on_before_run)
        self._api.on(TurnBeginEvent.CHANNEL, self._on_turn_begin)
        self._api.on(ToolResultEvent.CHANNEL, self._on_tool_result)
        self._api.on(BeforeSendEvent.CHANNEL, self._before_send)

    def _on_before_run(self, _: BeforeRunEvent) -> None:
        self._turn_index = 0
        self._tool_calls_used = 0

    def _on_turn_begin(self, event: TurnBeginEvent) -> None:
        self._turn_index = event.turn_index

    def _on_tool_result(self, _: ToolResultEvent) -> None:
        self._tool_calls_used += 1

    def _before_send(
        self, event: BeforeSendEvent
    ) -> dict[str, list[AgentMessage]] | None:
        runway = self._runway()
        if runway is None:
            return None
        turns_left, tools_left = runway
        if not _warning_triggered(turns_left, tools_left, self._warn_within):
            return None

        messages = list(event.messages)
        if _last_step(turns_left, tools_left, threshold=2) and self._finalize_tool:
            messages.append(_finalize_now_message(self._finalize_tool))
            return {"messages": messages}

        text = _format_warning(turns_left, tools_left, self._finalize_tool)
        messages.append(_reminder_message(text))
        return {"messages": messages}

    def _runway(self) -> tuple[int | None, int | None] | None:
        cfg = self._api.services.get(LOOP_BUDGET_SERVICE)
        if not isinstance(cfg, LoopConfig):
            return None
        if cfg.max_turns is None and cfg.max_tool_calls is None:
            return None
        turns_left = (
            cfg.max_turns - self._turn_index if cfg.max_turns is not None else None
        )
        tools_left = (
            cfg.max_tool_calls - self._tool_calls_used
            if cfg.max_tool_calls is not None
            else None
        )
        return turns_left, tools_left


def _warning_triggered(
    turns_left: int | None,
    tools_left: int | None,
    warn_within: int,
) -> bool:
    return (turns_left is not None and turns_left <= warn_within) or (
        tools_left is not None and tools_left <= warn_within
    )


def _last_step(
    turns_left: int | None,
    tools_left: int | None,
    *,
    threshold: int,
) -> bool:
    return (turns_left is not None and turns_left <= threshold) or (
        tools_left is not None and tools_left <= threshold
    )


def _budget_meta(kind: str) -> MessageMeta:
    return MessageMeta(synthetic=True, synthetic_kind=kind, origin="loop_budget")


def _reminder_message(text: str) -> UserMessage:
    return UserMessage(
        role="user",
        content=[TextContent(type="text", text=text)],
        timestamp=_time.time(),
        meta=_budget_meta("budget_reminder"),
    )


def _finalize_now_message(finalize_tool: str) -> UserMessage:
    return UserMessage(
        role="user",
        content=[
            TextContent(
                type="text",
                text=(
                    f"SYSTEM: Your investigation time is up. You MUST call "
                    f"`{finalize_tool}` NOW with your best findings. "
                    f"Do NOT make any more investigation calls."
                ),
            )
        ],
        timestamp=_time.time(),
        meta=_budget_meta("budget_finalize"),
    )


def _format_warning(
    turns_left: int | None,
    tools_left: int | None,
    finalize_tool: str = "",
) -> str:
    parts: list[str] = []
    if turns_left is not None:
        parts.append(f"{max(turns_left, 0)} turn(s)")
    if tools_left is not None:
        parts.append(f"{max(tools_left, 0)} tool call(s)")
    budget = " and ".join(parts)
    tool_hint = (
        f" Call `{finalize_tool}` NOW."
        if finalize_tool
        else " Submit your final response NOW."
    )
    last = (turns_left is not None and turns_left <= 1) or (
        tools_left is not None and tools_left <= 1
    )
    if last:
        return (
            f"[budget] This is effectively your LAST step ({budget} left before a "
            f"hard stop with no chance to summarize).{tool_hint}"
        )
    return (
        f"[budget] Only {budget} remaining before a hard stop. "
        f"Start wrapping up.{tool_hint}"
    )
