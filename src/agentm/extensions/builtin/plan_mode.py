# code-health: ignore-file[AM025] -- classifies untyped tool metadata, tool args, and trigger payloads
"""Builtin ``plan_mode`` atom -- a session mode in which nothing may change a file.

The agent reads, searches and runs inspection commands as usual; anything that
would write is refused. It leaves the mode by calling ``exit_plan_mode`` with a
plan, which an approver accepts or sends back.

**The mode is derived from the trajectory, not stored.** Every mode change is
already durable: a host toggle arrives as a ``ModeChange`` trigger on a turn,
and the agent's own switches are ``enter_plan_mode`` / ``exit_plan_mode`` tool
records. Replaying, resuming or forking a trajectory therefore lands in the
mode that history implies, with no separate field to keep in step.

**The permission boundary is what enforces it.** ``PERMISSION_POLICY_ROLE`` is
tree-scoped, so a spawned child inherits this very policy object and cannot be
used as a way around the mode. Hiding the write tools from the catalog is a
convenience on top: it saves tokens, but a child assembles its own catalog and
is not covered by it.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    FILE_OP_EDIT,
    FILE_OP_METADATA_KEY,
    FILE_OP_WRITE,
    PERMISSION_POLICY_ROLE,
    AgentMessage,
    AtomAPI,
    BeforeSendEvent,
    CancelSignal,
    FunctionTool,
    JsonValue,
    PermissionDecision,
    PermissionPolicy,
    PermissionRequest,
    TextContent,
    Tool,
    ToolResult,
    Trigger,
    Turn,
    text_message,
)
from agentm.core.abi.tool import ToolMetadataProvider
from agentm.extensions import ExtensionManifest

PLAN: Final = "plan"
ACT: Final = "act"
_MODES: Final[frozenset[str]] = frozenset({PLAN, ACT})

MODE_TRIGGER_SOURCE: Final = "mode"
PLAN_APPROVER_SERVICE: Final = "plan_mode.approver"
"""Service key for the object that accepts or rejects a submitted plan."""

PLAN_MODE_STATE_SERVICE: Final = "plan_mode.state"
"""Service key for the shared mode state, so a child reads the root's mode."""

_EXTRAS_KEY: Final = "plan_mode"


# --- Mode change trigger ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModeChange:
    """A host switched the session mode between runs."""

    mode: str
    reason: str = ""
    source: str = MODE_TRIGGER_SOURCE

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"invalid session mode: {self.mode!r}")


class _ModeChangeRenderer:
    """Turn a mode switch into the message the model reads."""

    def render(self, trigger: Trigger) -> list[AgentMessage]:
        if not isinstance(trigger, ModeChange):
            return []
        if trigger.mode == PLAN:
            text = (
                "Switching to plan mode. Do not change any files from here on. "
                "Work out what the change should be, then call `exit_plan_mode` "
                "with the plan."
            )
        else:
            text = "Leaving plan mode. You can change files again."
        if trigger.reason:
            text = f"{text}\n\n{trigger.reason}"
        return [text_message(text)]


class _ModeChangeCodec:
    """Persist the trigger so a resumed trajectory folds to the same mode."""

    def serialize(self, trigger: ModeChange) -> dict[str, JsonValue]:
        return {"mode": trigger.mode, "reason": trigger.reason}

    def deserialize(self, data: Mapping[str, JsonValue]) -> Trigger:
        raw = data.get("mode")
        mode = raw if raw in _MODES else ACT
        reason = data.get("reason")
        return ModeChange(
            mode=str(mode),
            reason=reason if isinstance(reason, str) else "",
        )


# --- Approval ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlanVerdict:
    """Whether a submitted plan may proceed, and what to say if it may not."""

    approved: bool
    feedback: str = ""


@runtime_checkable
class PlanApprover(Protocol):
    """Who decides that a plan is good enough to act on.

    Deliberately narrow so the decision can be a rubber stamp, a critic child
    session, or a human, without any of the rest of this atom knowing which.
    """

    async def review(self, plan: str) -> PlanVerdict: ...


@dataclass(frozen=True, slots=True)
class AutoApprover:
    """Accepts every plan. The mode still forces one to be written first."""

    async def review(self, plan: str) -> PlanVerdict:
        return PlanVerdict(approved=True)


# --- Mode state -------------------------------------------------------------


def _mode_from_extras(extras: JsonValue) -> str | None:
    if not isinstance(extras, Mapping):
        return None
    payload = extras.get(_EXTRAS_KEY)
    if not isinstance(payload, Mapping):
        return None
    mode = payload.get("mode")
    return mode if isinstance(mode, str) and mode in _MODES else None


def _fold(turns: Sequence[Turn], initial: str) -> str:
    """Replay every recorded mode change over the starting mode."""

    mode = initial
    for turn in turns:
        if isinstance(turn.trigger, ModeChange):
            mode = turn.trigger.mode
        for record in turn.tool_results:
            if record.result.is_error:
                continue
            recorded = _mode_from_extras(record.result.extras)
            if recorded is not None:
                mode = recorded
    return mode


@dataclass(slots=True)
class _ModeState:
    """The current mode, derived from committed turns.

    ``uncommitted`` carries a switch made by a tool in the turn that is still
    running. Without it, ``exit_plan_mode`` would not take effect until its own
    turn committed, and the driver does not pull a queued trigger while a prompt
    run is still going -- so the agent would sit in plan mode for one more turn
    after being told it could act.
    """

    api: AtomAPI
    initial: str
    folded: str = ""
    counted: int = -1
    uncommitted: str | None = None

    def current(self) -> str:
        turns = self.api.get_turns()
        if len(turns) != self.counted:
            self.counted = len(turns)
            self.folded = _fold(turns, self.initial)
            self.uncommitted = None
        return self.uncommitted or self.folded

    def switch(self, mode: str) -> None:
        self.uncommitted = mode


# --- What counts as changing a file -----------------------------------------

_QUOTED_RE: Final = re.compile(r"'[^']*'|\"[^\"]*\"")
_SEGMENT_RE: Final = re.compile(r"[;&|\n]+")
_ENV_ASSIGN_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_REDIRECT_RE: Final = re.compile(r"(?<![0-9<>&])>>?\s*(?![&(])([^\s;|&<>]+)")

_INPLACE_RE: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bsed\b[^|;]*\s-[a-zA-Z]*i(?:\s|$|[.'\"])"),
    re.compile(r"\bsed\b[^|;]*--in-place"),
    re.compile(r"\bawk\b[^|;]*-i\s+inplace"),
    re.compile(r"\bperl\b[^|;]*\s-[a-zA-Z]*i"),
)

#: Redirecting to these changes nothing anyone cares about.
_HARMLESS_TARGETS: Final[frozenset[str]] = frozenset(
    {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"}
)

#: Tokens that stand in front of the command that actually runs.
_WRAPPERS: Final[frozenset[str]] = frozenset(
    {"sudo", "env", "time", "nohup", "nice", "command", "xargs", "builtin", "exec"}
)

_MUTATING_COMMANDS: Final[frozenset[str]] = frozenset(
    {
        "chmod",
        "chown",
        "cp",
        "dd",
        "install",
        "ln",
        "mkdir",
        "mv",
        "patch",
        "rm",
        "rmdir",
        "shred",
        "tee",
        "touch",
        "truncate",
        "unlink",
    }
)

_MUTATING_GIT: Final[frozenset[str]] = frozenset(
    {
        "add",
        "am",
        "apply",
        "checkout",
        "cherry-pick",
        "clean",
        "commit",
        "merge",
        "mv",
        "push",
        "rebase",
        "reset",
        "restore",
        "revert",
        "rm",
        "stash",
        "switch",
        "tag",
    }
)


def _strip_quoted(command: str) -> str:
    """Blank out quoted spans so text inside them is not read as shell syntax."""

    return _QUOTED_RE.sub(" ", command)


def _command_heads(command: str) -> list[list[str]]:
    """The token list of each pipeline segment, past wrappers and env prefixes."""

    segments: list[list[str]] = []
    for raw in _SEGMENT_RE.split(command):
        tokens = raw.split()
        while tokens and (_ENV_ASSIGN_RE.match(tokens[0]) or tokens[0] in _WRAPPERS):
            tokens = tokens[1:]
        if tokens:
            segments.append(tokens)
    return segments


def _bash_write_reason(command: str) -> str | None:
    """Why this shell command would change a file, or None if it would not."""

    for pattern in _INPLACE_RE:
        if pattern.search(command):
            return "it edits a file in place"
    stripped = _strip_quoted(command)
    for match in _REDIRECT_RE.finditer(stripped):
        target = match.group(1)
        if target not in _HARMLESS_TARGETS:
            return f"it redirects output into {target}"
    for tokens in _command_heads(stripped):
        head = tokens[0].rsplit("/", 1)[-1]
        if head in _MUTATING_COMMANDS:
            return f"`{head}` changes files"
        if head == "git":
            for token in tokens[1:]:
                if token.startswith("-"):
                    continue
                if token in _MUTATING_GIT:
                    return f"`git {token}` changes the working tree or history"
                break
    return None


@dataclass(slots=True)
class _WriteDetector:
    """Answers whether one tool call would change a file.

    File tools already say so themselves through the ``file_op`` metadata
    vocabulary, so the names are learned rather than listed. A tool that
    declares nothing is not blocked; ``deny`` is the way to cover one.
    """

    deny: tuple[str, ...]
    bash_tools: frozenset[str]
    protect_bash: bool
    writers: set[str] = field(default_factory=set)

    def learn(self, tools: Sequence[Tool]) -> None:
        for tool in tools:
            if not isinstance(tool, ToolMetadataProvider):
                continue
            if tool.metadata.get(FILE_OP_METADATA_KEY) in (FILE_OP_WRITE, FILE_OP_EDIT):
                self.writers.add(tool.name)

    def is_writer(self, name: str) -> bool:
        return name in self.writers or any(
            fnmatch.fnmatchcase(name, pattern) for pattern in self.deny
        )

    def reason(self, name: str, args: Mapping[str, object]) -> str | None:
        if self.is_writer(name):
            return f"`{name}` changes files"
        if not self.protect_bash or name not in self.bash_tools:
            return None
        command = args.get("cmd") or args.get("command")
        if not isinstance(command, str):
            return None
        return _bash_write_reason(command)


# --- Permission boundary ----------------------------------------------------

_DENY_GUIDANCE: Final = (
    "This session is in plan mode, so nothing can change a file yet, and {reason}. "
    "Nothing was modified.\n\n"
    "Keep reading and working out what the change should be. When you know, call "
    "`exit_plan_mode` with the plan; once it is accepted you can make the change "
    "yourself."
)


@dataclass(slots=True)
class _PlanModePermissionPolicy:
    """Refuses file changes while the session is in plan mode.

    Wraps whatever policy was bound before it rather than replacing it, so
    installing this atom never silently drops another boundary.
    """

    state: _ModeState
    detector: _WriteDetector
    inner: PermissionPolicy | None

    async def decide(
        self,
        request: PermissionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> PermissionDecision:
        if self.state.current() == PLAN:
            reason = self.detector.reason(request.tool_name, request.args)
            if reason is not None:
                logger.info(
                    "plan_mode: refused {} ({})",
                    request.tool_name,
                    reason,
                )
                return PermissionDecision(
                    kind="deny",
                    source="mode",
                    audience=request.audience,
                    guidance=_DENY_GUIDANCE.format(reason=reason),
                )
        if self.inner is not None:
            return await self.inner.decide(request, signal=signal)
        return PermissionDecision(kind="allow", source="mode")


# --- Tools ------------------------------------------------------------------


class _EnterPlanModeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(
        description="Why this needs planning before anything is changed.",
    )


class _ExitPlanModeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: str = Field(
        description=(
            "The change you intend to make: what you will change, where, and "
            "how you will tell whether it worked."
        ),
    )


def _result(
    text: str, *, mode: str | None = None, is_error: bool = False
) -> ToolResult:
    extras: JsonValue = {_EXTRAS_KEY: {"mode": mode}} if mode is not None else None
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        is_error=is_error,
        extras=extras,
    )


_PLAN_SYSTEM_NOTE: Final = (
    "\n\n## Plan mode\n"
    "This session is in plan mode. Read, search and run whatever inspects the "
    "project, but nothing may change a file yet; anything that would is refused.\n\n"
    "Work out what the change should be, then call `exit_plan_mode` with the "
    "plan. Once it is accepted the session leaves plan mode and you carry out "
    "the work yourself."
)


@dataclass(slots=True)
class _PlanModeRuntime:
    api: AtomAPI
    state: _ModeState
    detector: _WriteDetector
    hide_tools: bool

    def _approver(self) -> PlanApprover:
        approver = self.api.services.get(PLAN_APPROVER_SERVICE)
        if isinstance(approver, PlanApprover):
            return approver
        if approver is not None:
            logger.warning(
                "plan_mode: service {!r} does not satisfy PlanApprover; "
                "falling back to auto-approval",
                PLAN_APPROVER_SERVICE,
            )
        return AutoApprover()

    def on_before_send(
        self,
        event: BeforeSendEvent,
    ) -> dict[str, str | list[Tool]] | None:
        self.detector.learn(event.tools)
        if self.state.current() != PLAN:
            return None
        overrides: dict[str, str | list[Tool]] = {
            "system": (event.system or "") + _PLAN_SYSTEM_NOTE
        }
        if self.hide_tools:
            overrides["tools"] = [
                tool for tool in event.tools if not self.detector.is_writer(tool.name)
            ]
        return overrides

    async def enter(self, args: dict[str, object]) -> ToolResult:
        params = _EnterPlanModeParams.model_validate(args)
        if self.state.current() == PLAN:
            return _result("Already in plan mode; nothing changed.")
        self.state.switch(PLAN)
        logger.info("plan_mode: entered ({})", params.reason)
        return _result(
            "In plan mode now. Nothing you do can change a file until the plan "
            "is accepted. Call `exit_plan_mode` with the plan when you have one.",
            mode=PLAN,
        )

    async def exit(self, args: dict[str, object]) -> ToolResult:
        params = _ExitPlanModeParams.model_validate(args)
        if self.state.current() != PLAN:
            return _result(
                "This session is not in plan mode, so there is nothing to leave. "
                "Carry on with the work.",
                is_error=True,
            )
        verdict = await self._approver().review(params.plan)
        if not verdict.approved:
            logger.info("plan_mode: plan sent back")
            return _result(
                "The plan was not accepted, so the session stays in plan mode "
                "and nothing has been changed.\n\n"
                f"{verdict.feedback}\n\n"
                "Work the point through and call `exit_plan_mode` again with "
                "the revised plan."
            )
        logger.info("plan_mode: plan accepted, leaving plan mode")
        message = (
            "The plan was accepted. You can change files again. Carry it out "
            "yourself now, starting from the plan you just submitted."
        )
        if verdict.feedback:
            message = f"{message}\n\n{verdict.feedback}"
        return _result(message, mode=ACT)


# --- Atom -------------------------------------------------------------------


class PlanModeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_in_plan: bool = False
    """Open the session in plan mode, so a plan is written before any change."""

    hide_tools: bool = True
    """Also drop the file-changing tools from the catalog while planning."""

    protect_bash: bool = True
    """Refuse shell commands that would write, redirect, or edit in place."""

    bash_tools: list[str] = Field(default_factory=lambda: ["bash"])
    deny: list[str] = Field(default_factory=list)
    """Extra tool-name patterns to treat as file-changing (fnmatch)."""


MANIFEST = ExtensionManifest(
    name="plan_mode",
    description=(
        "A session mode in which nothing may change a file, left by submitting "
        "a plan for approval."
    ),
    registers=(
        "event:before_send",
        "tool:enter_plan_mode",
        "tool:exit_plan_mode",
        "permission:plan_mode",
        "service:plan_mode.approver",
        "trigger_renderer:mode",
        "trigger_codec:mode",
    ),
    config_schema=PlanModeConfig,
    requires=(),
)


def install(api: AtomAPI, config: PlanModeConfig) -> None:
    inherited = api.services.get(PLAN_MODE_STATE_SERVICE)
    if isinstance(inherited, _ModeState):
        # A child session inherited the root's state; the mode is the tree's,
        # not this session's, and its own trajectory has no mode records.
        state = inherited
    else:
        state = _ModeState(
            api=api,
            initial=PLAN if config.start_in_plan else ACT,
        )
        api.services.register(PLAN_MODE_STATE_SERVICE, state, scope="tree")

    detector = _WriteDetector(
        deny=tuple(config.deny),
        bash_tools=frozenset(config.bash_tools),
        protect_bash=config.protect_bash,
    )
    runtime = _PlanModeRuntime(
        api=api,
        state=state,
        detector=detector,
        hide_tools=config.hide_tools,
    )

    api.services.bind(
        PERMISSION_POLICY_ROLE,
        _PlanModePermissionPolicy(
            state=state,
            detector=detector,
            inner=api.services.get_role(PERMISSION_POLICY_ROLE),
        ),
        replace=True,
    )
    if api.services.get(PLAN_APPROVER_SERVICE) is None:
        api.services.register(PLAN_APPROVER_SERVICE, AutoApprover(), scope="tree")

    api.register_trigger_renderer(MODE_TRIGGER_SOURCE, _ModeChangeRenderer())
    api.register_trigger_codec(MODE_TRIGGER_SOURCE, _ModeChangeCodec())
    api.on(BeforeSendEvent.CHANNEL, runtime.on_before_send)
    api.register_tool(
        FunctionTool(
            name="enter_plan_mode",
            description=(
                "Enter plan mode: stop changing files and work out what the "
                "change should be first. Use when the work is large enough that "
                "acting straight away would be guessing."
            ),
            parameters=_EnterPlanModeParams,
            fn=runtime.enter,
        )
    )
    api.register_tool(
        FunctionTool(
            name="exit_plan_mode",
            description=(
                "Submit the plan and ask to leave plan mode. Call this once the "
                "plan is settled; if it is accepted you carry it out yourself."
            ),
            parameters=_ExitPlanModeParams,
            fn=runtime.exit,
        )
    )


__all__ = (
    "ACT",
    "MANIFEST",
    "PLAN",
    "PLAN_APPROVER_SERVICE",
    "PLAN_MODE_STATE_SERVICE",
    "AutoApprover",
    "ModeChange",
    "PlanApprover",
    "PlanModeConfig",
    "PlanVerdict",
    "install",
)
