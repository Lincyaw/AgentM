"""Builtin ``phased_budget`` atom — per-phase tool-type restrictions.

Gives the agent a ``next_phase`` tool to signal readiness to move on.
In each phase, only certain tool categories are allowed; others are
blocked with a message explaining what to do instead.

Phases:
  1. **explore** — read/bash/grep allowed, edit/write blocked.
     Agent calls ``next_phase`` when done reading.
  2. **fix** — edit/write/bash allowed, read limited to files already
     seen in explore phase.
  3. **verify** — bash/read/edit all allowed (run tests, iterate).

```yaml
extensions:
  - module: agentm.extensions.builtin.phased_budget
    config:
      explore_budget: 12   # max read-class calls in explore phase
```
"""

from __future__ import annotations

from pydantic import BaseModel

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolCallEvent,
    ToolResult,
)
from agentm.core.lib import pydantic_to_tool_schema
from agentm.extensions import ExtensionManifest
from loguru import logger


class PhasedBudgetConfig(BaseModel):
    explore_budget: int = 12


MANIFEST = ExtensionManifest(
    name="phased_budget",
    description="Per-phase tool restrictions with explicit phase transitions.",
    registers=("tool:next_phase",),
    config_schema=PhasedBudgetConfig,
    requires=(),
    api_version=1,
    tier=1,
)

_READ_TOOLS = frozenset({"read", "glob", "grep"})
_WRITE_TOOLS = frozenset({"edit", "write"})


class _NextPhaseParams(BaseModel):
    summary: str
    edit_plan: str


class _PhasedBudgetRuntime:
    __slots__ = ("_api", "_phase", "_explore_budget", "_explore_count", "_seen_files")

    def __init__(self, api: ExtensionAPI, config: PhasedBudgetConfig) -> None:
        self._api = api
        self._phase = "explore"
        self._explore_budget = config.explore_budget
        self._explore_count = 0
        self._seen_files: set[str] = set()

    def install(self) -> None:
        self._api.on(ToolCallEvent.CHANNEL, self._on_tool_call)
        self._api.register_tool(
            FunctionTool(
                name="next_phase",
                description=(
                    "Signal that you are done exploring and ready to write "
                    "fixes. You MUST provide:\n"
                    "- summary: what you found (the bug root cause)\n"
                    "- edit_plan: EXACT description of your edit — which file, "
                    "what old code to replace, what new code to write.\n"
                    "After calling this, you must immediately use the `edit` "
                    "tool to execute your edit_plan."
                ),
                parameters=pydantic_to_tool_schema(_NextPhaseParams),
                fn=self._next_phase,
            )
        )

    async def _next_phase(self, args: dict) -> ToolResult:
        summary = args.get("summary", "")
        edit_plan = args.get("edit_plan", "")
        if self._phase == "explore":
            if not edit_plan.strip():
                return ToolResult(
                    content=[TextContent(
                        type="text",
                        text=(
                            "Cannot transition: edit_plan is required. "
                            "Describe exactly which file to edit, what old code "
                            "to replace, and what new code to write."
                        ),
                    )],
                    is_error=True,
                )
            plan_lower = edit_plan.lower()
            has_file = "/" in edit_plan
            has_action = any(
                kw in plan_lower
                for kw in ("replace", "change", "add", "remove", "insert",
                           "delete", "modify", "old_string", "new_string")
            )
            if not (has_file and has_action):
                return ToolResult(
                    content=[TextContent(
                        type="text",
                        text=(
                            "edit_plan rejected: must contain a file PATH "
                            "(with /) and a concrete ACTION (replace/change/"
                            "add/remove/insert). Example: 'In /repo/src/foo.ts "
                            "replace `if (x)` with `if (x && y)`.'"
                        ),
                    )],
                    is_error=True,
                )
            self._phase = "fix"
            logger.info("phased_budget: → fix phase ({})", summary)
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=(
                        f"Phase changed to FIX.\n\n"
                        f"Root cause: {summary}\n\n"
                        f"Your edit plan: {edit_plan}\n\n"
                        f"NOW EXECUTE YOUR PLAN: use the `edit` tool with "
                        f"old_string (the code to replace) and new_string "
                        f"(the fixed code). Do it immediately."
                    ),
                )],
                is_error=False,
            )
        elif self._phase == "fix":
            self._phase = "verify"
            logger.info("phased_budget: → verify phase")
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text="Phase changed to VERIFY. Run tests to confirm your fix works.",
                )],
                is_error=False,
            )
        return ToolResult(
            content=[TextContent(type="text", text=f"Already in {self._phase} phase.")],
            is_error=False,
        )

    async def _on_tool_call(self, event: ToolCallEvent) -> dict | None:
        name = event.tool_name

        if name in ("next_phase", "load_skill", "task_create", "task_update",
                    "task_list", "task_get", "finish"):
            return None

        if self._phase == "explore":
            if name in _WRITE_TOOLS:
                return {
                    "block": True,
                    "reason": (
                        f"[phased_budget] You are in EXPLORE phase. "
                        f"Cannot use '{name}' yet. "
                        f"Call next_phase first when you are ready to write fixes. "
                        f"(explore budget: {self._explore_count}/{self._explore_budget})"
                    ),
                }

            if name in _READ_TOOLS:
                self._explore_count += 1
                path = event.args.get("path", "")
                if path:
                    self._seen_files.add(path)

                if self._explore_count >= self._explore_budget:
                    return {
                        "block": True,
                        "reason": (
                            f"[phased_budget] Explore budget exhausted "
                            f"({self._explore_count}/{self._explore_budget} reads used). "
                            f"You MUST call next_phase now and start fixing. "
                            f"Files you read: {sorted(self._seen_files)}"
                        ),
                    }

        elif self._phase == "fix":
            if name == "bash":
                cmd = event.args.get("cmd", "")
                is_read_cmd = any(
                    kw in cmd for kw in ("cat ", "head ", "tail ", "sed -n",
                                         "find ", "grep ", "ls ", "less ")
                )
                if is_read_cmd:
                    return {
                        "block": True,
                        "reason": (
                            "[phased_budget] BLOCKED. You are in FIX phase. "
                            "Stop searching. Execute your edit_plan NOW using "
                            "the `edit` tool. Call edit(path=<file>, "
                            "old_string=<code to replace>, "
                            "new_string=<fixed code>)."
                        ),
                    }
            elif name in _READ_TOOLS and name != "bash":
                path = event.args.get("path", "")
                if path and path not in self._seen_files:
                    return {
                        "block": True,
                        "reason": (
                            "[phased_budget] BLOCKED. You are in FIX phase. "
                            "Do not read new files. Execute your edit_plan NOW "
                            "using the `edit` tool."
                        ),
                    }

        return None


def install(api: ExtensionAPI, config: PhasedBudgetConfig) -> None:
    _PhasedBudgetRuntime(api, config).install()


__all__ = ("MANIFEST", "PhasedBudgetConfig", "install")
