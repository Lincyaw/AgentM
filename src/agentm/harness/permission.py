"""Permission mode enforcement for the Agent Harness.

Provides PermissionMode, PermissionPolicy, and PermissionMiddleware to control
what tools an agent is allowed to execute at runtime. Enforcement is entirely
middleware-based — SimpleAgentLoop requires zero changes.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentm.core.tool import Tool
from agentm.harness.middleware import MiddlewareBase, inject_into_system_message
from agentm.harness.types import LoopContext, Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PermissionMode enum
# ---------------------------------------------------------------------------

class PermissionMode(Enum):
    """Permission modes controlling agent tool execution rights.

    DEFAULT     — All tools available, no special constraints.
    READONLY    — Only tools marked as readonly may be executed.
    SUPERVISED  — All tools permitted; every call is audited.
    UNRESTRICTED — No permission checks; middleware short-circuits.
    """

    DEFAULT = "default"
    READONLY = "readonly"
    SUPERVISED = "supervised"
    UNRESTRICTED = "unrestricted"


# ---------------------------------------------------------------------------
# Permission constraint block injected into system prompt in READONLY mode
# ---------------------------------------------------------------------------

_READONLY_CONSTRAINT = (
    "<permission_constraint>\n"
    "You are in READONLY mode. You may only use tools that read data.\n"
    "Do NOT attempt to call any tool that modifies, creates, or deletes data.\n"
    "If you need a write operation, describe what you would do instead.\n"
    "</permission_constraint>"
)


# ---------------------------------------------------------------------------
# PermissionPolicy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PermissionPolicy:
    """Immutable permission policy for an agent.

    Determines whether a specific tool call is allowed based on the mode,
    per-tool overrides, and the tool's readonly annotation.
    """

    mode: PermissionMode = PermissionMode.DEFAULT

    # Per-tool overrides: tool_name -> allowed (True/False).
    # Takes precedence over the mode's default behavior.
    tool_overrides: dict[str, bool] = field(default_factory=dict)

    # Audit callback for SUPERVISED mode.
    # Signature: (tool_name, tool_args, result) -> None
    audit_fn: Callable[..., Awaitable[None]] | None = None

    def can_execute(self, tool_name: str, tool: Tool | None = None) -> bool:
        """Determine if a tool call is allowed.

        Resolution order:
        1. tool_overrides (explicit per-tool allow/deny)
        2. Mode-based rule:
           - DEFAULT / SUPERVISED / UNRESTRICTED -> always True
           - READONLY -> True only if tool.readonly is True
        """
        # 1. Explicit per-tool override
        if tool_name in self.tool_overrides:
            return self.tool_overrides[tool_name]

        # 2. Mode-based rule
        if self.mode == PermissionMode.READONLY:
            if tool is None:
                return False
            return tool.readonly

        # DEFAULT, SUPERVISED, UNRESTRICTED -> allow
        return True


# ---------------------------------------------------------------------------
# PermissionMiddleware
# ---------------------------------------------------------------------------

class PermissionMiddleware(MiddlewareBase):
    """Middleware that enforces permission policies on tool calls.

    Should be the first (outermost) middleware in the chain so that
    permission denial happens before any other middleware processes the call.
    """

    def __init__(
        self,
        policy: PermissionPolicy,
        tools_dict: dict[str, Tool],
    ) -> None:
        self._policy = policy
        self._tools_dict = tools_dict

    @property
    def policy(self) -> PermissionPolicy:
        return self._policy

    async def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_next: Callable[[str, dict[str, Any]], Awaitable[str]],
        ctx: LoopContext,
    ) -> str:
        mode = self._policy.mode

        # UNRESTRICTED: short-circuit immediately (no overhead)
        if mode == PermissionMode.UNRESTRICTED:
            return await call_next(tool_name, tool_args)

        # READONLY: check permission, reject if disallowed
        if mode == PermissionMode.READONLY:
            tool = self._tools_dict.get(tool_name)
            if not self._policy.can_execute(tool_name, tool):
                logger.info(
                    "PermissionMiddleware READONLY: denied tool '%s'", tool_name
                )
                return (
                    f"Permission denied: tool '{tool_name}' is not allowed "
                    f"in READONLY mode. Only read-only tools may be used."
                )

        # DEFAULT and SUPERVISED: execute the call
        result = await call_next(tool_name, tool_args)

        # SUPERVISED: audit after execution
        if mode == PermissionMode.SUPERVISED:
            if self._policy.audit_fn is not None:
                try:
                    await self._policy.audit_fn(tool_name, tool_args, result)
                except Exception:
                    logger.exception(
                        "PermissionMiddleware: audit_fn failed for tool '%s'",
                        tool_name,
                    )
            else:
                logger.info(
                    "PermissionMiddleware SUPERVISED: tool='%s' args=%s",
                    tool_name,
                    tool_args,
                )

        return result

    async def on_llm_start(
        self, messages: list[Message], ctx: LoopContext
    ) -> list[Message]:
        if self._policy.mode != PermissionMode.READONLY:
            return messages
        return inject_into_system_message(messages, _READONLY_CONSTRAINT)
