"""Builtin ``runtime_context`` atom.

Prepends a small ``<runtime_context>`` block describing workspace cwd
and host runtime facts to the assembled system prompt. See
``.claude/designs/runtime-context-atom.md`` for rationale: scenarios
should not have to hand-build the workspace string, and channel/gateway
layers should not compose prompt content. This atom resolves runtime
facts dynamically at session start so any scenario opting in gets
ground-truth cwd / platform without scenario-specific code.
"""

from __future__ import annotations
from typing import Any

import platform

from pydantic import BaseModel

from agentm.core.abi import BeforeRunEvent
from agentm.core.lib import expand_path
from agentm.extensions import ExtensionManifest


class RuntimeContextConfig(BaseModel):
    pass


MANIFEST = ExtensionManifest(
    name="runtime_context",
    description="Injects workspace cwd + host runtime facts into the system prompt.",
    registers=("event:before_agent_start",),
    config_schema=RuntimeContextConfig,
    requires=(),  # Leaf atom: reads only session.ctx.cwd + stdlib platform.
    api_version=1,
    tier=1,
)


def _build_block(cwd: str) -> str:
    workspace = str(expand_path(cwd).resolve())
    sysname = platform.system()
    runtime = (
        f"{'macOS' if sysname == 'Darwin' else sysname} "
        f"{platform.machine()}, Python {platform.python_version()}"
    )
    return (
        "# Runtime Context\n\n"
        "<runtime_context>\n"
        f"workspace: {workspace}\n"
        f"runtime: {runtime}\n"
        "</runtime_context>\n\n"
        "This is your execution environment. When the user asks where you "
        "are, what you can see, or what your working directory is, answer "
        "from the workspace path above — do not guess or fall back to a "
        "home directory. All file paths you reference and shell commands "
        "you run use this workspace as the working directory."
    )


class _RuntimeContextRuntime:
    def __init__(self, session: Any) -> None:
        self._session = session
        self._block = _build_block(session.ctx.cwd)

    def install(self) -> None:
        self._session.bus.on(BeforeRunEvent.CHANNEL, self.before_agent_start)

    def before_agent_start(self, event: BeforeRunEvent) -> dict[str, str] | None:
        current = str(event.system or "")
        updated = f"{self._block}\n\n{current}" if current else self._block
        return {"system": updated}


def install(session: Any, config: RuntimeContextConfig) -> None:
    del config
    _RuntimeContextRuntime(session).install()


__all__ = (
    "MANIFEST",
    "RuntimeContextConfig",
    "install",
)
