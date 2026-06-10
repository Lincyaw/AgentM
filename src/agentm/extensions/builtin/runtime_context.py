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

import platform
from pathlib import Path
from pydantic import BaseModel

from agentm.extensions import ExtensionManifest
from agentm.core.abi.events import BeforeAgentStartEvent
from agentm.core.abi.extension import ExtensionAPI


class RuntimeContextConfig(BaseModel):
    pass


MANIFEST = ExtensionManifest(
    name="runtime_context",
    description="Injects workspace cwd + host runtime facts into the system prompt.",
    registers=("event:before_agent_start",),
    config_schema=RuntimeContextConfig,
    requires=(),  # Leaf atom: reads only api.cwd + stdlib platform.
)


def _build_block(cwd: str) -> str:
    workspace = str(Path(cwd).expanduser().resolve())
    sysname = platform.system()
    runtime = (
        f"{'macOS' if sysname == 'Darwin' else sysname} "
        f"{platform.machine()}, Python {platform.python_version()}"
    )
    return (
        "<runtime_context>\n"
        f"workspace: {workspace}\n"
        f"runtime: {runtime}\n"
        "</runtime_context>\n"
        "\n"
        "When the user asks where you are, what you can see, or what your "
        "working directory is, answer from the workspace path above. Do not "
        "guess and do not fall back to a home directory. All shell commands "
        "run with this as cwd."
    )


def install(api: ExtensionAPI, config: RuntimeContextConfig) -> None:
    block = _build_block(api.cwd)

    def before_agent_start(event: BeforeAgentStartEvent) -> dict[str, str]:
        current = str(event.system or "")
        updated = f"{block}\n\n{current}" if current else block
        event.system = updated
        return {"system": updated}

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)
