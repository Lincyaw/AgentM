"""Builtin ``runtime_context`` atom.

Prepends a ``# Runtime Context`` block describing workspace cwd and host
runtime facts to the assembled system prompt.  Scenarios should not have
to hand-build the workspace string, and channel/gateway layers should not
compose prompt content.
"""

from __future__ import annotations

import platform

from pydantic import BaseModel

from agentm.core.abi import AtomAPI, AtomInstallPriority, BeforeRunEvent
from agentm.core.lib import expand_path
from agentm.extensions import ExtensionManifest


class RuntimeContextConfig(BaseModel):
    pass


MANIFEST = ExtensionManifest(
    name="runtime_context",
    description="Injects workspace cwd + host runtime facts into the system prompt.",
    registers=("event:before_run",),
    config_schema=RuntimeContextConfig,
    requires=(),
    priority=AtomInstallPriority.CONTEXT,
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
        f"workspace: {workspace}\n"
        f"runtime: {runtime}\n\n"
        "This is your execution environment. When the user asks where you "
        "are, what you can see, or what your working directory is, answer "
        "from the workspace path above. All file paths you reference and "
        "shell commands you run use this workspace as the working directory."
    )


class _RuntimeContextRuntime:
    def __init__(self, api: AtomAPI) -> None:
        self._api = api
        self._block = _build_block(api.ctx.cwd)

    def install(self) -> None:
        self._api.on(BeforeRunEvent.CHANNEL, self.before_run)

    def before_run(self, event: BeforeRunEvent) -> dict[str, str] | None:
        current = str(event.system or "")
        updated = f"{self._block}\n\n{current}" if current else self._block
        return {"system": updated}


def install(api: AtomAPI, config: RuntimeContextConfig) -> None:
    del config
    _RuntimeContextRuntime(api).install()
