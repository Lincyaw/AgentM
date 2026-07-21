"""Single owner for provider-facing system prompt assembly."""

from __future__ import annotations

import platform
from xml.sax.saxutils import escape

from loguru import logger
from pydantic import BaseModel, ConfigDict

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    BeforeRunEvent,
    BeforeSendEvent,
    BusPriority,
)
from agentm.core.lib import expand_path, expand_path_from_cwd
from agentm.extensions import ExtensionManifest


class PromptAssemblyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str | None = None
    prompt_file: str | None = None
    discover_project_context: bool = True
    include_runtime_context: bool = False
    include_tool_index: bool = False


MANIFEST = ExtensionManifest(
    name="prompt_assembly",
    description=(
        "Assemble configured/project context, runtime facts, and the effective "
        "tool index into one provider-facing system prompt."
    ),
    registers=("event:before_run", "event:before_send"),
    config_schema=PromptAssemblyConfig,
    requires=(),
    priority=AtomInstallPriority.CONTEXT,
)

_CONTEXT_FILENAMES = ("AGENTS.md", "CLAUDE.md")


def _discover_context_files(cwd: str) -> str:
    parts: list[str] = []
    resolved = expand_path(cwd).resolve()
    chain = [resolved, *resolved.parents]
    chain.reverse()
    for directory in chain:
        for filename in _CONTEXT_FILENAMES:
            candidate = directory / filename
            if not candidate.is_file():
                continue
            try:
                parts.append(candidate.read_text(encoding="utf-8").rstrip())
            except OSError as exc:
                logger.warning(
                    "prompt_assembly: could not read context file {}: {}",
                    candidate,
                    exc,
                )
    return "\n\n".join(part for part in parts if part)


def _resolve_prompt(
    config: PromptAssemblyConfig,
    *,
    cwd: str,
    scenario_dir: str | None,
) -> str:
    if config.prompt_file:
        path = (
            expand_path_from_cwd(config.prompt_file, scenario_dir)
            if scenario_dir
            else expand_path(config.prompt_file)
        )
        return path.read_text(encoding="utf-8")
    if config.prompt is not None:
        return config.prompt
    return _discover_context_files(cwd) if config.discover_project_context else ""


def _runtime_block(cwd: str) -> str:
    workspace = str(expand_path(cwd).resolve())
    system = platform.system()
    runtime = (
        f"{'macOS' if system == 'Darwin' else system} "
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


def _join_prompt(*parts: str | None) -> str | None:
    present = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(present) if present else None


class _PromptAssemblyRuntime:
    def __init__(
        self,
        api: AtomAPI,
        config: PromptAssemblyConfig,
        prompt: str,
    ) -> None:
        self._api = api
        self._prompt = prompt
        self._runtime = (
            _runtime_block(api.ctx.cwd) if config.include_runtime_context else ""
        )
        self._include_tool_index = config.include_tool_index

    def install(self) -> None:
        if self._prompt or self._runtime:
            self._api.on(
                BeforeRunEvent.CHANNEL,
                self._before_run,
                priority=BusPriority.PRE,
            )
        if self._include_tool_index:
            self._api.on(
                BeforeSendEvent.CHANNEL,
                self._before_send,
                priority=BusPriority.POST,
            )

    def _before_run(self, event: BeforeRunEvent) -> dict[str, str | None]:
        return {"system": _join_prompt(self._prompt, self._runtime, event.system)}

    def _before_send(self, event: BeforeSendEvent) -> dict[str, str] | None:
        if not event.tools:
            return None
        lines = [
            "# Tools",
            "",
            "You have the following tools available. Use the appropriate tool "
            "when the task calls for it - prefer tools over generating answers "
            "from memory when a tool can provide authoritative, up-to-date "
            "information. You may call multiple tools in a single turn if "
            "needed.",
            "",
            "<available_tools>",
        ]
        for tool in event.tools:
            lines.extend(
                (
                    "  <tool>",
                    f"    <name>{escape(tool.name)}</name>",
                    f"    <description>{escape(tool.description)}</description>",
                    "  </tool>",
                )
            )
        lines.append("</available_tools>")
        block = "\n".join(lines)
        return {"system": _join_prompt(event.system, block) or block}


def install(api: AtomAPI, config: PromptAssemblyConfig) -> None:
    prompt = _resolve_prompt(
        config,
        cwd=api.ctx.cwd,
        scenario_dir=api.ctx.scenario_dir,
    )
    _PromptAssemblyRuntime(api, config, prompt).install()


__all__ = ("MANIFEST", "PromptAssemblyConfig", "install")
