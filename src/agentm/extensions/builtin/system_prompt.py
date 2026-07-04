from __future__ import annotations

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI, SYSTEM_PROMPT_PROVIDER
from agentm.core.lib import expand_path, expand_path_from_cwd
from agentm.extensions import ExtensionManifest

class SystemPromptConfig(BaseModel):
    prompt: str | None = None
    prompt_file: str | None = None

MANIFEST = ExtensionManifest(
    name="system_prompt",
    description=(
        "Prepends configured prompt text to the system prompt. The text is "
        "supplied inline via ``prompt`` or read from a file via "
        "``prompt_file`` (useful when an orchestrator stages a long prompt on "
        "disk and configures the atom with ``-e system_prompt:{\"prompt_file\""
        ":\"/path\"}``). ``prompt_file`` wins when both are set. When neither "
        "is set, context files (CLAUDE.md / AGENTS.md) are loaded from the "
        "filesystem hierarchy."
    ),
    registers=("event:before_agent_start",),
    config_schema=SystemPromptConfig,
    requires=(),
    provides_role=(SYSTEM_PROMPT_PROVIDER,),
)

_CONTEXT_FILENAMES = ("AGENTS.md", "CLAUDE.md")


def _discover_context_files(cwd: str) -> str:
    """Walk from filesystem root down to *cwd*, collecting AGENTS.md / CLAUDE.md."""
    parts: list[str] = []
    resolved = expand_path(cwd).resolve()
    chain = [resolved, *resolved.parents]
    chain.reverse()
    for directory in chain:
        for filename in _CONTEXT_FILENAMES:
            candidate = directory / filename
            if candidate.is_file():
                try:
                    parts.append(candidate.read_text(encoding="utf-8").rstrip())
                except OSError as exc:
                    # The file exists but is unreadable — its context is dropped
                    # from the system prompt, so flag it rather than stay silent.
                    logger.warning(
                        "system_prompt: could not read context file {}: {}",
                        candidate,
                        exc,
                    )
    return "\n\n".join(parts)


def _resolve_prompt(
    config: SystemPromptConfig, *, cwd: str, scenario_dir: str | None
) -> str:
    if config.prompt_file:
        if scenario_dir:
            p = expand_path_from_cwd(config.prompt_file, scenario_dir)
        else:
            p = expand_path(config.prompt_file)
        return p.read_text(encoding="utf-8")
    if config.prompt is not None:
        return config.prompt
    return _discover_context_files(cwd)


def install(api: ExtensionAPI, config: SystemPromptConfig) -> None:
    prompt = _resolve_prompt(config, cwd=api.cwd, scenario_dir=api.scenario_dir)
    if not prompt:
        return

    def before_agent_start(event: BeforeAgentStartEvent) -> dict[str, str]:
        current = str(event.system or "")
        updated = f"{prompt}\n\n{current}" if current else prompt
        event.system = updated
        return {"system": updated}

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)
