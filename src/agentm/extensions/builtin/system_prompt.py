"""Builtin ``system_prompt`` atom per extension-as-scenario §7."""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI, SYSTEM_PROMPT_PROVIDER
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
        ":\"/path\"}``). ``prompt_file`` wins when both are set."
    ),
    registers=("event:before_agent_start",),
    # Neither key is top-level ``required``: the skip-when-unconfigured filter
    # (session_helpers.missing_required_fields) only understands a flat
    # ``required`` list and would otherwise reject a ``prompt_file``-only
    # config as "missing prompt". install() instead resolves whichever source
    # is present and contributes nothing when neither yields text.
    config_schema=SystemPromptConfig,
    requires=(),  # Leaf atom: prepends configured prompt text only.
    provides_role=(SYSTEM_PROMPT_PROVIDER,),
)

def _resolve_prompt(config: SystemPromptConfig) -> str:
    """Resolve the prompt text from config: ``prompt_file`` (read) or ``prompt``.

    A non-empty ``prompt_file`` takes precedence over inline ``prompt``. A
    missing file raises (surfaced as an install diagnostic by the session
    factory) rather than silently producing no prompt.
    """
    if config.prompt_file:
        return Path(config.prompt_file).read_text(encoding="utf-8")
    return config.prompt or ""

def install(api: ExtensionAPI, config: SystemPromptConfig) -> None:
    prompt = _resolve_prompt(config)
    # Empty config (e.g. the auto-discovery floor's {"prompt": ""}) contributes
    # nothing — register no handler rather than prepend stray separators.
    if not prompt:
        return

    def before_agent_start(event: BeforeAgentStartEvent) -> dict[str, str]:
        current = str(event.system or "")
        updated = f"{prompt}\n\n{current}" if current else prompt
        event.system = updated
        return {"system": updated}

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)
