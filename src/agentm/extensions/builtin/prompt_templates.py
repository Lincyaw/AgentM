"""Builtin prompt-template expansion atom."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.core.prompt_templates import (
    PromptTemplateRecord,
    expand_prompt_template,
    load_prompt_templates,
)
from agentm.extensions import ExtensionManifest
from agentm.harness.events import ResourcesDiscoverEvent, SessionReadyEvent
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="prompt_templates",
    description="Expand /<name> args templates before the agent loop runs.",
    registers=("event:input", "event:resources_discover", "event:session_ready"),
    config_schema={
        "type": "object",
        "properties": {
            "prompt_paths": {"type": "array", "items": {"type": "string"}},
            "include_defaults": {"type": "boolean"},
        },
        "additionalProperties": True,
    },
)


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    include_defaults = bool(config.get("include_defaults", True))
    configured_prompt_paths = [str(path) for path in config.get("prompt_paths", [])]
    cache: list[PromptTemplateRecord] = []

    async def _populate(_: SessionReadyEvent) -> None:
        responses = await api.events.emit(
            "resources_discover",
            ResourcesDiscoverEvent(cwd=api.cwd, reason="startup"),
        )
        prompt_paths = list(configured_prompt_paths)
        for response in responses:
            if not isinstance(response, dict):
                continue
            extra_paths = response.get("prompt_paths")
            if not isinstance(extra_paths, list):
                continue
            prompt_paths.extend(str(path) for path in extra_paths)
        cache[:] = load_prompt_templates(
            cwd=api.cwd,
            agent_dir=str(Path.home() / ".agentm"),
            prompt_paths=tuple(prompt_paths),
            include_defaults=include_defaults,
        )

    def _on_input(event: dict[str, Any]) -> None:
        text = event.get("text")
        if not isinstance(text, str) or not text.startswith("/"):
            return
        expanded = expand_prompt_template(text, cache)
        if expanded is not None:
            event["text"] = expanded

    api.on("session_ready", _populate)
    api.on("input", _on_input)
