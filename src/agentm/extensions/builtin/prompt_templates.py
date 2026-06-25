"""Builtin prompt-template registry + slash-command expansion atom.

Owns the on-disk markdown template loader, the ``/<name> args`` substituter,
and the in-memory named-prompt registry used by the compaction subsystem.
Constructs a :class:`_PromptRegistry` instance at install time and registers
it under the ``"prompt_templates"`` service key so other atoms reach it via
``api.get_service("prompt_templates")``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import (
    ExtensionAPI,
    PROMPT_REGISTRY,
    PromptRegistry,
    PromptTemplateRecord,
    ResourcesDiscoverEvent,
    SessionReadyEvent,
)
from agentm.core.lib import parse_frontmatter
from agentm.extensions import ExtensionManifest

# --- Module-level helpers (private) ----------------------------------------

def _parse_command_args(args_string: str) -> list[str]:
    args: list[str] = []
    current: list[str] = []
    in_quote: str | None = None

    for char in args_string:
        if in_quote is not None:
            if char == in_quote:
                in_quote = None
            else:
                current.append(char)
            continue
        if char in {'"', "'"}:
            in_quote = char
            continue
        if char in {" ", "\t"}:
            if current:
                args.append("".join(current))
                current = []
            continue
        current.append(char)

    if current:
        args.append("".join(current))
    return args

def _substitute_args(body: str, args: list[str]) -> str:
    result = body

    result = re.sub(
        r"\$(\d+)",
        lambda match: args[int(match.group(1)) - 1]
        if 0 < int(match.group(1)) <= len(args)
        else "",
        result,
    )

    def replace_slice(match: re.Match[str]) -> str:
        start = max(int(match.group(1)) - 1, 0)
        length_group = match.group(2)
        if length_group is None:
            return " ".join(args[start:])
        length = int(length_group)
        return " ".join(args[start : start + length])

    result = re.sub(r"\$\{@:(\d+)(?::(\d+))?\}", replace_slice, result)
    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args)
    result = result.replace("$@", all_args)
    return result

def _normalize_path(raw_path: str, cwd: str) -> str:
    expanded = raw_path.strip()
    if expanded == "~":
        return str(Path.home())
    if expanded.startswith("~/"):
        return str(Path.home() / expanded[2:])
    if expanded.startswith("~"):
        return str(Path.home() / expanded[1:])
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(cwd, expanded))

def _fallback_description(body: str) -> str:
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        return f"{stripped[:60]}..." if len(stripped) > 60 else stripped
    return ""

def _load_template_file(file_path: str, source: str) -> PromptTemplateRecord | None:
    try:
        raw = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return None

    metadata, body = parse_frontmatter(raw)
    if raw.startswith("---") and not metadata and body == raw:
        return None
    name = Path(file_path).stem
    description_value = metadata.get("description")
    description = (
        description_value
        if isinstance(description_value, str) and description_value
        else _fallback_description(body)
    )
    argument_hint_value = metadata.get("argument-hint")
    argument_hint = (
        argument_hint_value if isinstance(argument_hint_value, str) else None
    )
    return PromptTemplateRecord(
        name=name,
        description=description,
        argument_hint=argument_hint,
        body=body,
        file_path=os.path.abspath(file_path),
        source=source,
    )

def _load_templates_from_dir(
    directory: str,
    source: str,
) -> list[PromptTemplateRecord]:
    if not os.path.isdir(directory):
        return []
    templates: list[PromptTemplateRecord] = []
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return templates
    for entry in entries:
        if not entry.endswith(".md"):
            continue
        path = os.path.join(directory, entry)
        if not os.path.isfile(path):
            continue
        template = _load_template_file(path, source)
        if template is not None:
            templates.append(template)
    return templates

# --- Registry --------------------------------------------------------------

class _PromptRegistry:
    """In-memory implementation of :class:`PromptRegistry`.

    Holds the named-prompt dict and bridges to filesystem template loading.
    One instance per session, owned by this atom and published via
    ``api.set_service("prompt_templates", registry)``.
    """

    def __init__(self, project_prompt_dirs: tuple[str, ...] | None = None) -> None:
        self._project_dirs = project_prompt_dirs
        self._registry: dict[str, str] = {}

    def load_prompt_templates(
        self,
        *,
        cwd: str,
        agent_dir: str,
        prompt_paths: list[str] | tuple[str, ...] = (),
        include_defaults: bool = True,
    ) -> list[PromptTemplateRecord]:
        templates: list[PromptTemplateRecord] = []

        if include_defaults:
            templates.extend(
                _load_templates_from_dir(os.path.join(agent_dir, "prompts"), "user")
            )
            project_dirs: tuple[str, ...] = self._project_dirs or ()
            for project_dir in project_dirs:
                templates.extend(
                    _load_templates_from_dir(project_dir, "project")
                )

        for raw_path in prompt_paths:
            resolved_path = _normalize_path(raw_path, cwd)
            if os.path.isdir(resolved_path):
                templates.extend(_load_templates_from_dir(resolved_path, "path"))
                continue
            if os.path.isfile(resolved_path) and resolved_path.endswith(".md"):
                template = _load_template_file(resolved_path, "path")
                if template is not None:
                    templates.append(template)

        return templates

    def expand_prompt_template(
        self,
        text: str,
        templates: list[PromptTemplateRecord],
    ) -> str | None:
        if not text.startswith("/"):
            return None
        command, _, raw_args = text[1:].partition(" ")
        if not command:
            return None
        for template in templates:
            if template.name != command:
                continue
            return _substitute_args(template.body, _parse_command_args(raw_args.strip()))
        return None

    def register_prompt(self, name: str, body: str) -> None:
        self._registry[name] = body

    def get_prompt(self, name: str) -> str | None:
        return self._registry.get(name)

# --- Manifest + install ----------------------------------------------------

class PromptTemplatesConfig(BaseModel):
    prompt_paths: list[str] = []
    include_defaults: bool = True

MANIFEST = ExtensionManifest(
    name="prompt_templates",
    description="Expand /<name> args templates and host the named-prompt registry.",
    registers=("event:input", "event:resources_discover", "event:session_ready"),
    config_schema=PromptTemplatesConfig,
    requires=(),  # Leaf atom: loads prompt templates from resources.
    provides_role=(PROMPT_REGISTRY,),
)

async def install(api: ExtensionAPI, config: PromptTemplatesConfig) -> None:
    include_defaults = config.include_defaults
    configured_prompt_paths = list(config.prompt_paths)

    project_dirs: tuple[str, ...] = tuple(
        str(p) for p in api.get_project_layout().prompts_dirs()
    )
    registry: PromptRegistry = _PromptRegistry(project_prompt_dirs=project_dirs)
    from agentm.core.abi import PROMPT_TEMPLATES_SERVICE
    api.set_service(PROMPT_TEMPLATES_SERVICE, registry)

    cache: list[PromptTemplateRecord] = []

    async def _populate(_: SessionReadyEvent) -> None:
        responses = await api.events.emit(
            ResourcesDiscoverEvent.CHANNEL,
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
        cache[:] = registry.load_prompt_templates(
            cwd=api.cwd,
            agent_dir=str(Path.home() / ".agentm"),
            prompt_paths=tuple(prompt_paths),
            include_defaults=include_defaults,
        )

    def _on_input(event: dict[str, Any]) -> None:
        text = event.get("text")
        if not isinstance(text, str) or not text.startswith("/"):
            return
        expanded = registry.expand_prompt_template(text, cache)
        if expanded is not None:
            event["text"] = expanded

    api.on(SessionReadyEvent.CHANNEL, _populate)
    api.on("input", _on_input)
