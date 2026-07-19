"""Builtin prompt-template registry + slash-command expansion atom.

Owns the on-disk markdown template loader, the ``/<name> args`` substituter,
and the in-memory named-prompt registry used by the compaction subsystem.
Constructs a :class:`_PromptRegistry` instance at install time and registers
it under the ``"prompt_templates"`` service key so other atoms reach it via
``api.services.get("prompt_templates")``.

TODO(migration): the current branch has no ``InputEvent`` / command surface and
no project-layout / ``agentm_home_dir`` discovery, so the slash-command
expansion and filesystem auto-population that this atom performed on ``main``
are not wired here. The in-memory named-prompt registry (``register_prompt`` /
``get_prompt``) — the piece the ``compaction_prompts`` atom depends on — is
fully preserved and published as a service. ``load_prompt_templates`` /
``expand_prompt_template`` remain as a callable API for whoever re-introduces
an input hook, but nothing drives them automatically.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import AtomAPI, AtomInstallPriority
from agentm.core.lib import expand_path_from_cwd
from agentm.extensions import ExtensionManifest

# Locally owned service key (single-file contract: no atom-to-atom imports).
# ``compaction_prompts`` publishes/consumes the same string.
PROMPT_TEMPLATES_SERVICE = "prompt_templates"

# --- Module-level helpers (private) ----------------------------------------


class PromptTemplateRecord(BaseModel):
    """One loaded markdown prompt template.

    Local replacement for the ``main``-branch abi type of the same name.
    """

    name: str
    description: str = ""
    argument_hint: str | None = None
    body: str = ""
    file_path: str = ""
    source: str = ""


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Split a leading ``---`` YAML-ish frontmatter block from the body.

    Local minimal replacement for ``agentm.core.lib.parse_frontmatter`` (not
    present on this branch). Only ``key: value`` scalar pairs are recognized —
    enough for ``description`` / ``argument-hint``.
    """

    if not raw.startswith("---"):
        return {}, raw
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    metadata: dict[str, str] = {}
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            body = "\n".join(lines[index + 1 :])
            return metadata, body
        key, sep, value = line.partition(":")
        if sep:
            metadata[key.strip()] = value.strip()
    # No closing fence: treat the whole thing as body.
    return {}, raw


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
        lambda match: (
            args[int(match.group(1)) - 1]
            if 0 < int(match.group(1)) <= len(args)
            else ""
        ),
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
    except OSError as exc:
        logger.debug("prompt_templates: failed to read {}: {}", file_path, exc)
        return None

    metadata, body = _parse_frontmatter(raw)
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
    except OSError as exc:
        logger.debug("prompt_templates: failed to list {}: {}", directory, exc)
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
    """In-memory named-prompt registry + markdown template loader.

    One instance per session, owned by this atom and published via
    ``api.services.register("prompt_templates", registry)``.
    """

    def __init__(self, project_prompt_dirs: tuple[str, ...] | None = None) -> None:
        self._project_dirs = project_prompt_dirs
        self._registry: dict[str, str] = {}

    def load_prompt_templates(
        self,
        *,
        cwd: str,
        agent_dir: str | None = None,
        prompt_paths: list[str] | tuple[str, ...] = (),
        include_defaults: bool = True,
    ) -> list[PromptTemplateRecord]:
        templates: list[PromptTemplateRecord] = []

        if include_defaults:
            if agent_dir:
                templates.extend(
                    _load_templates_from_dir(
                        os.path.join(agent_dir, "prompts"), "user"
                    )
                )
            project_dirs: tuple[str, ...] = self._project_dirs or ()
            for project_dir in project_dirs:
                templates.extend(_load_templates_from_dir(project_dir, "project"))

        for raw_path in prompt_paths:
            resolved_path = str(expand_path_from_cwd(raw_path, cwd))
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
            return _substitute_args(
                template.body, _parse_command_args(raw_args.strip())
            )
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
    description="Host the named-prompt registry used by the compaction subsystem.",
    # TODO(migration): on main this also registered event:input /
    # event:resources_discover / event:session_ready to expand /<name> args
    # templates. Those surfaces do not exist on this branch.
    registers=(),
    config_schema=PromptTemplatesConfig,
    requires=(),
    priority=AtomInstallPriority.CONTEXT,
)


class _PromptTemplatesRuntime:
    def __init__(self, api: AtomAPI, config: PromptTemplatesConfig) -> None:
        self._api = api
        self._include_defaults = config.include_defaults
        self._configured_prompt_paths = list(config.prompt_paths)
        self._registry = _PromptRegistry()
        self._cache: list[PromptTemplateRecord] = []

    def install(self) -> None:
        self._api.services.register(
            PROMPT_TEMPLATES_SERVICE, self._registry, scope="session"
        )
        # TODO(migration): populate() / on_input() from main are not wired —
        # this branch has no ResourcesDiscover response contract or InputEvent
        # to drive filesystem template loading and slash expansion.


def install(api: AtomAPI, config: PromptTemplatesConfig) -> None:
    _PromptTemplatesRuntime(api, config).install()
