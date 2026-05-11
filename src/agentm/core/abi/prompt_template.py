"""Prompt-template ABI — record type and registry Protocol.

The :class:`PromptTemplateRecord` dataclass and :class:`PromptRegistry`
Protocol form the stable atom-facing surface. The default implementation
lives in the ``prompt_templates`` builtin atom and is registered as a
service at install time via ``api.set_service("prompt_templates", ...)``.
Consumers reach the registry through ``api.get_service("prompt_templates")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class PromptTemplateRecord:
    name: str
    description: str
    argument_hint: str | None
    body: str
    file_path: str
    source: str


@runtime_checkable
class PromptRegistry(Protocol):
    """Atom-facing port for the prompt-template registry.

    The default implementation is provided by the ``prompt_templates``
    builtin atom and exposed via ``api.get_service("prompt_templates")``.
    """

    def load_prompt_templates(
        self,
        *,
        cwd: str,
        agent_dir: str,
        prompt_paths: list[str] | tuple[str, ...] = (),
        include_defaults: bool = True,
    ) -> list[PromptTemplateRecord]: ...

    def expand_prompt_template(
        self,
        text: str,
        templates: list[PromptTemplateRecord],
    ) -> str | None: ...

    def register_prompt(self, name: str, body: str) -> None: ...

    def get_prompt(self, name: str) -> str | None: ...


__all__ = ["PromptRegistry", "PromptTemplateRecord"]
