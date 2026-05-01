"""Prompt-template record type — public ABI for prompt-template atoms.

Pure data type. The loading / expansion helpers stay in
``_internal/prompt_templates.py`` and reach atoms through
``ExtensionAPI.prompt_templates``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptTemplateRecord:
    name: str
    description: str
    argument_hint: str | None
    body: str
    file_path: str
    source: str


__all__ = ["PromptTemplateRecord"]
