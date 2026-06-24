"""Prompt builders/config carrier for verifier audit agents."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import ExtensionAPI
from agentm.extensions import ExtensionManifest


class AuditContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    instruction: str
    payload: dict[str, Any] = Field(default_factory=dict)


MANIFEST = ExtensionManifest(
    name="audit_context",
    description="Builds one bounded verifier audit prompt.",
    registers=(),
    config_schema=AuditContextConfig,
)


def _payload_prompt(role: str, instruction: str, payload: Mapping[str, Any]) -> str:
    return (
        f"## Audit role\n{role}\n\n"
        f"## Bounded question\n{instruction}\n\n"
        "## Input payload\n"
        "```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n```\n\n"
        "Return only the structured `submit_result` payload."
    )


def build_audit_prompt(
    *,
    role: str,
    instruction: str,
    payload: Mapping[str, Any],
) -> str:
    return _payload_prompt(role, instruction, payload)


def install(api: ExtensionAPI, config: AuditContextConfig) -> None:
    del api, config


__all__: Final = [
    "MANIFEST",
    "install",
    "build_audit_prompt",
]
