"""Prompt builders/config carrier for verifier discovery gates."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from agentm.core.abi import ExtensionAPI
from agentm.extensions import ExtensionManifest


class GateContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_kind: str
    task: dict[str, Any]
    submitted_result: dict[str, Any]
    child_session: dict[str, Any] | None = None
    attempt: int = 0


MANIFEST = ExtensionManifest(
    name="gate_context",
    description="Builds one discovery result prompt for gate review.",
    registers=(),
    config_schema=GateContextConfig,
)


def build_gate_prompt(
    *,
    task_kind: str,
    task: Mapping[str, Any],
    submitted_result: Mapping[str, Any],
    child_session: Mapping[str, Any] | None,
    attempt: int,
) -> str:
    """Build a gate review prompt for one seed/hop discovery result."""
    payload = {
        "task_kind": task_kind,
        "attempt": attempt,
        "task": task,
        "submitted_result": submitted_result,
        "child_session": child_session or {},
    }
    return (
        "## Gate input\n"
        "Review this one discovery result for investigation completeness. "
        "Use `task.fault_reference_document` as the authoritative fault "
        "signature when deciding whether the investigation covered the "
        "right trace, metric, log, caller, and propagation checks. "
        "Return only the structured `submit_result` payload.\n\n"
        "```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n```"
    )


def install(api: ExtensionAPI, config: GateContextConfig) -> None:
    del api, config


__all__: Final = ["MANIFEST", "install", "build_gate_prompt"]
