"""Prompt builder/config carrier for seed verification agents."""
from __future__ import annotations

import json
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import ExtensionAPI
from agentm.extensions import ExtensionManifest

_FAULT_CONTEXT_TEMPLATE = "Injected fault: {fault_kind} on {target}"


class SeedContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    fault_kind: str
    params: str = ""
    fault_doc: str = ""
    observation_surface: dict[str, Any] = Field(default_factory=dict)
    judge_context: str = ""


MANIFEST = ExtensionManifest(
    name="seed_context",
    description="Builds structured seed verification prompts.",
    registers=(),
    config_schema=SeedContextConfig,
)


def build_seed_prompt(
    *,
    target: str,
    fault_kind: str,
    params: str,
    fault_doc: str,
    observation_surface: dict[str, Any] | None = None,
    judge_context: str = "",
) -> str:
    """Build the user prompt for verifying an injection target."""
    sections: list[str] = []

    # -- Fault reference document --
    if fault_doc:
        sections.append(f"## Fault reference document: {fault_kind}\n{fault_doc}")
    else:
        sections.append(f"## Fault reference document: {fault_kind}\n(no reference document available)")

    # -- Current injection --
    line = _FAULT_CONTEXT_TEMPLATE.format(fault_kind=fault_kind, target=target)
    if params:
        line += f" ({params})"
    sections.append("## Current injection\n" + line)

    if observation_surface:
        sections.append(
            "## Observation surface\n"
            "This deterministic map lists nearby services, available telemetry, "
            "and visible normal/abnormal anomalies. Use it to scope your checks; "
            "do not treat it as proof of causality.\n"
            "```json\n"
            + json.dumps(
                observation_surface,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n```"
        )

    # -- Task --
    task_lines = [
        f"Target: **{target}**",
        "Verify whether this injected fault actually took effect by comparing normal vs abnormal traces, metrics, logs, and caller/link behavior for the target.",
    ]
    if target.startswith("link:") and "->" in target:
        left, right = target.removeprefix("link:").split("->", 1)
        task_lines.append(
            f"Link endpoints: **{left}** and **{right}**. Establish which direction is actually exercised with normal parent-span joins; if the configured direction is both or unclear, check both `{left} -> {right}` and `{right} -> {left}`. Also compare the rule-bearing/source service's own outbound/client spans to the peer; the peer's server span may stay healthy for link delay/loss/partition faults. In the abnormal window, missing child spans across a partitioned link can be the expected fault signature only when source-owned parent work or caller attempts are still present and show timeout/error/log/latency symptoms. If the source service is idle because an upstream flow stopped calling it, do not confirm the link from zero traffic alone. For non-severing link faults such as bandwidth, delay, loss, duplicate, or corrupt, zero target spans or zero datastore calls alone is not enough to confirm the injection; require fault-shaped link/client/caller evidence, or mark the seed rejected/inconclusive if the link was not visibly exercised."
        )
    sections.append("## Task\n" + "\n".join(task_lines))

    if judge_context:
        sections.append(
            "## Retry / audit feedback context\n"
            + judge_context
            + "\n\nUse this context to continue from the previous failed attempt. "
            "It summarizes prior submitted evidence and the gate/audit objections, "
            "but it is not a substitute for final evidence. Re-run or repair SQL "
            "as needed and submit a verdict only when the new evidence answers "
            "the missing checks."
        )

    return "\n\n".join(sections)


def install(api: ExtensionAPI, config: SeedContextConfig) -> None:
    del api, config


__all__: Final = ["MANIFEST", "install", "build_seed_prompt"]
