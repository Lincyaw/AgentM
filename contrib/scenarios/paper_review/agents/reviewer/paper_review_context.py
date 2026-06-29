"""Context injection for paper_review reviewer workers."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest


class PaperReviewContextConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = ""
    skill_name: str = ""
    pass_name: str = ""
    task: str = ""
    input_path: str = ""
    output_artifact_path: str = ""
    prior_artifact_paths: list[str] = Field(default_factory=list)


MANIFEST = ExtensionManifest(
    name="paper_review_context",
    description="Inject paper review pass context into reviewer workers.",
    registers=("event:before_agent_start",),
    config_schema=PaperReviewContextConfig,
)


def _section(title: str, body: str) -> str:
    clean = body.strip()
    if not clean:
        return ""
    return f"## {title}\n\n{clean}"


def install(api: ExtensionAPI, config: PaperReviewContextConfig) -> None:
    parts: list[str] = [
        "# Paper Review Workflow Context",
        "",
        f"Role: {config.role}",
        f"Pass: {config.pass_name}",
        f"Required skill: {config.skill_name}",
        "",
        "Before doing any review work, call `load_skill` with exactly this "
        f"name: `{config.skill_name}`. Use the loaded skill instructions as "
        "the authority for this role. If that skill is unavailable, write "
        "a Markdown error artifact explaining the missing skill to the output "
        "artifact path and then return a short status.",
        "",
        "Only load the required skill for this role. Do not load other skills.",
        "",
        "Use the available file tools and bash as needed to inspect the "
        "paper. If the input path is a directory, discover the appropriate "
        "entry file yourself instead of assuming every `.tex` file under the "
        "directory belongs to the paper. Prefer likely entry points such as "
        "`thesis.tex`, `main.tex`, or `paper.tex`, then follow LaTeX "
        "`\\input` / `\\include` references. Avoid unrelated reference, "
        "template, cache, and hidden-tooling directories unless the paper "
        "itself references them.",
        "",
        "Write the complete Markdown artifact for this pass to the output "
        "artifact path below using the `write` tool. The workflow will not "
        "copy your final response into that file. After writing the file, "
        "return a short Markdown status line that names the artifact path. "
        "Do not wrap the status in JSON.",
    ]

    task_section = _section("Task", config.task)
    if task_section:
        parts.extend(["", task_section])

    if config.input_path:
        parts.extend(["", _section("Input Path", config.input_path)])

    if config.output_artifact_path:
        parts.extend(["", _section("Output Artifact Path", config.output_artifact_path)])

    if config.prior_artifact_paths:
        paths = "\n".join(f"- {path}" for path in config.prior_artifact_paths)
        parts.extend([
            "",
            _section("Prior Pass Artifact Paths", paths),
            "",
            "Read these Markdown files as needed. They are the persisted "
            "outputs from earlier passes and are the only handoff channel.",
        ])

    context = "\n".join(part for part in parts if part != "")

    def before_agent_start(event: BeforeAgentStartEvent) -> dict[str, str]:
        current = str(event.system or "")
        updated = f"{current}\n\n{context}" if current else context
        event.system = updated
        return {"system": updated}

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)


__all__: Final = ["MANIFEST", "install"]
