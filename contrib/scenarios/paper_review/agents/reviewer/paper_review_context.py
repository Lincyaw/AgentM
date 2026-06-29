"""Context injection for paper_review reviewer workers."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest


class PaperReviewContextConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = ""
    skill_name: str = ""
    pass_name: str = ""
    task: str = ""
    input_path: str = ""
    prior_markdown: str = ""


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
        "the authority for this role. If that skill is unavailable, return "
        "a Markdown error section explaining the missing skill.",
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
        "Return Markdown only, using the loaded skill's output shape where "
        "possible. Do not wrap the result in JSON.",
    ]

    task_section = _section("Task", config.task)
    if task_section:
        parts.extend(["", task_section])

    if config.input_path:
        parts.extend(["", _section("Input Path", config.input_path)])

    if config.prior_markdown:
        parts.extend([
            "",
            _section("Prior Pass Markdown Artifacts", config.prior_markdown),
        ])

    context = "\n".join(part for part in parts if part != "")

    def before_agent_start(event: BeforeAgentStartEvent) -> dict[str, str]:
        current = str(event.system or "")
        updated = f"{current}\n\n{context}" if current else context
        event.system = updated
        return {"system": updated}

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)


__all__: Final = ["MANIFEST", "install"]
