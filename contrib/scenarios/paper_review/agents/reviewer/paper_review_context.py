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


def _paper_reader_process_rules() -> str:
    return _section(
        "Pass 1 Linear Reading Process Requirements",
        "\n".join([
            "These extra requirements apply only to `paper-reader`.",
            "",
            "1. First discover the real reading order from the paper entry "
            "point. Follow the rendered document order: title/front matter, "
            "abstract, introduction, then later chapters/sections.",
            "2. Do not batch-read future sections or chapters. Read one "
            "reading unit at a time: abstract, section, subsection, or a "
            "small paragraph chunk when a section is large.",
            "3. Treat `Reading Notes` as the reader's live mental state, not "
            "as a final summary. After every reading unit, persist the "
            "current notes to the output artifact path using `write` or "
            "`edit` before reading the next unit. The trace must show "
            "multiple continuous artifact updates, not one final write after "
            "reading the whole paper.",
            "4. If the output artifact already exists or `write` reports "
            "that it exists, read it first and then overwrite or edit it. "
            "Do not skip incremental persistence because of an existing file.",
            "5. Each persisted update must reflect only text already read. "
            "Never use later sections to explain, resolve, soften, or "
            "reinterpret confusion from an earlier section.",
            "6. In Pass 1, keep unresolved confusion in `Open Questions` and "
            "keep future-looking commitments in `Promises To Verify Later`. "
            "Do not mark promises fulfilled and do not move questions to "
            "`Resolved Questions`.",
            "7. The artifact should expose the thinking trajectory: each "
            "update should make clear what was known, what became confusing, "
            "what questions were opened, and what promises were recorded at "
            "that point in the reading order.",
            "8. The final artifact may clean up formatting, but it must "
            "preserve the first-read state trajectory accumulated through "
            "the incremental updates.",
        ]),
    )


def _paper_review_final_rules() -> str:
    return _section(
        "Final Report Synthesis Requirements",
        "\n".join([
            "These extra requirements apply only to the final `paper-review` pass.",
            "",
            "1. Read every prior pass artifact before writing the final report.",
            "2. The final report is a synthesis, not a pass-by-pass concatenation. "
            "Do not preserve duplicate findings just because multiple passes "
            "reported them.",
            "3. Merge overlapping findings into one canonical item. Preserve the "
            "strongest wording, all important source locations, and the pass "
            "evidence that supports the merged item.",
            "4. Group findings by aspect first, then sort within each aspect by "
            "severity and revision impact. Choose aspect names that fit the "
            "paper, such as evidence support, argument structure, consistency "
            "and notation, reader flow, prose/style, figures/tables, and "
            "future-work feasibility.",
            "5. Avoid duplicate blocker/major items for the same underlying "
            "problem. If one issue is a special case of another, keep the "
            "broader canonical issue and mention the special case under it.",
            "6. Keep a concise executive summary and a prioritized revision "
            "plan. Counts must match the actual grouped findings.",
            "7. Include pass provenance compactly, for example `Sources: "
            "reader, consistency, evidence`, instead of copying full prior "
            "artifact sections verbatim.",
            "8. Use the current run context when writing metadata. If the exact "
            "date is not provided, omit the date instead of inventing one.",
        ]),
    )


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

    if config.role == "paper-reader":
        parts.extend(["", _paper_reader_process_rules()])
    elif config.role == "paper-review":
        parts.extend(["", _paper_review_final_rules()])

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
