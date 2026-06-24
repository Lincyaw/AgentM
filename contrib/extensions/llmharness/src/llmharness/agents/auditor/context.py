"""Auditor context atom — builds the system prompt from raw data passed by the parent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel

from llmharness.context_index import build_context_index

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR: Final = Path(__file__).parent / "prompts"
_BUILTIN_NAMES: Final = frozenset(p.stem for p in _PROMPTS_DIR.glob("*.md"))


def load_auditor_prompt(name: str = "minimal_index") -> str:
    """Load auditor prompt by name or absolute path."""
    md = _PROMPTS_DIR / f"{name}.md"
    if md.is_file():
        return md.read_text(encoding="utf-8")
    path = Path(name).expanduser()
    if path.is_file():
        return path.read_text(encoding="utf-8")
    raise ValueError(
        f"unknown auditor prompt {name!r}; "
        f"available: {sorted(_BUILTIN_NAMES)}"
    )


# ---------------------------------------------------------------------------
# System-prompt assembly
# ---------------------------------------------------------------------------


def build_auditor_system_prompt(
    *,
    check_errors: dict[str, str],
    continuation_notes: list[str],
    base_prompt: str | None = None,
    methodology: list[str] | None = None,
    context_index: dict[str, Any] | None = None,
) -> str:
    """Assemble the auditor system prompt for one firing."""
    framing = base_prompt if base_prompt is not None else load_auditor_prompt("minimal_index")

    sections: list[str] = [framing.rstrip(), ""]

    if methodology:
        sections.append("## METHODOLOGY (loaded by main agent)")
        sections.append(
            "The main agent loaded these domain-specific skills during its session. "
            "Use them to evaluate whether the agent's reasoning follows the methodology — "
            "not just whether it investigated every entity mentioned."
        )
        for i, skill_text in enumerate(methodology):
            sections.append(f"### Skill {i + 1}")
            sections.append(skill_text.strip())
        sections.append("")

    if context_index is not None:
        sections.append("## CONTEXT_INDEX (primary navigation view)")
        sections.append(
            "This is an LSP-style index over the visible trajectory prefix. "
            "Use it to locate entities, observations, claims, candidate lifecycle "
            "events, obligations, and contract failures. It is not a causal proof."
        )
        sections.append(json.dumps(context_index, ensure_ascii=False))
        sections.append("")
    else:
        sections.append("## CONTEXT_INDEX (primary navigation view)")
        sections.append("{}")
        sections.append("")

    if check_errors:
        sections.append("## CHECK_ERRORS (non-blocking)")
        sections.append(
            json.dumps(check_errors, ensure_ascii=False)
        )
        sections.append("")

    sections.append("## CONTINUATION_NOTES (from your prior firing)")
    sections.append(json.dumps(list(continuation_notes), ensure_ascii=False))
    sections.append("")

    return "\n".join(sections)


def build_auditor_trajectory_prompt(
    *,
    trajectory: list[dict[str, Any]],
    continuation_notes: list[str],
    base_prompt: str | None = None,
    methodology: list[str] | None = None,
    context_index: dict[str, Any] | None = None,
) -> str:
    """Assemble the auditor system prompt for a trajectory-mode firing."""
    framing = (
        base_prompt
        if base_prompt is not None
        else load_auditor_prompt("trajectory")
    )
    sections: list[str] = [framing.rstrip(), ""]

    if methodology:
        sections.append("## METHODOLOGY (loaded by main agent)")
        sections.append(
            "The main agent loaded these domain-specific skills during its session. "
            "Use them to evaluate whether the agent's reasoning follows the methodology — "
            "not just whether it investigated every entity mentioned."
        )
        for i, skill_text in enumerate(methodology):
            sections.append(f"### Skill {i + 1}")
            sections.append(skill_text.strip())
        sections.append("")

    if context_index is not None:
        sections.append("## CONTEXT_INDEX (primary navigation view)")
        sections.append(json.dumps(context_index, ensure_ascii=False))
        sections.append("")

    sections.append("## TRAJECTORY")
    sections.append(f"conversation turns ({len(trajectory)} total):")
    sections.append(json.dumps(trajectory, ensure_ascii=False))
    sections.append("")

    sections.append("## CONTINUATION_NOTES (from your prior firing)")
    sections.append(json.dumps(list(continuation_notes), ensure_ascii=False))
    sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Atom
# ---------------------------------------------------------------------------


class AuditorContextConfig(BaseModel):
    symbols: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    check_errors: dict[str, str] = {}
    continuation_notes: list[str] = []
    prompt_name: str = "minimal_index"
    trajectory_snapshot: list[dict[str, Any]] | None = None
    mode: Literal["index", "trajectory"] = "index"
    context_index: dict[str, Any] | None = None
    methodology: list[str] = []

MANIFEST = ExtensionManifest(
    name="auditor_context",
    description="Build the auditor system prompt from the context index.",
    registers=("event:before_agent_start",),
    config_schema=AuditorContextConfig,
)

def install(api: ExtensionAPI, config: AuditorContextConfig) -> None:
    base_prompt = load_auditor_prompt(config.prompt_name)
    meth = config.methodology or None
    context_index = config.context_index
    if context_index is None and config.trajectory_snapshot is not None:
        context_index = build_context_index(
            trajectory=config.trajectory_snapshot,
            symbols=config.symbols,
            references=config.references,
        ).to_dict()

    if config.mode == "trajectory" and config.trajectory_snapshot is not None:
        prompt_text = build_auditor_trajectory_prompt(
            trajectory=config.trajectory_snapshot,
            continuation_notes=config.continuation_notes,
            base_prompt=base_prompt,
            methodology=meth,
            context_index=context_index,
        )
    else:
        prompt_text = build_auditor_system_prompt(
            check_errors=config.check_errors,
            continuation_notes=config.continuation_notes,
            base_prompt=base_prompt,
            methodology=meth,
            context_index=context_index,
        )

    def _before_start(event: BeforeAgentStartEvent) -> dict[str, str]:
        return {"system": prompt_text}

    api.on(BeforeAgentStartEvent.CHANNEL, _before_start)

__all__: Final = [
    "MANIFEST",
    "build_auditor_system_prompt",
    "build_auditor_trajectory_prompt",
    "install",
    "load_auditor_prompt",
]
