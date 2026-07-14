"""Auditor context atom — builds the system prompt from raw data passed by the parent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR: Final = Path(__file__).parent / "prompts"
_BUILTIN_NAMES: Final = frozenset(p.stem for p in _PROMPTS_DIR.glob("*.md"))


def load_auditor_prompt(name: str = "index") -> str:
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
    goal_condition: str | None = None,
) -> str:
    """Assemble the auditor system prompt for one firing."""
    framing = base_prompt if base_prompt is not None else load_auditor_prompt("index")

    sections: list[str] = [framing.rstrip(), ""]

    if goal_condition:
        sections.append("## GOAL CONDITION")
        sections.append(
            "The main agent must satisfy this condition to complete the task. "
            "Use it to judge whether the agent is on track, and whether its "
            "claims of completion are backed by actual evidence (e.g., test "
            "execution output, not just code that compiles)."
        )
        sections.append(goal_condition.strip())
        sections.append("")

    if methodology:
        sections.append("## METHODOLOGY (loaded by main agent)")
        sections.append(
            "The main agent loaded these domain-specific skills during its session. "
            "Use them as background for domain terms, expected evidence shapes, "
            "and causal reasoning patterns. Do not treat every checklist item in "
            "these skills as an automatic reminder-worthy gap; concrete facts must "
            "still come from the trajectory."
        )
        for i, skill_text in enumerate(methodology):
            sections.append(f"### Skill {i + 1}")
            sections.append(skill_text.strip())
        sections.append("")

    # No pre-rendered index dump: the auditor queries the index on demand
    # through its tools (list_turns / get_turn / list_entities /
    # list_claim_checks / ...). Pre-dumping the summary was noise (symbol
    # counts, fabricated_name hints that misfire on deep-research
    # trajectories) and the load-bearing signal — the claim/requirement
    # checks — is now served by list_claim_checks when the auditor wants it.

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

class AuditorContextConfig(BaseModel):
    # Extra keys the parent may still pass (symbols / references /
    # trajectory_snapshot / index_path) are for the trajectory_index_query
    # atom, not this one, and are ignored here by pydantic.
    check_errors: dict[str, str] = {}
    continuation_notes: list[str] = []
    prompt_name: str = "index"
    methodology: list[str] = []  # kept for backward compat
    goal_condition: str | None = None

MANIFEST = ExtensionManifest(
    name="auditor_context",
    description="Build the auditor system prompt from the context index.",
    registers=("event:before_agent_start",),
    config_schema=AuditorContextConfig,
)

def install(api: ExtensionAPI, config: AuditorContextConfig) -> None:
    base_prompt = load_auditor_prompt(config.prompt_name)
    meth = config.methodology or None

    prompt_text = build_auditor_system_prompt(
        check_errors=config.check_errors,
        continuation_notes=config.continuation_notes,
        base_prompt=base_prompt,
        methodology=meth,
        goal_condition=config.goal_condition,
    )

    def _before_start(event: BeforeAgentStartEvent) -> dict[str, str]:
        current = event.system or ""
        merged = f"{prompt_text}\n\n{current}" if current else prompt_text
        event.system = merged
        return {"system": merged}

    api.on(BeforeAgentStartEvent.CHANNEL, _before_start)

__all__: Final = [
    "MANIFEST",
    "build_auditor_system_prompt",
    "install",
    "load_auditor_prompt",
]
