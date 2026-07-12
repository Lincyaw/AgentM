"""Auditor context atom — builds the system prompt from raw data passed by the parent."""

from __future__ import annotations

import json
from collections import Counter
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


def _build_index_summary(context_index: dict[str, Any]) -> str:
    """Build a compact summary of the context index for the system prompt.

    Handles both the legacy format (entities/turns/observations) and the
    trajectory-index format (symbols/references/attention_hints).
    """
    entities = context_index.get("entities", [])
    symbols = context_index.get("symbols", [])
    observations = context_index.get("observations", [])
    references = context_index.get("references", [])
    candidates = context_index.get("candidates", [])
    turns = context_index.get("turns", [])

    if not entities and not turns and not symbols:
        return ""

    lines: list[str] = []

    if symbols:
        kind_counts = Counter(s.get("kind", "unknown") for s in symbols)
        lines.append(f"Index: {len(symbols)} symbols, {len(references)} references")
        lines.append(f"Symbol kinds: {dict(kind_counts)}")
        grounded = sum(1 for r in references if r.get("grounded"))
        if references:
            lines.append(f"Grounded references: {grounded}/{len(references)}")
    else:
        lines.append(f"Trajectory: {len(turns)} turns, {len(entities)} entities, {len(observations)} observations")
        type_counts = Counter(e.get("type", "unknown") for e in entities)
        if type_counts:
            lines.append(f"Entity types: {dict(type_counts)}")
        state_counts = Counter(c.get("state", "?") for c in candidates)
        if state_counts:
            lines.append(f"Candidate states: {dict(state_counts)}")

    attention_hints = context_index.get("attention_hints", [])
    if attention_hints:
        lines.append(f"Attention hints: {len(attention_hints)}")
        for h in attention_hints[:5]:
            lines.append(f"  - [{h.get('kind', '?')}] {h.get('summary', '')[:200]}")

    constraint_findings = context_index.get("constraint_findings") or []
    if constraint_findings:
        lines.append("\nConstraint analysis (question requirements vs gathered evidence):")
        for f in constraint_findings[:12]:
            status = f.get("status", "?")
            desc = str(f.get("description", ""))[:140]
            candidate = f.get("candidate", "")
            anchor = f.get("commit_step_id")
            src = f.get("confidence_source", "")
            line = f"  - [{status}] '{desc}' for candidate '{candidate}'"
            if status in ("violated", "omitted") and anchor:
                line += f" (anchor: step {anchor})"
            if src:
                line += f" [{src}]"
            lines.append(line)
            reason = str(f.get("reason", ""))
            if reason and status in ("violated", "omitted"):
                lines.append(f"      {reason[:160]}")

    claim_structure = context_index.get("claim_structure")
    if claim_structure:
        span_roles = claim_structure.get("span_roles", {})
        commit_spans = sorted(int(k) for k, v in span_roles.items() if v in ("commit", "verify", "finalize"))
        explore_spans = sorted(int(k) for k, v in span_roles.items() if v == "explore")
        lines.append("\nClaim analysis (Level 2):")
        lines.append(f"  Commitment spans: {commit_spans}")
        lines.append(f"  Exploration spans (NOT errors): {explore_spans}")
        points = claim_structure.get("commitment_points", [])
        if points:
            lines.append("  Ungrounded commitments:")
            for cp in points[:8]:
                lines.append(f"    span {cp.get('span')}: {cp.get('entity','')} (grounded={cp.get('grounded',False)}) — {cp.get('reason','')[:100]}")

    return "\n".join(lines)


def build_auditor_system_prompt(
    *,
    check_errors: dict[str, str],
    continuation_notes: list[str],
    base_prompt: str | None = None,
    methodology: list[str] | None = None,
    context_index: dict[str, Any] | None = None,
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

    if context_index is not None:
        summary = _build_index_summary(context_index)
        if summary:
            sections.append("## INDEX OVERVIEW")
            sections.append(summary)
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
            "Use them as background for domain terms, expected evidence shapes, "
            "and causal reasoning patterns. Do not treat every checklist item in "
            "these skills as an automatic reminder-worthy gap; concrete facts must "
            "still come from the trajectory."
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
    prompt_name: str = "index"
    trajectory_snapshot: list[dict[str, Any]] | None = None
    mode: Literal["index", "trajectory"] = "index"
    context_index: dict[str, Any] | None = None
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
    "build_auditor_trajectory_prompt",
    "install",
    "load_auditor_prompt",
]
