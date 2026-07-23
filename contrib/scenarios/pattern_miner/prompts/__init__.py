"""Prompt builders, bound to the agents that consume them (host-side).

``prompts/miner/`` holds the miner agent's angle prompts — one feedback
dimension per markdown file, plus the attribution contract.
``prompts/distiller/`` holds the distiller agent's contract. The manifest
system prompts carry "how to think"; these build "what to check" per run.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent
_MINER_DIR = _PROMPTS_DIR / "miner"
_DISTILLER_DIR = _PROMPTS_DIR / "distiller"

ANGLES: tuple[str, ...] = (
    "label_validity",
    "interpretation_fidelity",
    "requirement_coverage",
    "implementation_fidelity",
    "diagnosis_validity",
    "evidence_adequacy",
    "belief_consistency",
    "loop_dynamics",
)


def _miner_text(name: str) -> str:
    return (_MINER_DIR / f"{name}.md").read_text(encoding="utf-8")


def _case_header(meta: Mapping[str, object]) -> str:
    return (
        f"Case: task {meta.get('task_name')} / trial {meta.get('trial_name')}\n"
        f"Agent model: {meta.get('agent_model')}\n"
        f"Recorded reward: {meta.get('reward')}\n"
    )


_ENV_NOTE = (
    "\nA live task environment is attached (env_bash tool): the repository "
    "in its pre-fix state, the agent's final patch at /tmp/agent.patch, the "
    "oracle patch at /tmp/oracle.patch. Probing it yourself — applying "
    "patches, building, running tests, driving the scenario — creates "
    "independent evidence; prefer a measurement over an inference whenever "
    "the two could disagree.\n"
)


def build_angle_prompt(
    angle: str, meta: Mapping[str, object], *, env_attached: bool = False
) -> str:
    if angle not in ANGLES:
        raise ValueError(f"unknown angle {angle!r}; known: {', '.join(ANGLES)}")
    return (
        _case_header(meta)
        + "\n"
        + _miner_text(angle)
        + (_ENV_NOTE if env_attached else "")
        + "\n"
        + f"Dimension under review: {angle}. Explore the case through the "
        "case tools, then submit your report with submit_result, setting the "
        f"report's dimension field to {angle!r}.\n"
    )


def build_attribution_prompt(
    meta: Mapping[str, object], reports_json: str, *, env_attached: bool = False
) -> str:
    return (
        _case_header(meta)
        + "\n"
        + _miner_text("attribution")
        + (_ENV_NOTE if env_attached else "")
        + "\nPer-dimension reports for this case:\n\n"
        + reports_json
        + "\n\nRe-check any cited turns you need through the case tools, "
        "then submit your attribution with submit_result.\n"
    )


def build_distill_prompt(
    dimension: str,
    findings_json: str,
    existing_items_json: str | None,
) -> str:
    parts = [
        (_DISTILLER_DIR / "distill.md").read_text(encoding="utf-8"),
        f"\nDimension under distillation: {dimension}\n",
        "Findings across the batch (one entry per case finding, with trial "
        "provenance and the case's root attribution where available):\n",
        findings_json,
    ]
    if existing_items_json:
        parts.append(
            "\nExisting checklist items for this dimension (merge into these; "
            "add a new item only for an uncovered mechanism):\n"
        )
        parts.append(existing_items_json)
    parts.append("\nSubmit the merged checklist for this dimension with submit_result.")
    return "\n".join(parts)
