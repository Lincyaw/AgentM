"""Auditor context atom — builds the system prompt from raw data passed by the parent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel

from llmharness.context_index import build_context_index
from llmharness.schema import Edge, Event, Finding, Phase

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


def _degrade_event(ev_dict: dict[str, object]) -> dict[str, object]:
    d: dict[str, object] = {
        "id": ev_dict.get("id"),
        "kind": ev_dict.get("kind"),
        "summary": ev_dict.get("summary"),
        "source_turns": ev_dict.get("source_turns", []),
    }
    if "status" in ev_dict and ev_dict["status"] is not None:
        d["status"] = ev_dict["status"]
    return d


def _degrade_edge(ed_dict: dict[str, object]) -> dict[str, object]:
    d: dict[str, object] = {
        "src": ed_dict.get("src"),
        "dst": ed_dict.get("dst"),
        "kind": ed_dict.get("kind"),
        "reason": ed_dict.get("reason"),
    }
    if "role" in ed_dict and ed_dict["role"] is not None:
        d["role"] = ed_dict["role"]
    return d


def build_auditor_system_prompt(
    *,
    events: tuple[Event, ...],
    edges: tuple[Edge, ...],
    phases: tuple[Phase, ...] = (),
    findings: list[Finding],
    check_errors: dict[str, str],
    continuation_notes: list[str],
    summary_threshold: int = 30,
    base_prompt: str | None = None,
    methodology: list[str] | None = None,
    context_index: dict[str, Any] | None = None,
    context_mode: Literal["graph", "index", "both"] = "index",
) -> str:
    """Assemble the auditor system prompt for one firing."""
    framing = base_prompt if base_prompt is not None else load_auditor_prompt("minimal_index")
    degraded = len(events) > summary_threshold
    show_index = context_mode in {"index", "both"} and context_index is not None
    show_graph = context_mode in {"graph", "both"} or not show_index

    if degraded:
        events_payload = [_degrade_event(ev.to_dict()) for ev in events]
        edges_payload = [_degrade_edge(ed.to_dict()) for ed in edges]
    else:
        events_payload = [ev.to_dict() for ev in events]
        edges_payload = [ed.to_dict() for ed in edges]

    findings_payload = [f.to_dict() for f in findings]

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

    if show_index:
        sections.append("## CONTEXT_INDEX (primary navigation view)")
        sections.append(
            "This is an LSP-style index over the visible trajectory prefix. "
            "Use it to locate entities, observations, claims, candidate lifecycle "
            "events, obligations, and contract failures. It is not a causal proof."
        )
        sections.append(json.dumps(context_index, ensure_ascii=False))
        sections.append("")

    if phases and show_graph:
        section_name = "COMPAT_PHASES" if show_index else "PHASES"
        sections.append(f"## {section_name} (merged basic blocks)")
        sections.append(
            f"phases ({len(phases)} total). Each phase wraps one or more raw "
            "events; ``member_event_ids`` lists them in order. Consecutive "
            "``act`` events are coalesced into ``act_run`` blocks; "
            "``task`` / ``hyp`` / ``dec`` / ``concl`` always stay singleton. "
            "Reason at this level by default; consult the raw events block "
            "below when a specific witness needs verification."
        )
        sections.append(json.dumps([p.to_dict() for p in phases], ensure_ascii=False))
        sections.append("")

    if show_graph:
        section_name = "COMPAT_GRAPH" if show_index else "GRAPH"
        sections.append(f"## {section_name}")
        sections.append(
            f"events ({len(events_payload)} total"
            + (
                f", degraded — threshold={summary_threshold}, witness fields stripped)"
                if degraded
                else ")"
            )
            + ":"
        )
        sections.append(json.dumps(events_payload, ensure_ascii=False))
        sections.append("")
        sections.append(f"edges ({len(edges_payload)} total):")
        sections.append(json.dumps(edges_payload, ensure_ascii=False))
        sections.append("")

    sections.append("## FINDINGS (advisory)")
    sections.append(json.dumps(findings_payload, ensure_ascii=False))
    if check_errors:
        sections.append(
            "checks_failed: "
            + json.dumps(check_errors, ensure_ascii=False)
            + " (non-blocking; other checks ran)"
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
    context_mode: Literal["trajectory", "index", "both"] = "index",
) -> str:
    """Assemble the auditor system prompt for a trajectory-mode firing."""
    framing = (
        base_prompt
        if base_prompt is not None
        else load_auditor_prompt("trajectory")
    )
    show_index = context_mode in {"index", "both"} and context_index is not None
    show_trajectory = context_mode in {"trajectory", "both"} or not show_index

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

    if show_index:
        sections.append("## CONTEXT_INDEX (primary navigation view)")
        sections.append(json.dumps(context_index, ensure_ascii=False))
        sections.append("")

    if show_trajectory:
        section_name = "COMPAT_TRAJECTORY" if show_index else "TRAJECTORY"
        sections.append(f"## {section_name}")
        sections.append(
            f"conversation turns ({len(trajectory)} total):"
        )
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
    events: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    check_errors: dict[str, str] = {}
    continuation_notes: list[str] = []
    summary_threshold: int = 30
    prompt_name: str = "minimal_index"
    trajectory_snapshot: list[dict[str, Any]] | None = None
    mode: Literal["graph", "trajectory"] = "graph"
    context_index: dict[str, Any] | None = None
    context_mode: Literal["graph", "index", "both"] = "index"
    methodology: list[str] = []

MANIFEST = ExtensionManifest(
    name="auditor_context",
    description="Build the auditor system prompt from raw graph data.",
    registers=("event:before_agent_start",),
    config_schema=AuditorContextConfig,
)

def install(api: ExtensionAPI, config: AuditorContextConfig) -> None:
    events = tuple(Event.from_dict(e) for e in config.events)
    edges = tuple(Edge.from_dict(e) for e in config.edges)
    phases = tuple(Phase.from_dict(p) for p in config.phases)
    findings = [Finding.from_dict(f) for f in config.findings]

    base_prompt = load_auditor_prompt(config.prompt_name)
    meth = config.methodology or None
    context_index = config.context_index
    if context_index is None and config.trajectory_snapshot is not None:
        context_index = build_context_index(
            trajectory=config.trajectory_snapshot,
            events=events,
            edges=edges,
        ).to_dict()

    if config.mode == "trajectory" and config.trajectory_snapshot is not None:
        trajectory_context_mode: Literal["trajectory", "index", "both"]
        if config.context_mode == "both":
            trajectory_context_mode = "both"
        elif config.context_mode == "index":
            trajectory_context_mode = "index"
        else:
            trajectory_context_mode = "trajectory"
        prompt_text = build_auditor_trajectory_prompt(
            trajectory=config.trajectory_snapshot,
            continuation_notes=config.continuation_notes,
            base_prompt=base_prompt,
            methodology=meth,
            context_index=context_index,
            context_mode=trajectory_context_mode,
        )
    else:
        prompt_text = build_auditor_system_prompt(
            events=events,
            edges=edges,
            phases=phases,
            findings=findings,
            check_errors=config.check_errors,
            continuation_notes=config.continuation_notes,
            summary_threshold=config.summary_threshold,
            base_prompt=base_prompt,
            methodology=meth,
            context_index=context_index,
            context_mode=config.context_mode,
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
