"""System prompt assembly for the auditor child session.

The framing text lives in markdown files under ``prompts/`` (sibling to
this module). Pick a variant by name (e.g. ``"minimal"``) or by an
absolute path. The dynamic per-firing data is appended on top of the
chosen framing inside :func:`build_auditor_system_prompt`.

Available named variants:

* ``minimal`` (default) — pairs with the ``minimal`` profile (only
  ``submit_verdict``). No drill-down references.

Drop in a new variant by adding ``prompts/auditor_<name>.md``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from llmharness.schema import Edge, Event, Finding, Phase

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

DEFAULT_PROMPT_NAME = "minimal"
TRAJECTORY_PROMPT_NAME = "trajectory"


def _resolve(name_or_path: str) -> Path:
    candidate = name_or_path.strip()
    if not candidate:
        raise ValueError("empty prompt spec for auditor")
    if "/" in candidate or "\\" in candidate:
        path = Path(candidate).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"prompt file not found: {path}")
        return path
    for fname in (f"{candidate}.md", f"auditor_{candidate}.md"):
        path = _PROMPTS_DIR / fname
        if path.is_file():
            return path
    available = sorted(p.name for p in _PROMPTS_DIR.glob("*.md"))
    raise FileNotFoundError(f"unknown auditor prompt {candidate!r}; available: {available}")


@lru_cache(maxsize=64)
def _read(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


def load_auditor_prompt(name_or_path: str = DEFAULT_PROMPT_NAME) -> str:
    """Load the auditor framing text for the given variant."""
    return _read(str(_resolve(name_or_path)))


def _degrade_event(ev_dict: dict[str, object]) -> dict[str, object]:
    return {
        "id": ev_dict.get("id"),
        "kind": ev_dict.get("kind"),
        "summary": ev_dict.get("summary"),
        "source_turns": ev_dict.get("source_turns", []),
    }


def _degrade_edge(ed_dict: dict[str, object]) -> dict[str, object]:
    return {
        "src": ed_dict.get("src"),
        "dst": ed_dict.get("dst"),
        "kind": ed_dict.get("kind"),
        "reason": ed_dict.get("reason"),
    }


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
) -> str:
    """Assemble the auditor system prompt for one firing."""
    framing = base_prompt if base_prompt is not None else load_auditor_prompt(DEFAULT_PROMPT_NAME)
    degraded = len(events) > summary_threshold

    if degraded:
        events_payload = [_degrade_event(ev.to_dict()) for ev in events]
        edges_payload = [_degrade_edge(ed.to_dict()) for ed in edges]
    else:
        events_payload = [ev.to_dict() for ev in events]
        edges_payload = [ed.to_dict() for ed in edges]

    findings_payload = [f.to_dict() for f in findings]

    sections: list[str] = [framing.rstrip(), ""]

    if phases:
        sections.append("## PHASES (primary view — merged basic blocks)")
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

    sections.append("## GRAPH")
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
) -> str:
    """Assemble the auditor system prompt for a trajectory-mode firing."""
    framing = (
        base_prompt
        if base_prompt is not None
        else load_auditor_prompt(TRAJECTORY_PROMPT_NAME)
    )

    sections: list[str] = [framing.rstrip(), ""]

    sections.append("## TRAJECTORY")
    sections.append(
        f"conversation turns ({len(trajectory)} total):"
    )
    sections.append(json.dumps(trajectory, ensure_ascii=False))
    sections.append("")

    sections.append("## CONTINUATION_NOTES (from your prior firing)")
    sections.append(json.dumps(list(continuation_notes), ensure_ascii=False))
    sections.append("")

    return "\n".join(sections)


__all__ = [
    "DEFAULT_PROMPT_NAME",
    "TRAJECTORY_PROMPT_NAME",
    "build_auditor_system_prompt",
    "build_auditor_trajectory_prompt",
    "load_auditor_prompt",
]
