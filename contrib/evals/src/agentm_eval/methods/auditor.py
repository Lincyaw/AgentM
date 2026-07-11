"""LLM-harness auditor method adapter.

Single entry point for running the cognitive-audit auditor offline.
Delegates to the atom's ``build_auditor_config`` / ``build_auditor_prompt``
/ ``run_auditor_session`` — no duplicated session creation.

Usage::

    from agentm_eval.methods.auditor import run_auditor, AuditResult

    result = await run_auditor(trajectory, model="azure-gpt")
    for v in result.verdicts:
        print(v.reminder_text if v.surface_reminder else "(no surface)")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentm.core.abi import AgentMessage
from loguru import logger

from llmharness.schema import Verdict
from llmharness.state import CumulativeAuditState


@dataclass(slots=True)
class AuditResult:
    """Outcome of running the auditor over a trajectory."""

    verdicts: list[Verdict] = field(default_factory=list)
    surfaces: list[Verdict] = field(default_factory=list)
    state: CumulativeAuditState = field(default_factory=CumulativeAuditState.fresh)
    session_ids: list[str] = field(default_factory=list)


def index_to_context(idx: Any) -> dict[str, Any]:
    """Convert a TrajectoryIndex to the context dict the auditor tools expect.

    Produces ``symbols`` and ``references`` for the primary lookup path in
    ``auditor_index_tools._IndexState``, plus ``attention_hints`` mapped
    from grounding warnings.
    """
    symbols: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []

    for sym in idx.symbols.values():
        symbols.append({
            "id": sym.id,
            "name": sym.canonical_name,
            "kind": sym.kind,
            "entity_class": sym.entity_class,
            "aliases": sorted(sym.aliases),
        })
        for ref in idx.get_references(sym.id):
            references.append({
                "symbol_id": sym.id,
                "step_id": ref.step_id,
                "kind": ref.kind,
                "text": (ref.text or "")[:200],
                "grounded": ref.grounded,
            })

    attention_hints: list[dict[str, Any]] = []
    for w in idx.warnings():
        attention_hints.append({
            "kind": w.kind,
            "summary": f"{w.symbol_name}: {w.detail}",
            "symbol": w.symbol_name,
        })

    return {
        "symbols": symbols,
        "references": references,
        "attention_hints": attention_hints,
    }


async def run_auditor(
    messages: list[AgentMessage],
    *,
    model: str | None = None,
    provider: tuple[str, dict[str, Any]] | None = None,
    cwd: str = ".",
    auditor_prompt: str = "index",
    audit_interval: int | None = None,
    context_index: dict[str, Any] | None = None,
) -> AuditResult:
    """Run the auditor over a trajectory.

    Accepts typed ``AgentMessage`` objects; serialization is handled
    internally. When ``audit_interval`` is None (default), fires the
    auditor once (post-hoc). When set, fires every N turns with state
    accumulation (online-simulation mode).
    """
    from llmharness.atom import (
        build_auditor_config,
        build_auditor_prompt,
        run_auditor_session,
    )

    cumulative = CumulativeAuditState.fresh()
    result = AuditResult(state=cumulative)

    def _absorb(verdict_raw: dict[str, Any] | None, sid: str) -> None:
        result.session_ids.append(sid)
        if not isinstance(verdict_raw, dict):
            return
        verdict = Verdict.from_dict(verdict_raw)
        cumulative.absorb_auditor_verdict(verdict.to_dict())
        result.verdicts.append(verdict)
        if verdict.surface_reminder and verdict.reminder_text:
            result.surfaces.append(verdict)

    if audit_interval is None:
        config = build_auditor_config(
            cwd=cwd, model=model, provider=provider,
            auditor_prompt=auditor_prompt,
            messages=messages,
            context_index=context_index,
        )
        prompt = build_auditor_prompt()
        verdict_raw, sid = await run_auditor_session(config, prompt)
        _absorb(verdict_raw, sid)
        return result

    n_turns = len(messages)
    for turn in range(1, n_turns + 1):
        if turn % audit_interval != 0:
            continue
        try:
            config = build_auditor_config(
                cwd=cwd, model=model, provider=provider,
                auditor_prompt=auditor_prompt,
                continuation_notes=list(cumulative.last_continuation_notes),
                messages=messages[:turn],
                context_index=context_index,
            )
            prompt = build_auditor_prompt(list(cumulative.last_continuation_notes))
            verdict_raw, sid = await run_auditor_session(config, prompt)
            _absorb(verdict_raw, sid)
        except Exception as exc:
            logger.warning("auditor firing at turn {} failed: {}", turn, exc)

    return result
