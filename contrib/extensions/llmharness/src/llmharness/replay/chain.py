"""Bulk replay across every record in a sidecar.

Single-phase replay (``llmharness-replay {extractor,auditor} --turn N``)
is the right tool when you have one suspicious turn. Chain replay is
the right tool when you want to ask "across all 325 auditor firings in
this run, what changes if I swap the prompt / model?" — it iterates in
turn order and yields one :class:`ChainResult` per matched record.

Chain does **not** re-thread fresh extractor outputs back into the
auditor's input graph; each phase replays against its own recorded
inputs. That keeps replay hermetic — one bisection variable at a time
— which is what bug bisection actually needs. If you want to study
"how does a re-extracted graph change auditor verdicts?" run a fresh
``enable_auditor: false`` capture instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .engine import PhaseResult
from .record import ReplayRecord, iter_records
from .runner import replay_auditor_record, replay_extractor_record

PhaseFilter = Literal["extractor", "auditor", "both"]


@dataclass
class ChainResult:
    """One record's replay outcome paired with the recorded baseline."""

    record: ReplayRecord
    result: PhaseResult


async def chain_replay(
    records_path: Path,
    *,
    cwd: str,
    phase: PhaseFilter = "both",
    provider_override: tuple[str, dict[str, Any]] | None = None,
    prompt_override_extractor: str | None = None,
    prompt_override_auditor: str | None = None,
    on_progress: Callable[[int, ChainResult], None] | None = None,
) -> AsyncIterator[ChainResult]:
    """Iterate records in file order; replay matching phases sequentially.

    Sequential by design — provider rate limits make parallel replay a
    footgun, and the typical use case (debug bisection) doesn't need it.
    """
    idx = 0
    for record in iter_records(records_path):
        if phase != "both" and record.phase != phase:
            continue
        if record.phase == "extractor":
            result = await replay_extractor_record(
                record,
                cwd=cwd,
                provider_override=provider_override,
                prompt_override=prompt_override_extractor,
            )
        else:
            result = await replay_auditor_record(
                record,
                cwd=cwd,
                provider_override=provider_override,
                prompt_override=prompt_override_auditor,
            )
        cr = ChainResult(record=record, result=result)
        if on_progress is not None:
            on_progress(idx, cr)
        idx += 1
        yield cr


def chain_replay_sync(
    records_path: Path,
    *,
    cwd: str,
    phase: PhaseFilter = "both",
    provider_override: tuple[str, dict[str, Any]] | None = None,
    prompt_override_extractor: str | None = None,
    prompt_override_auditor: str | None = None,
    on_progress: Callable[[int, ChainResult], None] | None = None,
) -> list[ChainResult]:
    """Eager sync wrapper. Convenient for CLI / quick scripts."""

    async def _collect() -> list[ChainResult]:
        out: list[ChainResult] = []
        async for cr in chain_replay(
            records_path,
            cwd=cwd,
            phase=phase,
            provider_override=provider_override,
            prompt_override_extractor=prompt_override_extractor,
            prompt_override_auditor=prompt_override_auditor,
            on_progress=on_progress,
        ):
            out.append(cr)
        return out

    return asyncio.run(_collect())
