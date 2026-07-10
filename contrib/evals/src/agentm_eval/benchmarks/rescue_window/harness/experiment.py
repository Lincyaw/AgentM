"""Experiment orchestrator: corpus -> sample -> treatments -> rollout -> store.

This drives the prefix x treatment x K matrix and appends ``EvalUnit`` rows. It
is the only place the rollout modules are composed; phases (E0..E4) are just
different ``SamplingPolicy`` / ``TreatmentFactory`` / K configurations of this
one loop (DESIGN §5).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from agentm.core.abi.session_store import SessionStore

from ..model import EvalUnit, EvalUnitStore
from .adapter import ScenarioAdapter
from .corpus import TrajectoryRef, load_trajectory_messages
from .runner import RolloutConfig, run_intervention_rollout
from .sampler import PrefixSampler
from .treatments import TreatmentFactory

ProgressCallback = Callable[[EvalUnit], None]


async def run_landscape(
    corpus: list[TrajectoryRef],
    *,
    sampler: PrefixSampler,
    factory: TreatmentFactory,
    store: EvalUnitStore,
    session_store: SessionStore,
    adapter: ScenarioAdapter,
    config: RolloutConfig | None = None,
    k: int = 3,
    concurrency: int = 1,
    skip_existing: bool = True,
    provider_override: tuple[str, dict] | None = None,
    on_result: ProgressCallback | None = None,
) -> int:
    """Run the oracle landscape (E1) over a corpus. Returns rows written."""

    run_config = config or RolloutConfig()
    existing = store.existing_cells() if skip_existing else set()
    sem = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    written = 0

    async def _emit(unit: EvalUnit) -> None:
        nonlocal written
        async with write_lock:
            store.append([unit])
            written += 1
            if on_result is not None:
                on_result(unit)

    async def _rollout(ref: TrajectoryRef, prefix, treatment, seed: int) -> None:  # type: ignore[no-untyped-def]
        async with sem:
            unit = await run_intervention_rollout(
                ref=ref,
                prefix=prefix,
                treatment=treatment,
                seed=seed,
                store=session_store,
                adapter=adapter,
                config=run_config,
                provider_override=provider_override,
            )
        await _emit(unit)

    tasks: list[asyncio.Task[None]] = []
    for ref in corpus:
        messages = load_trajectory_messages(ref, store=session_store)
        prefixes = sampler.sample(ref, messages)
        if not prefixes:
            continue
        gt = adapter.ground_truth(ref)

        for prefix in prefixes:
            treatments = await factory.build(prefix, messages, gt)
            for treatment in treatments:
                for seed in range(k):
                    if (prefix.prefix_id, treatment.treatment_id, seed) in existing:
                        continue
                    tasks.append(
                        asyncio.create_task(_rollout(ref, prefix, treatment, seed))
                    )
    if tasks:
        await asyncio.gather(*tasks)
    return written
