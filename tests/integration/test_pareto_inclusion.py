"""Fail-stop test: Pareto inclusion correctness (B-1).

A candidate that is the strict winner on >=1 task must remain in the
candidate pool even when another candidate scores higher in aggregate.

Why this is load-bearing: GEPA's headline 35x-rollout efficiency hinges
on diversity preservation in the candidate pool. If Pareto inclusion
collapses to "best aggregate wins" (the Phase-1 hill-climbing behavior),
the search degenerates to greedy and loses the property that motivates
moving to Phase 2 in the first place. A regression here silently turns
an illumination algorithm into a (worse-performing) hill climb.

The test exercises ``tool_query_candidates`` directly — driving via the
agent loop is unnecessary because the inclusion logic lives in
``tool_propose_change._prune_dominated_candidates`` /
``tool_query_candidates._compute_win_tasks``, which share the same
strict-argmax-by-task definition. We validate by constructing a pool
on disk and asserting both sides of the contract: the niche winner is
on the frontier, and the pool size > 1.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
)
from agentm.core.abi.messages import AssistantMessage
from agentm.core.abi.extension import ProviderConfig
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession


_PROVIDER_MODULE = "agentm._tests.pareto_provider"


def _install_static_provider() -> str:
    if _PROVIDER_MODULE in sys.modules:
        return _PROVIDER_MODULE
    module = types.ModuleType(_PROVIDER_MODULE)

    async def _stream(
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del messages, model, tools, system, signal, thinking
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="ok")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "pareto-test",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="pareto-test",
                    provider="fake",
                    context_window=16_000,
                    max_output_tokens=2_000,
                ),
                name="pareto-test",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[_PROVIDER_MODULE] = module
    return _PROVIDER_MODULE


def _write_candidate(
    candidates_dir: Path,
    *,
    candidate_id: str,
    per_task_scores: dict[str, float],
    parent_id: str | None = None,
) -> None:
    candidates_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "change_spec": {
            "kind": "atom_source",
            "path": "tool_x.py",
            "new_content": "# stub",
            "target_atom": "tool_x",
        },
        "per_task_scores": per_task_scores,
        "holdout_scores": {},
        "eval_run_id": f"er_{candidate_id}",
        "created_at": 0.0,
    }
    (candidates_dir / f"{candidate_id}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_pareto_inclusion_retains_niche_winner(tmp_path: Path) -> None:
    """Build a pool of 2 candidates: A scores 0.80 on tasks 1-4 and 0.10
    on task 5; B scores 0.30 on tasks 1-4 and 0.95 on task 5. A wins
    aggregate (0.66 vs 0.43) but B is the unique winner on task 5. The
    Pareto frontier must contain BOTH — collapsing to A would lose B's
    niche-improvement contribution to future search.
    """
    scenario = "pareto_demo"
    candidates_dir = (
        tmp_path / ".agentm" / "decisions" / scenario / "candidates"
    )
    _write_candidate(
        candidates_dir,
        candidate_id="c_aggregate_strong",
        per_task_scores={
            "t1": 0.80,
            "t2": 0.80,
            "t3": 0.80,
            "t4": 0.80,
            "t5": 0.10,
        },
    )
    _write_candidate(
        candidates_dir,
        candidate_id="c_niche_winner",
        per_task_scores={
            "t1": 0.30,
            "t2": 0.30,
            "t3": 0.30,
            "t4": 0.30,
            "t5": 0.95,
        },
    )

    provider_module = _install_static_provider()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=[

                ("agentm.extensions.builtin.operations_local", {}),
                (
                    "agentm.extensions.builtin.tool_query_candidates",
                    {"default_scenario": scenario},
                ),
            ],
        )
    )
    try:
        tool = next(t for t in session.tools if t.name == "query_candidates")
        # First call.
        result = await tool.execute({})
        assert result.is_error is False, result.content[0].text
        payload = json.loads(result.content[0].text)
        ids = {entry["candidate_id"] for entry in payload["frontier"]}
        # Both candidates are unique winners on >=1 task -> both retained.
        assert ids == {"c_aggregate_strong", "c_niche_winner"}, payload
        # Pool size > 1 (the explicit acceptance condition from B-1).
        assert len(payload["frontier"]) > 1
        # Niche winner is on the frontier and claims task 5.
        niche = next(
            e for e in payload["frontier"] if e["candidate_id"] == "c_niche_winner"
        )
        assert "t5" in niche["win_tasks"]
        # Aggregate winner is on the frontier and claims tasks 1-4.
        aggregate = next(
            e for e in payload["frontier"]
            if e["candidate_id"] == "c_aggregate_strong"
        )
        assert set(aggregate["win_tasks"]) == {"t1", "t2", "t3", "t4"}

        # Idempotence: re-running yields the same result (cache regenerable).
        result2 = await tool.execute({})
        assert result2.is_error is False
        assert json.loads(result2.content[0].text) == payload
    finally:
        await session.shutdown()


