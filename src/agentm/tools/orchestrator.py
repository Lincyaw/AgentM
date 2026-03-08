"""Orchestrator tools for worker dispatch, status polling, and hypothesis management."""

from __future__ import annotations

import json
from typing import Any


# Module-level references, set by builder at startup
_task_manager: Any = None
_agent_pool: Any = None
_trajectory: Any = None


async def spawn_worker(
    task_type: str,
    instructions: str,
) -> str:
    """Launch a background worker agent. Returns task status JSON.

    Args:
        task_type: "scout", "verify", or "deep_analyze".
        instructions: Detailed instructions for the worker, including prior findings,
            which hypothesis is being tested, and what evidence to look for.
    """
    if task_type not in ("scout", "verify", "deep_analyze"):
        return json.dumps({"error": f"Invalid task_type: {task_type}. Use scout, verify, or deep_analyze."})

    subgraph = _agent_pool.get_worker(task_type)
    task_id = await _task_manager.submit(
        agent_id=f"worker-{task_type}",
        instruction=instructions,
        task_type=task_type,
        subgraph=subgraph,
        config={},
    )
    return json.dumps({
        "task_id": task_id,
        "task_type": task_type,
        "status": "running",
    })


async def wait_for_workers(timeout_seconds: float = 30) -> str:
    """Block until at least one worker completes or timeout.

    Args:
        timeout_seconds: Max wait time in seconds (default 30).
    """
    results = await _task_manager.get_all_status(wait_seconds=timeout_seconds)
    return json.dumps(results, default=str, ensure_ascii=False)


def update_hypothesis(
    id: str,
    description: str,
    status: str = "formed",
    evidence_summary: str | None = None,
    parent_id: str | None = None,
) -> str:
    """Create or update a hypothesis.

    Args:
        id: Hypothesis identifier (e.g. "H1").
        description: What this hypothesis claims.
        status: formed, investigating, confirmed, rejected, refined, or inconclusive.
        evidence_summary: Optional summary of supporting/contradicting evidence.
        parent_id: Optional parent hypothesis ID if this is a refinement.
    """
    if _trajectory is not None:
        _trajectory.record_sync(
            event_type="hypothesis_update",
            agent_path=["orchestrator"],
            data={
                "hypothesis_id": id,
                "status": status,
                "description": description,
                "evidence_summary": evidence_summary,
                "parent_id": parent_id,
            },
            hypothesis_id=id,
        )
    return f"Hypothesis {id} updated: {status} — {description}"


def remove_hypothesis(id: str) -> str:
    """Remove a hypothesis from the board.

    Args:
        id: Hypothesis identifier to remove.
    """
    if _trajectory is not None:
        _trajectory.record_sync(
            event_type="hypothesis_update",
            agent_path=["orchestrator"],
            data={
                "hypothesis_id": id,
                "status": "removed",
                "description": "",
            },
            hypothesis_id=id,
        )
    return f"Hypothesis {id} removed"
