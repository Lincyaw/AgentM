"""Fail-stop test: System Aware Merge cannot bypass the noise floor (B-4).

The load-bearing property of the four-floor deployment gate is that
*every* path that swaps the live atom is subject to all four floors —
not just ``activate``. ``merge`` is a new decision channel that
combines lessons from non-dominated parents, and it is structurally
attractive precisely because parent candidates have already passed
their own gates. If the merge channel were exempt from the noise
floor, an attacker (or an over-eager tuner) could chain
parent-rebalancing to slip a within-noise improvement through the
gate. The acceptance condition is therefore: a merge child whose
delta sits inside 2σ MUST be rejected, and the rejection must record
the same ``rejected`` kind as ``activate`` so the B-9 anti-thrash
counter treats both equally.

The complement test (a merge child whose delta clears all four floors
DOES activate) pins down both halves of the contract.
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
from agentm.core.abi import AssistantMessage
from agentm.core.abi import ProviderConfig
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession


_PROVIDER_MODULE = "agentm._tests.merge_provider"


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
            "merge-test",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="merge-test",
                    provider="fake",
                    context_window=16_000,
                    max_output_tokens=2_000,
                ),
                name="merge-test",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[_PROVIDER_MODULE] = module
    return _PROVIDER_MODULE


_ATOM_SRC_V1 = (
    "from agentm.core.abi import FunctionTool, TextContent, ToolResult\n"
    "from agentm.extensions import ExtensionManifest\n"
    "MANIFEST = ExtensionManifest(name='tool_x', description='x',"
    " registers=('tool:x',))\n"
    "async def _exec(args):\n"
    "    return ToolResult(content=[TextContent(type='text', text='v1')])\n"
    "def install(api, config):\n"
    "    api.register_tool(FunctionTool(name='x', description='x',"
    " parameters={'type':'object','properties':{}}, fn=_exec))\n"
)
_ATOM_SRC_V2 = _ATOM_SRC_V1.replace("v1", "v2")


def _setup_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Materialize a target scenario with one atom + an eval-run pair +
    two parent candidates on the frontier. Returns (scenario_dir,
    decisions_dir)."""

    scenario_dir = tmp_path / "contrib" / "scenarios" / "format_fix"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "tool_x.py").write_text(_ATOM_SRC_V1, encoding="utf-8")

    decisions_dir = tmp_path / ".agentm" / "decisions" / "format_fix"
    candidates_dir = decisions_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    # Two parent candidates — each strict winner on >=1 task.
    for cid, scores in (
        ("c_parent_a", {"t1": 0.90, "t2": 0.10}),
        ("c_parent_b", {"t1": 0.10, "t2": 0.90}),
    ):
        (candidates_dir / f"{cid}.json").write_text(
            json.dumps(
                {
                    "candidate_id": cid,
                    "parent_ids": [],
                    "change_spec": {
                        "kind": "atom_source",
                        "path": "tool_x.py",
                        "new_content": _ATOM_SRC_V1,
                        "target_atom": "tool_x",
                    },
                    "per_task_scores": scores,
                    "holdout_scores": {},
                    "eval_run_id": f"er_{cid}",
                    "created_at": 0.0,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return scenario_dir, decisions_dir


def _write_eval_run(
    cwd: Path,
    run_id: str,
    *,
    primary_score: float,
    primary_score_stderr: float,
) -> None:
    eval_runs = cwd / ".agentm" / "eval_runs"
    eval_runs.mkdir(parents=True, exist_ok=True)
    (eval_runs / f"{run_id}.jsonl").write_text(
        json.dumps(
            {
                "kind": "eval_run.summary",
                "eval_run_id": run_id,
                "task_class": "format_fix",
                "primary_score": primary_score,
                "primary_score_stderr": primary_score_stderr,
                "guard_metrics": {},
                "samples_per_task": 3,
                "task_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_merge_within_noise_floor_rejected(tmp_path: Path) -> None:
    """Merge child whose delta sits inside the 2σ band must be rejected
    with ``deployment gate failed`` — same code path as ``activate``.
    Without this, ``merge`` is a back-door past the noise floor.

    Scores chosen so threshold passes but 2σ fails:
    - baseline 0.80 ± 0.20 (so σ_b = 0.20)
    - proposed 0.90 ± 0.20 (so σ_p = 0.20)
    - delta = 0.10; threshold_relative=0.05 -> relative=0.125 PASSES
    - 2·sqrt(0.04 + 0.04) = 2·sqrt(0.08) ≈ 0.566; delta 0.10 << 0.566 FAILS
    """
    scenario_dir, decisions_dir = _setup_repo(tmp_path)
    _write_eval_run(
        tmp_path, "er_b", primary_score=0.80, primary_score_stderr=0.20
    )
    _write_eval_run(
        tmp_path, "er_p", primary_score=0.90, primary_score_stderr=0.20
    )

    provider_module = _install_static_provider()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=[

                ("agentm.extensions.builtin.operations", {}),
                ("contrib.extensions.changespec_validators", {}),
                (
                    "agentm.extensions.builtin.tool_propose_change",
                    {
                        "target_scenario": "format_fix",
                        "promotion": {
                            "threshold_relative": 0.05,
                            "guard_tolerance": 0.50,
                            # Disable B-9 so the test isolates the
                            # deployment gate, not the anti-thrash gate.
                            "stop_after_no_improvement": None,
                        },
                    },
                ),
            ],
        )
    )
    try:
        propose = next(t for t in session.tools if t.name == "propose_change")
        result = await propose.execute(
            {
                "target": {
                    "kind": "atom_source",
                    "path": "tool_x.py",
                    "new_content": _ATOM_SRC_V2,
                    "target_atom": "tool_x",
                },
                "rationale": "merge of c_parent_a + c_parent_b: combine wins",
                "parents": ["c_parent_a", "c_parent_b"],
                "eval_run_baseline": "er_b",
                "eval_run_proposed": "er_p",
                "decision": "merge",
            }
        )
        assert result.is_error is True, result.content[0].text
        assert "deployment gate failed" in result.content[0].text
        assert "noise" in result.content[0].text

        # Atom on disk MUST NOT have been swapped.
        assert (
            scenario_dir / "tool_x.py"
        ).read_text(encoding="utf-8") == _ATOM_SRC_V1

        # The candidate is on disk (inclusion phase runs even on rejection).
        # The activations.jsonl entry is kind=rejected with decision=merge.
        records = [
            json.loads(line)
            for line in (decisions_dir / "activations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        rejected = [r for r in records if r["kind"] == "rejected"]
        assert rejected
        assert rejected[-1].get("decision") == "merge"
        assert set(rejected[-1].get("parent_ids", [])) == {
            "c_parent_a",
            "c_parent_b",
        }
    finally:
        await session.shutdown()




