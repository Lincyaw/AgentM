"""Smoke test for P4: ``baseline_fork`` with ``rca:baseline`` as control.

Why this is load-bearing: the new branch in ``_run_baseline_fork`` is
the only path that wires a control session with no audit-replay sidecar
to the offline runner. A regression that silently routes ``rca:baseline``
controls through the old ``run_offline_auditor_over_control`` path
would degrade to "empty OfflineAuditRun, no reminder, no fork" without
raising, which is the exact mode P4 set out to fix.

This test stubs ``_execute_session`` and the offline runner so the wiring
itself is exercised without spawning real child sessions.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentm_rca.eval.agent import AgentMAgent, _SessionRun
from rcabench_platform.v3.sdk.llm_eval.agents.base_agent import AgentResult
from rcabench_platform.v3.sdk.llm_eval.trajectory.schema import (
    AgentTrajectory,
    Trajectory,
    Turn,
)


def _fake_session_run(
    *, scenario: str, session_log_id: str, message_count: int = 10
) -> _SessionRun:
    final_messages = [object() for _ in range(message_count)]
    result = AgentResult(
        response="{}",
        trajectory=Trajectory(
            agent_trajectories=[
                AgentTrajectory(
                    agent_name=f"agentm:{scenario}",
                    system_prompt="",
                    turns=[Turn(messages=[])],
                )
            ]
        ),
        trace_id=session_log_id,
        metadata={"scenario": scenario},
    )
    return _SessionRun(
        result=result,
        final_messages=final_messages,
        response="{}",
        submission_dump=None,
        submit_final_report_seen=False,
        system_prompt="",
        session_id=session_log_id,
        root_session_id=session_log_id,
        session_log_id=session_log_id,
        audit_replay_path=f"/tmp/.agentm/audit_replay/{session_log_id}.jsonl",
    )


@pytest.mark.asyncio
async def test_baseline_fork_routes_baseline_control_through_offline_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """``rca:baseline`` control must:

    1. Run the control session under the ``rca:baseline`` scenario.
    2. Drive ``replay_pipeline_over_trajectory`` over its trajectory.
    3. When the synthesised auditor record surfaces a reminder, run a
       branch session with the prefix + seeded reminder.
    4. Report ``intervention_status='forked'`` plus the correct
       control/branch scenarios in the result metadata.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENTM_SKIP_DOTENV", "1")

    agent = AgentMAgent(
        scenario="rca:baseline",
        intervention_mode="baseline_fork",
        provider="anthropic",
        model="claude-test",
    )

    # Track scenarios + initial messages for each _execute_session call.
    calls: list[dict[str, Any]] = []

    async def fake_execute_session(
        self: AgentMAgent,
        *,
        incident: str | None,
        data_dir: str,
        scenario: str,
        initial_messages: list[Any] | None = None,
        seed_reminder_text: str | None = None,
        **kwargs: Any,
    ) -> _SessionRun:
        calls.append(
            {
                "scenario": scenario,
                "initial_messages": initial_messages,
                "seed_reminder_text": seed_reminder_text,
            }
        )
        # First call = control (baseline) with 10 msgs; second = branch with 6.
        if len(calls) == 1:
            return _fake_session_run(scenario=scenario, session_log_id="control-sid")
        return _fake_session_run(
            scenario=scenario, session_log_id="branch-sid", message_count=6
        )

    monkeypatch.setattr(AgentMAgent, "_execute_session", fake_execute_session)

    # Stub the offline runner so we don't need a real provider. Build a
    # synthetic auditor ReplayRecord that surfaces a reminder at turn 4.
    from llmharness.audit._runner import StepResult
    from llmharness.replay import offline_driver as _offline_driver_mod
    from llmharness.replay.offline_driver import OfflineRunResult
    from llmharness.replay.record import ReplayRecord
    from llmharness.schema import Reminder

    fake_auditor_record = ReplayRecord(
        phase="auditor",
        turn_index=4,
        root_session_id="control-sid",
        ts_ns=1,
        compose_kwargs={},
        payload={},
        provider=None,
        output={"surface_reminder": True, "reminder_text": "consider X"},
        status="ok",
    )
    fake_step = StepResult(
        fired_extractor=True,
        fired_auditor=True,
        surfaced_reminder=Reminder(text="consider X"),
        auditor_record=fake_auditor_record,
    )

    async def fake_replay_pipeline_over_trajectory(**kwargs: Any) -> OfflineRunResult:
        return OfflineRunResult(
            reminder=Reminder(text="consider X"),
            state=kwargs.get("cumulative") or None,  # type: ignore[arg-type]
            sidecar_path=kwargs.get("sidecar_path"),
            all_step_results=[fake_step],
        )

    monkeypatch.setattr(
        _offline_driver_mod,
        "replay_pipeline_over_trajectory",
        fake_replay_pipeline_over_trajectory,
    )
    # The agent imports the symbol inside a helper; patch the binding it
    # actually resolves via fresh import too.
    import agentm_rca.eval.agent as _agent_mod  # noqa: F401

    # Stub write_strict_ab_replay so we don't need real sidecar files.
    import llmharness as _llmharness_mod

    def fake_write_strict_ab_replay(**kwargs: Any) -> Any:
        out = kwargs["out_path"]
        return out

    monkeypatch.setattr(
        _llmharness_mod, "write_strict_ab_replay", fake_write_strict_ab_replay
    )

    result = await agent.run(incident="boom", data_dir=str(tmp_path / "sample-1"))

    # 1. Control was run with rca:baseline.
    assert calls[0]["scenario"] == "rca:baseline"
    assert calls[0]["initial_messages"] is None

    # 2. Branch was run with the (default) branch scenario, with prefix
    #    messages and the seeded reminder text.
    assert len(calls) == 2
    assert calls[1]["scenario"] == "rca:baseline"  # branch default == control
    assert calls[1]["seed_reminder_text"] == "consider X"
    assert isinstance(calls[1]["initial_messages"], list)
    # turn_index=4 → prefix length 5 (5 messages, [:5]).
    assert len(calls[1]["initial_messages"]) == 5

    # 3. Metadata reflects the fork.
    assert result.metadata is not None
    assert result.metadata["intervention_mode"] == "baseline_fork"
    assert result.metadata["control_scenario"] == "rca:baseline"
    assert result.metadata["intervention_status"] == "forked"
