"""AgentM wrapper implementing rcabench-platform's :class:`BaseAgent`.

Delegates to :func:`agentm.cli.run.run_investigation_headless` for the
actual RCA investigation.

Usage::

    from agentm.agents.eval_agent import AgentMAgent

    agent = AgentMAgent(
        scenario_dir="config/scenarios/rca_hypothesis",
        config_path="config/system.yaml",
    )
    ok, fail = await benchmark.rollout(agent)

Registration::

    from rcabench_platform.v3.sdk.llm_eval.agents import AGENT_REGISTRY
    from agentm.agents.eval_agent import AgentMAgent

    AGENT_REGISTRY.register(AgentMAgent)
"""

from __future__ import annotations

from typing import Any

from rcabench_platform.v3.sdk.llm_eval.agents.base_agent import AgentResult, BaseAgent, RunContext
from rcabench_platform.v3.sdk.llm_eval.trajectory.schema import Trajectory


class AgentMAgent(BaseAgent):
    """Hypothesis-driven multi-agent RCA agent (AgentM).

    Args:
        scenario_dir: Path to the AgentM scenario directory
            (e.g. ``config/scenarios/rca_hypothesis``).
        config_path: Path to the AgentM system config YAML
            (e.g. ``config/system.yaml``).
        max_steps: Maximum orchestrator steps per sample.
        timeout: Per-sample timeout in seconds (``0`` = no limit).
    """

    def __init__(
        self,
        scenario_dir: str = "config/scenarios/rca_hypothesis",
        config_path: str = "config/system.yaml",
        max_steps: int = 100,
        timeout: float = 0,
    ) -> None:
        self._scenario_dir = scenario_dir
        self._config_path = config_path
        self._max_steps = max_steps
        self._timeout = timeout

    @staticmethod
    def name() -> str:
        return "agentm"

    def version(self) -> str | None:
        try:
            from importlib.metadata import version

            return version("agentm")
        except Exception:
            return None

    async def run(
        self,
        incident: str,
        data_dir: str,
        **kwargs: Any,
    ) -> AgentResult:
        from agentm.cli.run import run_investigation_headless

        ctx: RunContext | None = kwargs.get("ctx")
        max_steps = kwargs.get("max_steps", self._max_steps)
        timeout = kwargs.get("timeout", self._timeout)

        def _on_headless_start(run_id: str, traj_path: str | None) -> None:
            if ctx:
                ctx.emit({"type": "running", "run_id": run_id})
                if traj_path:
                    ctx.emit({"type": "trajectory_update", "path": traj_path})

        response_json, trajectory_json, run_id, traj_file_path = await run_investigation_headless(
            data_dir=data_dir,
            incident=incident,
            scenario_dir=self._scenario_dir,
            config_path=self._config_path,
            max_steps=max_steps,
            timeout=timeout,
            on_start=_on_headless_start,
        )

        trajectory = None
        if trajectory_json:
            try:
                trajectory = Trajectory.from_json(trajectory_json)
            except Exception:
                pass

        return AgentResult(
            response=response_json or "",
            trajectory=trajectory,
            metadata={
                "run_id": run_id,
                "trajectory_file": traj_file_path,
            },
        )
