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

import json
import logging
from typing import Any

from rcabench_platform.v3.sdk.llm_eval.agents.base_agent import (
    AgentResult,
    BaseAgent,
    RunContext,
)
from rcabench_platform.v3.sdk.llm_eval.trajectory.schema import Trajectory


logger = logging.getLogger(__name__)

_EMPTY_CAUSAL_GRAPH_RESPONSE = json.dumps(
    {
        "nodes": [],
        "edges": [],
        "root_causes": [],
        "component_to_service": {},
    },
    ensure_ascii=False,
)


class AgentMAgent(BaseAgent):
    """Hypothesis-driven multi-agent RCA agent (AgentM).

    Args:
        scenario_dir: Path to the AgentM scenario directory
            (e.g. ``config/scenarios/rca_hypothesis``).
        config_path: Path to the AgentM system config YAML
            (e.g. ``config/system.yaml``).
        max_steps: Maximum orchestrator steps per sample.
        timeout: Per-sample timeout in seconds (``0`` = no limit).
        exp_id: Experiment identifier from the eval framework.
    """

    def __init__(
        self,
        scenario_dir: str = "config/scenarios/rca_hypothesis",
        config_path: str = "config/system.yaml",
        max_steps: int = 100,
        timeout: float = 0,
        exp_id: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self._scenario_dir = scenario_dir
        self._config_path = config_path
        self._max_steps = max_steps
        self._timeout = timeout
        self._exp_id = exp_id

        # Resolve model name: env override > scenario config
        import os
        from pathlib import Path

        self._model_name: str | None = os.environ.get("AGENTM_ORCHESTRATOR_MODEL")
        if not self._model_name:
            try:
                from agentm.config.loader import load_scenario_config

                scenario_yaml = Path(scenario_dir) / "scenario.yaml"
                sc = load_scenario_config(scenario_yaml)
                self._model_name = sc.orchestrator.model
            except Exception:
                pass

    def model_name(self) -> str | None:
        """Orchestrator model name (for eval DB tracking)."""
        return self._model_name

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

        try:
            (
                response_json,
                trajectory_json,
                run_id,
                traj_file_path,
            ) = await run_investigation_headless(
                data_dir=data_dir,
                incident=incident,
                scenario_dir=self._scenario_dir,
                config_path=self._config_path,
                max_steps=max_steps,
                timeout=timeout,
                on_start=_on_headless_start,
                exp_id=self._exp_id,
            )
        except Exception as exc:
            logger.exception(
                "headless run failed; fallback to empty CausalGraph "
                "(data_dir=%s, scenario_dir=%s, config_path=%s, max_steps=%s, timeout=%s, exp_id=%s)",
                data_dir,
                self._scenario_dir,
                self._config_path,
                max_steps,
                timeout,
                self._exp_id,
            )
            return AgentResult(
                response=_EMPTY_CAUSAL_GRAPH_RESPONSE,
                trajectory=None,
                metadata={
                    "run_id": None,
                    "exp_id": self._exp_id,
                    "trajectory_file": None,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "fallback_reason": "exception",
                },
            )

        metadata: dict[str, Any] = {
            "run_id": run_id,
            "exp_id": self._exp_id,
            "trajectory_file": traj_file_path,
        }

        response = response_json
        if not response:
            logger.warning(
                "headless returned empty structured response; fallback to empty CausalGraph "
                "(run_id=%s, data_dir=%s)",
                run_id,
                data_dir,
            )
            response = _EMPTY_CAUSAL_GRAPH_RESPONSE
            metadata["fallback_reason"] = "empty_response"

        trajectory = None
        if trajectory_json:
            try:
                trajectory = Trajectory.from_json(trajectory_json)
            except Exception:
                logger.warning(
                    "failed to parse trajectory json (run_id=%s, trajectory_file=%s)",
                    run_id,
                    traj_file_path,
                    exc_info=True,
                )

        return AgentResult(
            response=response,
            trajectory=trajectory,
            metadata=metadata,
        )
