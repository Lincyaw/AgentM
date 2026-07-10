"""rcabench-platform integration for the AgentM RCA scenario.

Registers the FPG-based processer for ops-lite datasets so the judge
uses ``fpg.compare_model_to_ground_truth`` instead of rcabench-platform's
``AgentRCAOutput`` schema (which is incompatible with fpg ModelRCAOutput).
"""

from __future__ import annotations

from loguru import logger

from agentm_eval.benchmarks.rca.agent import AgentMAgent


def _register_processers() -> None:
    try:
        from rcabench_platform.v3.sdk.llm_eval.eval.processer import (
            PROCESSER_FACTORY,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("rcabench_platform processer registration skipped: {}", exc)
        return

    from agentm_eval.benchmarks.rca.fpg_processer import FpgProcesser

    for alias in ("ops-lite", "opslite", "ops-lite-clean", "RCABench", "rcabench"):
        PROCESSER_FACTORY.register(alias, FpgProcesser)


_register_processers()


__all__ = ["AgentMAgent"]
