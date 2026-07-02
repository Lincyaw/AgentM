"""rcabench-platform integration for the AgentM RCA scenario.

The only runtime hook this package installs at import time is a
``PROCESSER_FACTORY.register`` call — a public registration API that
rcabench-platform documents for exactly this purpose. Everything else
(trace_id forwarding from the agent to the eval DB; judge-client
``default_query``/``verify_ssl`` support for the LiteLLM-behind-Warpgate
gateway) lives upstream in rcabench-platform itself as of v0.4.44:

* ``BaseAgent.AgentResult.trace_id`` — agent fills in;
  ``BaseBenchmark._wrap_agent`` forwards to ``RolloutResult.trace_id``
  and then to ``evaluation_data.trace_id``.
* ``ModelProviderConfig.extra_query`` / ``extra_headers`` /
  ``verify_ssl`` — judge client honours all three when constructing its
  ``AsyncOpenAI`` instance.

Configure both from the eval YAML — no in-process monkey-patches.
"""

from __future__ import annotations

from loguru import logger

from rca_eval.agent import AgentMAgent


def _register_processer_aliases() -> None:
    """Make alternate dataset names route to ``RCABenchProcesser``.

    rcabench-platform's ``PROCESSER_FACTORY`` keys by ``sample.dataset``
    (case-insensitive). The ops-lite datapack on Hugging Face
    (``anon-ops/ops-lite``) stores samples under ``dataset="ops-lite"``
    but its on-disk layout is identical to RCABench, so it can reuse
    ``RCABenchProcesser``. ``PROCESSER_FACTORY.register`` is a public
    extension point — this is a registration, not a patch.
    """

    try:
        from rcabench_platform.v3.sdk.llm_eval.eval.processer import (
            PROCESSER_FACTORY,
            RCABenchProcesser,
        )
    except Exception as exc:  # noqa: BLE001 — optional dep at import time
        logger.debug("rcabench_platform processer registration skipped: {}", exc)
        return
    for alias in ("ops-lite", "opslite", "ops-lite-clean"):
        PROCESSER_FACTORY.register(alias, RCABenchProcesser)


_register_processer_aliases()


__all__ = ["AgentMAgent"]
