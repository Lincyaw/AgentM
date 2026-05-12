"""rcabench-platform integration for the AgentM RCA scenario."""

from __future__ import annotations

from contextvars import ContextVar

from agentm_rca.eval.agent import AgentMAgent


def _register_processer_aliases() -> None:
    """Make alternate dataset names route to RCABenchProcesser.

    rcabench-platform's PROCESSER_FACTORY keys by ``sample.dataset``
    (case-insensitive). The ops-lite datapack on Hugging Face
    (``anon-ops/ops-lite``) stores samples under ``dataset="ops-lite"``
    but its on-disk layout is identical to RCABench, so it can reuse
    RCABenchProcesser. Registering an alias keeps the rcabench-platform
    tree untouched.
    """

    try:
        from rcabench_platform.v3.sdk.llm_eval.eval.processer import (
            PROCESSER_FACTORY,
            RCABenchProcesser,
        )
    except Exception:  # noqa: BLE001 — optional dep at import time
        return
    for alias in ("ops-lite", "opslite"):
        PROCESSER_FACTORY.register(alias, RCABenchProcesser)


def _patch_judge_client_for_warpgate() -> None:
    """Make rcabench-platform's LLM judge speak to LiteLLM-behind-Warpgate.

    ``rcabench_platform`` instantiates ``AsyncOpenAI(base_url=..., api_key=...)``
    directly and has no concept of the per-request ``warpgate-ticket``
    query string that the local LiteLLM gateway demands. Without this
    patch the per-evidence judge calls return 401/403.

    We override the lazy ``judge_client`` property so that when
    ``WARPGATE_TICKET`` is set in the environment, the OpenAI client is
    built with ``default_query={"warpgate-ticket": ...}`` and TLS verify
    is disabled when ``OPENAI_VERIFY_SSL`` is falsy — mirroring the agent
    side in ``agentm_rca.eval.agent._build_provider``.
    """

    import os

    try:
        from rcabench_platform.v3.sdk.llm_eval.eval.processer.base_llm_processor import (
            BaseLLMJudgeProcesser,
        )
        from openai import AsyncOpenAI
    except Exception:  # noqa: BLE001
        return

    def _truthy(name: str, default: bool = True) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off", ""}

    def _patched_judge_client(self):  # type: ignore[no-untyped-def]
        if self._judge_client is not None:
            return self._judge_client
        kwargs: dict[str, object] = {
            "base_url": self._judge_provider_config.get("base_url")
            or os.environ.get("OPENAI_BASE_URL")
            or None,
            "api_key": self._judge_provider_config.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
            or "sk-warpgate",
            "max_retries": self._judge_max_retries,
        }
        ticket = os.environ.get("WARPGATE_TICKET")
        if ticket:
            kwargs["default_query"] = {"warpgate-ticket": ticket}
        if not _truthy("OPENAI_VERIFY_SSL", default=True):
            import httpx

            kwargs["http_client"] = httpx.AsyncClient(verify=False)
        self._judge_client = AsyncOpenAI(**kwargs)  # type: ignore[arg-type]
        return self._judge_client

    BaseLLMJudgeProcesser.judge_client = property(_patched_judge_client)  # type: ignore[assignment]


_current_rollout_trace_id: "ContextVar[str | None]" = ContextVar(
    "agentm_rca_rollout_trace_id", default=None
)


def _patch_rollout_trace_id_forwarding() -> None:
    """Lift ``metadata['trace_id']`` from ``AgentResult`` onto the
    ``RolloutResult`` so it lands in ``evaluation_data.trace_id``.

    Why this is non-trivial: rcabench-platform's
    ``BaseBenchmark._wrap_agent`` does not forward any trace identifier
    — ``AgentResult`` has no ``trace_id`` field, and the inner runner
    drops ``AgentResult.metadata`` entirely when it builds the
    ``RolloutResult``. To recover it without modifying
    rcabench-platform, we route the value through a
    :class:`contextvars.ContextVar` (race-safe under the ``rollout``
    loop's ``concurrency`` parallelism) and post-fix the
    ``RolloutResult`` after the inner runner returns.

    Per-call mutations of ``agent.run`` would race because the agent
    instance is shared across concurrent rollouts; ContextVar values
    are copied per task at coroutine creation, so each rollout reads
    its own trace_id back.
    """

    try:
        from rcabench_platform.v3.sdk.llm_eval.agents.base_agent import BaseAgent
        from rcabench_platform.v3.sdk.llm_eval.eval.benchmarks import base_benchmark
    except Exception:  # noqa: BLE001
        return

    from agentm_rca.eval.agent import AgentMAgent

    if getattr(AgentMAgent.run, "__agentm_traceid_patched__", False):
        return

    original_run = AgentMAgent.run

    async def _run_capturing(self, incident, data_dir, **kwargs):  # type: ignore[no-untyped-def]
        result = await original_run(self, incident, data_dir, **kwargs)
        tid = (result.metadata or {}).get("trace_id")
        if tid:
            _current_rollout_trace_id.set(tid)
        return result

    _run_capturing.__agentm_traceid_patched__ = True  # type: ignore[attr-defined]
    AgentMAgent.run = _run_capturing  # type: ignore[assignment]

    original_wrap = base_benchmark.BaseBenchmark._wrap_agent

    def _wrap_agent_with_trace_id(self, agent, on_event=None, **kwargs):  # type: ignore[no-untyped-def]
        inner = original_wrap(self, agent, on_event=on_event, **kwargs)

        async def _wrapped(sample):  # type: ignore[no-untyped-def]
            _current_rollout_trace_id.set(None)
            result = await inner(sample)
            tid = _current_rollout_trace_id.get()
            if tid:
                result.trace_id = tid
            return result

        return _wrapped

    base_benchmark.BaseBenchmark._wrap_agent = _wrap_agent_with_trace_id  # type: ignore[assignment]
    # Silence unused-import noise.
    _ = BaseAgent


_register_processer_aliases()
_patch_judge_client_for_warpgate()
_patch_rollout_trace_id_forwarding()


__all__ = ["AgentMAgent"]
