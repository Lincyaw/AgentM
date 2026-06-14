"""``rcabench-platform`` BaseAgent adapter for the AgentM RCA scenario.

Discovered by ``rcabench-platform``'s ``llm_eval.agents`` entry point and
invoked via ``rca llm-eval run --agent agentm``. Bridges the
``incident + data_dir -> AgentRCAOutput JSON`` contract to an in-process
``AgentSession`` running the local ``rca`` scenario.

Two pieces of glue do the work:

* The orchestrator's ``submit_final_report`` tool in
  :mod:`atoms.default.finalize` validates against the rcabench-platform
  ``AgentRCAOutput`` contract and emits the model's
  ``model_dump_json(by_alias=True)`` as its tool result. The adapter
  subscribes to ``tool_result`` on the session bus and parses that
  authoritative payload via :meth:`AgentRCAOutput.parse_str` — never the
  unvalidated ``tool_call`` args. A failed validation re-runs the model
  without polluting captured state.
* The session's final message list is walked once after ``prompt`` returns
  and translated into a ``rcabench-platform`` :class:`Trajectory`. The
  system prompt is captured separately from the first
  ``before_send_to_llm`` event because :meth:`AgentSession.prompt` does not
  return it.

Boundary: this module is a host-side driver (not a atom — no
``MANIFEST`` / ``install`` pair, never named in a scenario manifest), so
the ``agentm.core.runtime.*`` imports inside :func:`run_one` are
intentional. If this file is ever promoted to an atom, route session
construction through ``ExtensionAPI`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

from rcabench_platform.v3.sdk.llm_eval.agents.base_agent import (
    AgentResult,
    BaseAgent,
    RunContext,
)
from rcabench_platform.v3.sdk.llm_eval.trajectory.schema import (
    AgentTrajectory,
    Message,
    ToolCall,
    Trajectory,
    Turn,
)
from loguru import logger

_DEFAULT_MODEL = "claude-sonnet-4-6"
# RCA investigations dispatch workers, poll, and run many SQL queries; the
# kernel's stock 32-turn cap exhausts before ``submit_final_report`` is
# reached. Bump the default and let the framework's ``--max-steps`` (or
# ``--ak max_turns=N``) override.
_DEFAULT_MAX_TURNS = 128
# Empty AgentRCAOutput-shaped fallback when the orchestrator never reaches
# ``submit_final_report``. Keeps the wire shape consistent with successful
# runs so the platform's parsers don't choke.
_EMPTY_AGENT_RCA_OUTPUT: dict[str, list[Any]] = {
    "root_causes": [],
    "propagation": [],
}

# Module path of the llmharness cognitive-audit adapter. A scenario that
# mounts this module is treated as a "harness scenario" by the eval
# driver — it auto-wires the distill-binding atom so the offline
# distillation pipeline can join replay records to ground truth.
# Detection is by manifest composition (see :func:`_scenario_mounts_harness`),
# not by scenario name — string-sniffing the scenario id silently breaks
# when new variants are added.
_HARNESS_ADAPTER_MODULE = "llmharness.atom"


def _provider_name_from_base_url(base_url: str) -> str:
    """Derive a stable provider registry slug from a base URL.

    PR #95 requires every non-canonical OpenAI-compatible endpoint to
    register under a unique name. The host segment is unique per gateway
    in practice (LiteLLM, Doubao Ark, DeepSeek, etc.) and stays stable
    across runs, which keeps observability traces comparable.
    """

    from urllib.parse import urlparse

    host = urlparse(base_url).hostname or "openai-compat"
    return host.replace(".", "-")


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"false", "0", "no", "off", ""}


def _try_resolve_profile(model: str) -> tuple[str, dict[str, Any]] | None:
    """Attempt to resolve ``model`` as a ``~/.agentm/config.toml`` profile."""
    try:
        from agentm.ai import DEFAULT_PROVIDER_REGISTRY
        from agentm.core.lib import resolve_model_profile

        profile = resolve_model_profile(model)
        if profile is None:
            return None
        return DEFAULT_PROVIDER_REGISTRY.build(
            profile.provider, profile.to_build_config()
        )
    except Exception:  # noqa: BLE001
        return None


def _build_provider(provider: str, model: str) -> tuple[str, dict[str, Any]]:
    """Same env-var convention as ``agentm.cli._build_provider``.

    Duplicated here (rather than imported) because the eval adapter is
    discovered via ``llm_eval.agents`` entry point and must not assume
    the CLI module has been imported.
    """
    if provider == "anthropic":
        cfg: dict[str, Any] = {"model": model}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            cfg["base_url"] = base_url
        return ("agentm.extensions.builtin.llm_anthropic", cfg)

    if provider == "openai":
        cfg = {"model": model}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            cfg["base_url"] = base_url
            # A non-canonical ``base_url`` requires a distinct ``name`` so two
            # custom endpoints can't clobber each other under the bare
            # ``openai`` registry slot. Honor an explicit override via
            # ``AGENTM_PROVIDER_NAME`` and otherwise derive a stable slug from
            # the base host so eval rollouts don't crash.
            cfg["name"] = os.environ.get(
                "AGENTM_PROVIDER_NAME"
            ) or _provider_name_from_base_url(base_url)
        ticket = os.environ.get("WARPGATE_TICKET")
        if ticket:
            cfg["default_query"] = {"warpgate-ticket": ticket}
        if not _env_bool("OPENAI_VERIFY_SSL", default=True):
            cfg["verify_ssl"] = False
        return ("agentm.extensions.builtin.llm_openai", cfg)

    raise ValueError(f"unknown provider {provider!r}; expected 'anthropic' or 'openai'")


def _coerce_max_turns(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        result = int(value)
    except (TypeError, ValueError):
        return fallback
    return result if result > 0 else fallback


def _coerce_max_interventions(value: Any, fallback: int) -> int:
    """Coerce ``--ak max_interventions=...`` to a non-negative int.

    Distinct from :func:`_coerce_max_turns` (which floors at 1) because
    ``max_interventions=0`` is a legitimate setting — it disables every
    branch and runs only the control segment, which is useful for
    A/B-baseline experiments that want chained-fork metadata without
    paying for branch rollouts.
    """
    if value is None:
        return fallback
    try:
        result = int(value)
    except (TypeError, ValueError):
        return fallback
    return result if result >= 0 else fallback


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


@dataclass(frozen=True)
class _SessionRun:
    result: AgentResult
    final_messages: list[Any]
    response: str
    submission_dump: Any
    submit_final_report_seen: bool
    system_prompt: str
    session_id: str
    root_session_id: str
    session_log_id: str
    audit_replay_path: str


class AgentMAgent(BaseAgent):
    """Run an RCA investigation through an in-process AgentM session."""

    def __init__(
        self,
        *,
        scenario: str = "rca",
        model: str | None = None,
        provider: str | None = None,
        provider_tuple: tuple[str, dict[str, Any]] | None = None,
        exp_id: str | None = None,
        max_turns: Any = None,
        chained_fork: Any = None,
        max_interventions: Any = None,
        **_extra: Any,
    ) -> None:
        # ``rcabench-platform`` passes ``exp_id`` plus any ``--ak key=value``
        # kwargs to the agent ctor. We accept and ignore the unknowns so a
        # stale flag in someone's eval YAML does not crash the rollout.
        # ``--ak`` values arrive as strings, so coerce explicitly.
        self._scenario = scenario
        self._model = model or os.environ.get("AGENTM_MODEL", _DEFAULT_MODEL)
        self._provider = provider or os.environ.get("AGENTM_PROVIDER") or "anthropic"
        self._provider_tuple = provider_tuple
        if provider_tuple is not None:
            self._model = provider_tuple[1].get("model", self._model)
        elif provider_tuple is None and model:
            self._provider_tuple = _try_resolve_profile(model)
            if self._provider_tuple is not None:
                self._model = self._provider_tuple[1].get("model", self._model)
        self._exp_id = exp_id
        self._max_turns = _coerce_max_turns(max_turns, _DEFAULT_MAX_TURNS)
        self._chained_fork = _coerce_bool(chained_fork, default=False)
        self._max_interventions = _coerce_max_interventions(max_interventions, 10)

        # Stay lenient on unknown kwargs (stale eval YAMLs survive), but
        # surface them at WARNING so a hand-edited config still pointing
        # at pre-refactor flag names (intervention_mode / fork_audit /
        # fork_policy / control_scenario / branch_scenario) doesn't
        # silently degrade to a vanilla control session.
        if _extra:
            logger.warning(
                f"AgentMAgent: ignoring unknown kwargs {sorted(_extra.keys())}. Pre-refactor names like intervention_mode / fork_policy / fork_audit / control_scenario / branch_scenario are removed; use chained_fork=true and max_interventions=N instead."
            )

    @staticmethod
    def name() -> str:
        return "agentm"

    def model_name(self) -> str | None:
        return self._model

    async def run(
        self,
        incident: str,
        data_dir: str,
        **kwargs: Any,
    ) -> AgentResult:
        if self._chained_fork:
            return await self._run_chained_fork(
                incident=incident,
                data_dir=data_dir,
                **kwargs,
            )
        return await self._run_single_session(
            incident=incident,
            data_dir=data_dir,
            scenario=self._scenario,
            **kwargs,
        )

    async def _run_single_session(
        self,
        *,
        incident: str | None,
        data_dir: str,
        scenario: str,
        initial_messages: list[Any] | None = None,
        seed_reminder_text: str | None = None,
        metadata_extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        run = await self._execute_session(
            incident=incident,
            data_dir=data_dir,
            scenario=scenario,
            initial_messages=initial_messages,
            seed_reminder_text=seed_reminder_text,
            **kwargs,
        )
        if not metadata_extra:
            return run.result
        metadata = dict(run.result.metadata or {})
        metadata.update(metadata_extra)
        return AgentResult(
            response=run.result.response,
            trajectory=run.result.trajectory,
            trace_id=run.result.trace_id,
            metadata=metadata,
        )

    async def _run_chained_fork(
        self,
        *,
        incident: str,
        data_dir: str,
        **kwargs: Any,
    ) -> AgentResult:
        from llmharness import (
            AuditorSettings,
            ExtractorSettings,
            SessionPayload,
            run_fork_tree_experiment,
        )

        # The user-facing ``--ak chained_fork=true`` flag now drives the
        # fork-tree engine; the linear chain is the degenerate
        # ``max_surfaces_per_node=1`` policy. We run the full tree.
        #
        # Side-table keyed by session_log_id so we can recover the full
        # _SessionRun (and its AgentResult / metadata) for each node after
        # the experiment finishes. _SessionRun structurally matches
        # SessionPayload (both have session_log_id + final_messages), so
        # the factory returns it directly.
        session_runs: dict[str, _SessionRun] = {}

        async def factory(
            *,
            initial_messages: list[Any] | None,
            seed_reminder_text: str | None,
        ) -> SessionPayload:
            run = await self._execute_session(
                incident=incident if initial_messages is None else None,
                data_dir=data_dir,
                scenario=self._scenario,
                initial_messages=initial_messages,
                seed_reminder_text=seed_reminder_text,
                **kwargs,
            )
            session_runs[run.session_log_id] = run
            return run  # type: ignore[return-value]  # structural match on Protocol

        experiment = await run_fork_tree_experiment(
            session_factory=factory,
            cwd=os.getcwd(),
            provider=_build_provider(self._provider, self._model),
            extractor_settings=ExtractorSettings.default(),
            auditor_settings=AuditorSettings.default(),
            extractor_interval=5,
            audit_interval=5,
            # ``max_interventions`` historically capped the linear chain
            # length; map it onto the tree's depth guard so an existing
            # ``--ak max_interventions=N`` keeps bounding how deep
            # interventions stack.
            max_depth=self._max_interventions,
        )

        # The reported response is the ROOT/control node's submission: the
        # baseline answer (no intervention) is what gets judged, so the
        # control-vs-intervention comparison stays honest. Every node's
        # submission + intervention path lives in metadata so a
        # control-vs-leaf comparison is recoverable downstream.
        root_run = session_runs[experiment.root.backbone_session_id]
        metadata = dict(root_run.result.metadata or {})
        # Tree topology comes straight from the experiment header (matches
        # the ``<root_sid>.chained.jsonl`` first-line bundle so
        # rcabench-platform and downstream tools see the same shape).
        # Augment with presentational fields the sidecar doesn't carry:
        # ``base_scenario``, ``forktree_replay_path`` (string path), and
        # per-node ``run`` summaries from the rca eval driver.
        tree_meta: dict[str, Any] = dict(experiment.header)
        tree_meta["base_scenario"] = self._scenario
        tree_meta["forktree_replay_path"] = (
            str(experiment.forktree_replay_path)
            if experiment.forktree_replay_path is not None
            else None
        )
        tree_meta["nodes"] = [
            {
                **node_header,
                "run": _run_metadata(session_runs[node_header["backbone_session_id"]]),
            }
            for node_header in experiment.header["nodes"]
        ]
        metadata["intervention_mode"] = "fork_tree"
        # The ``chained_fork`` key name is kept for backward compat with
        # existing dashboards / parsers, but holds a fork-tree shape
        # (``nodes`` list with parent links + paths), not a linear
        # ``segments`` list. Downstream consumers should switch on
        # ``intervention_mode == "fork_tree"`` and read ``nodes``.
        metadata["chained_fork"] = tree_meta
        return AgentResult(
            response=root_run.result.response,
            trajectory=root_run.result.trajectory,
            trace_id=root_run.result.trace_id,
            metadata=metadata,
        )

    async def _execute_session(
        self,
        *,
        incident: str | None,
        data_dir: str,
        scenario: str,
        initial_messages: list[Any] | None = None,
        seed_reminder_text: str | None = None,
        **kwargs: Any,
    ) -> _SessionRun:
        from agentm.core.abi import (
            AgentSessionConfig,
            BeforeSendToLlmEvent,
            EventBus,
            LoopConfig,
            TextContent as _TextContent,
            ToolResultEvent,
        )
        from agentm.core.runtime import AgentSession
        from agentm.core.runtime import create_agent_session
        from rcabench_platform.v3.sdk.evaluation.v2 import AgentRCAOutput

        ctx: RunContext | None = kwargs.get("ctx")
        max_turns = _coerce_max_turns(kwargs.get("max_steps"), self._max_turns)
        os.environ["AGENTM_RCA_DATA_DIR"] = data_dir

        captured: dict[str, Any] = {
            "submission": None,
            "system_prompt": "",
        }

        def _on_tool_result(event: ToolResultEvent) -> None:
            if event.tool_name != "submit_final_report":
                return
            if event.result.is_error:
                return
            text = "".join(
                block.text
                for block in event.result.content
                if isinstance(block, _TextContent)
            )
            try:
                output = AgentRCAOutput.parse_str(text)
            except Exception:
                # Tool returned a non-conforming payload; treat as missing.
                return
            captured["submission"] = output
            if ctx is not None:
                ctx.emit(
                    {
                        "type": "progress",
                        "message": "submit_final_report accepted",
                    }
                )

        def _on_before_llm(event: BeforeSendToLlmEvent) -> None:
            if not captured["system_prompt"] and event.system:
                captured["system_prompt"] = event.system

        bus = EventBus()
        bus.on("tool_result", _on_tool_result)
        bus.on("before_send_to_llm", _on_before_llm)

        # When a pre-resolved config.toml profile is available, use it
        # directly so the agent's endpoint/key are explicit and no
        # ambient OPENAI_*/ANTHROPIC_* env vars can bleed in.  Fall back
        # to the legacy env-based builder for callers that still pass
        # raw model/provider strings (e.g. rcabench-platform eval YAML).
        if self._provider_tuple is not None:
            provider_module, provider_config = self._provider_tuple
        else:
            provider_module, provider_config = _build_provider(
                self._provider, self._model
            )

        # Auto-write a distill meta sidecar so ``llmharness-distill export``
        # can pair this rollout's replay sidecar to ground truth without a
        # hand-fabricated ``meta.json``. ``data_dir`` is the only stable
        # per-sample identifier rcabench-platform exposes to agents (it does
        # not forward ``sample.id`` via kwargs), so we use its basename as
        # ``sample_id``. The binding atom is mounted dynamically per session
        # rather than statically in the manifest because each sample needs a
        # distinct id and an env-var fallback would race under ``-n>1``.
        # Mount conditionally — binding is dead weight without the harness
        # adapter that produces the replay sidecar in the first place.
        # Detection: introspect the resolved scenario manifest, not the
        # scenario name string. See :func:`_scenario_mounts_harness`.
        extra_extensions: list[tuple[str, dict[str, Any]]] = []
        if _scenario_mounts_harness(scenario):
            sample_id = os.path.basename(data_dir.rstrip("/")) or "unknown"
            extra_extensions.append(
                (
                    "llmharness.distill.binding",
                    {
                        "sample_id": sample_id,
                        # No clean dataset_name source exists in this codepath
                        # (rcabench-platform does not forward sample.dataset
                        # to agents). Leave blank rather than hardcode a
                        # single dataset; consumers that need it can read
                        # ``dataset_path`` instead.
                        "dataset_name": "",
                        "dataset_path": data_dir,
                    },
                )
            )
        if seed_reminder_text:
            extra_extensions.append(
                (
                    "llmharness.replay.reminder_seed",
                    {"text": seed_reminder_text},
                )
            )

        config = AgentSessionConfig(
            cwd=os.getcwd(),
            provider=(provider_module, provider_config),
            scenario=scenario,
            extra_extensions=extra_extensions,
            initial_messages=list(initial_messages or []),
            bus=bus,
            loop_config=LoopConfig(max_turns=max_turns, max_tool_calls_per_turn=20),
        )

        session = await create_agent_session(AgentSession, config)
        try:
            if ctx is not None:
                ctx.emit({"type": "running", "run_id": session.session_id})
            if incident is None:
                final_messages = await session.tick()
            else:
                final_messages = await session.prompt(incident)
        finally:
            await session.shutdown()

        submission: AgentRCAOutput | None = captured["submission"]
        if submission is None:
            response = json.dumps(_EMPTY_AGENT_RCA_OUTPUT)
            submission_dump: Any = None
        else:
            response = submission.model_dump_json(by_alias=True)
            submission_dump = submission.model_dump(mode="json", by_alias=True)

        trajectory = _build_trajectory(
            agent_name=f"agentm:{scenario}",
            system_prompt=captured["system_prompt"],
            final_messages=final_messages,
        )

        # OTel-correct trace_id (= ``root_session_id``, shared by all
        # sessions in the rollout tree) goes onto
        # ``AgentResult.trace_id``; rcabench-platform >= 0.4.44 forwards
        # it via ``RolloutResult`` into ``evaluation_data.trace_id``.
        # A single ``trace_id =`` filter then recovers the parent plus
        # every spawned extractor / auditor child JSONL. ``session_id``
        # is kept in metadata for completeness — it identifies the
        # parent's session-root span specifically.
        session_log_id = session.session_manager.get_session_id()
        audit_replay_path = os.path.join(
            os.getcwd(), ".agentm", "audit_replay", f"{session_log_id}.jsonl"
        )
        metadata = {
            "model": self._model,
            "scenario": scenario,
            "max_turns": max_turns,
            "submit_final_report_seen": submission is not None,
            "submission": submission_dump,
            "session_id": session.session_id,
            "session_log_id": session_log_id,
            "audit_replay_path": audit_replay_path,
        }
        result = AgentResult(
            response=response,
            trajectory=trajectory,
            trace_id=session.root_session_id,
            metadata=metadata,
        )
        return _SessionRun(
            result=result,
            final_messages=final_messages,
            response=response,
            submission_dump=submission_dump,
            submit_final_report_seen=submission is not None,
            system_prompt=captured["system_prompt"],
            session_id=session.session_id,
            root_session_id=session.root_session_id,
            session_log_id=session_log_id,
            audit_replay_path=audit_replay_path,
        )


def _scenario_mounts_harness(scenario: str) -> bool:
    """True iff ``scenario`` mounts the llmharness cognitive-audit adapter.

    Detected by loading the resolved scenario manifest and looking for
    ``llmharness.atom`` in its extensions list. Replaces the
    historical ``"harness" in scenario_name`` string-sniff so new harness
    variants (or renames) don't silently miss the distill-binding wire-up.

    Error policy: a malformed scenario manifest is a hard error — we let
    ``ScenarioLoadError`` propagate so the rollout fails loudly rather
    than silently producing an unjoinable replay sidecar. The only
    swallowed case is "manifest legitimately not found on disk": the
    loader wraps the underlying ``FileNotFoundError`` as
    ``ScenarioLoadError(..., cause=FileNotFoundError(...))``. We
    recognise this shape via the ``.cause`` attribute (the loader does
    not chain with ``from exc``, so ``__cause__`` is None — inspect the
    explicit attribute instead). The not-found case is logged at
    WARNING so the wire-up failure is observable in stderr.
    """
    from agentm.extensions.loader import ScenarioLoadError, load_scenario

    try:
        extensions, _meta = load_scenario(scenario)
    except ScenarioLoadError as exc:
        if isinstance(exc.cause, FileNotFoundError):
            logger.warning(
                f"rca eval: scenario {scenario!r} not found; treating as non-harness (distill binding will not be mounted). Check AGENTM_PROJECT_ROOT and contrib/scenarios/ layout. ({exc})"
            )
            return False
        raise
    return any(module == _HARNESS_ADAPTER_MODULE for module, _ in extensions)


def _run_metadata(run: _SessionRun) -> dict[str, Any]:
    return {
        "response": run.response,
        "submission": run.submission_dump,
        "submit_final_report_seen": run.submit_final_report_seen,
        "session_id": run.session_id,
        "root_session_id": run.root_session_id,
        "session_log_id": run.session_log_id,
        "audit_replay_path": run.audit_replay_path,
    }


def _build_trajectory(
    *,
    agent_name: str,
    system_prompt: str,
    final_messages: list[Any],
) -> Trajectory:
    """Translate AgentM's session messages to rcabench's Message schema."""
    from agentm.core.abi import (
        AssistantMessage,
        TextContent,
        ToolCallBlock,
        ToolResultMessage,
        UserMessage,
    )

    messages: list[Message] = []
    for msg in final_messages:
        if isinstance(msg, UserMessage):
            text = "".join(c.text for c in msg.content if isinstance(c, TextContent))
            messages.append(Message(role="user", content=text))
        elif isinstance(msg, AssistantMessage):
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ToolCallBlock):
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=json.dumps(block.arguments, ensure_ascii=False),
                        )
                    )
            messages.append(
                Message(
                    role="assistant",
                    content="\n".join(text_parts),
                    tool_calls=tool_calls or None,
                )
            )
        elif isinstance(msg, ToolResultMessage):
            for result_block in msg.content:
                text = "".join(
                    c.text for c in result_block.content if isinstance(c, TextContent)
                )
                messages.append(
                    Message(
                        role="tool",
                        content=text,
                        tool_call_id=result_block.tool_call_id,
                    )
                )

    return Trajectory(
        agent_trajectories=[
            AgentTrajectory(
                agent_name=agent_name,
                system_prompt=system_prompt,
                turns=[Turn(messages=messages)],
            )
        ]
    )


__all__ = ["AgentMAgent"]
