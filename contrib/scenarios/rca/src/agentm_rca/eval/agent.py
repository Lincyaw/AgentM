"""``rcabench-platform`` BaseAgent adapter for the AgentM RCA scenario.

Discovered by ``rcabench-platform``'s ``llm_eval.agents`` entry point and
invoked via ``rca llm-eval run --agent agentm``. Bridges the
``incident + data_dir -> CausalGraph JSON`` contract to an in-process
``AgentSession`` running the local ``rca`` scenario.

Two pieces of glue do the work:

* The orchestrator's ``submit_final_report`` tool in
  :mod:`agentm_rca.tools.finalize` validates against the rcabench-platform
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

Boundary: this module is a host-side driver (not a §11 atom — no
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

_INTERVENTION_BASELINE_FORK = "baseline_fork"
_BASELINE_FORK_ALIASES = {
    "baseline_fork",
    "baseline-fork",
    "fork_audit",
    "fork-audit",
}


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


def _build_provider(
    provider: str, model: str
) -> tuple[str, dict[str, Any]]:
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
            # PR #95 (8b8231e) made ``name`` mandatory whenever
            # ``base_url`` is non-canonical, to prevent two custom
            # endpoints from clobbering each other under the bare
            # ``openai`` registry slot. Honor an explicit override via
            # ``AGENTM_PROVIDER_NAME`` and otherwise derive a stable
            # slug from the base host so eval rollouts don't crash.
            cfg["name"] = os.environ.get(
                "AGENTM_PROVIDER_NAME"
            ) or _provider_name_from_base_url(base_url)
        ticket = os.environ.get("WARPGATE_TICKET")
        if ticket:
            cfg["default_query"] = {"warpgate-ticket": ticket}
        if not _env_bool("OPENAI_VERIFY_SSL", default=True):
            cfg["verify_ssl"] = False
        return ("agentm.extensions.builtin.llm_openai", cfg)

    raise ValueError(
        f"unknown provider {provider!r}; expected 'anthropic' or 'openai'"
    )


def _coerce_max_turns(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        result = int(value)
    except (TypeError, ValueError):
        return fallback
    return result if result > 0 else fallback


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
class _ReminderCandidate:
    turn_index: int
    text: str
    record: Any


@dataclass(frozen=True)
class _OfflineAuditRun:
    reminder: _ReminderCandidate | None
    records: list[Any]


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
        exp_id: str | None = None,
        max_turns: Any = None,
        intervention_mode: str | None = None,
        control_scenario: str | None = None,
        branch_scenario: str | None = None,
        fork_audit: Any = None,
        fork_policy: str = "first_surface",
        **_extra: Any,
    ) -> None:
        # ``rcabench-platform`` passes ``exp_id`` plus any ``--ak key=value``
        # kwargs to the agent ctor. We accept and ignore the unknowns so a
        # stale flag in someone's eval YAML does not crash the rollout.
        # ``--ak`` values arrive as strings, so coerce explicitly.
        self._scenario = scenario
        self._model = model or os.environ.get("AGENTM_MODEL", _DEFAULT_MODEL)
        self._provider = (
            provider
            or os.environ.get("AGENTM_PROVIDER")
            or "anthropic"
        )
        self._exp_id = exp_id
        self._max_turns = _coerce_max_turns(max_turns, _DEFAULT_MAX_TURNS)
        if intervention_mode:
            mode = intervention_mode.strip()
            self._intervention_mode = (
                _INTERVENTION_BASELINE_FORK
                if mode in _BASELINE_FORK_ALIASES
                else mode
            )
        elif _coerce_bool(fork_audit):
            self._intervention_mode = _INTERVENTION_BASELINE_FORK
        else:
            self._intervention_mode = None
        self._fork_policy = fork_policy
        self._control_scenario = (
            control_scenario
            or os.environ.get("AGENTM_FORK_CONTROL_SCENARIO")
            or _default_control_scenario(scenario)
        )
        self._branch_scenario = (
            branch_scenario
            or os.environ.get("AGENTM_FORK_BRANCH_SCENARIO")
            or self._control_scenario
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
        if self._intervention_mode == _INTERVENTION_BASELINE_FORK:
            return await self._run_baseline_fork(
                incident=incident,
                data_dir=data_dir,
                **kwargs,
            )
        if self._intervention_mode:
            raise ValueError(f"unknown intervention_mode={self._intervention_mode!r}")
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

    async def _run_baseline_fork(
        self,
        *,
        incident: str,
        data_dir: str,
        **kwargs: Any,
    ) -> AgentResult:
        if self._fork_policy != "first_surface":
            raise ValueError(
                "baseline_fork currently supports only fork_policy='first_surface'"
            )

        control = await self._execute_session(
            incident=incident,
            data_dir=data_dir,
            scenario=self._control_scenario,
            **kwargs,
        )
        offline_audit = await _run_offline_auditor_over_control(
            control=control,
            cwd=os.getcwd(),
            provider=_build_provider(self._provider, self._model),
        )
        reminder = offline_audit.reminder
        if reminder is None:
            metadata = dict(control.result.metadata or {})
            metadata["intervention_mode"] = _INTERVENTION_BASELINE_FORK
            metadata["intervention_status"] = "no_surface_reminder"
            metadata["control_scenario"] = self._control_scenario
            metadata["branch_scenario"] = self._branch_scenario
            metadata["control"] = _run_metadata(control)
            metadata["offline_auditor_firings"] = len(offline_audit.records)
            return AgentResult(
                response=control.result.response,
                trajectory=control.result.trajectory,
                trace_id=control.result.trace_id,
                metadata=metadata,
            )

        prefix_messages = _fork_prefix_messages(
            control.final_messages,
            turn_index=reminder.turn_index,
        )
        branch = await self._execute_session(
            incident=None,
            data_dir=data_dir,
            scenario=self._branch_scenario,
            initial_messages=prefix_messages,
            seed_reminder_text=reminder.text,
            **kwargs,
        )
        branch_audit = await _run_offline_auditor_over_control(
            control=branch,
            cwd=os.getcwd(),
            provider=_build_provider(self._provider, self._model),
            stop_on_first_surface=False,
        )
        strict_replay_path = _write_strict_ab_replay(
            control=control,
            branch=branch,
            offline_auditor_records=offline_audit.records,
            branch_auditor_records=branch_audit.records,
            reminder=reminder,
            cwd=os.getcwd(),
        )
        metadata = dict(branch.result.metadata or {})
        metadata["intervention_mode"] = _INTERVENTION_BASELINE_FORK
        metadata["intervention_status"] = "forked"
        metadata["control_scenario"] = self._control_scenario
        metadata["branch_scenario"] = self._branch_scenario
        metadata["strict_ab_replay_path"] = strict_replay_path
        metadata["fork"] = {
            "policy": self._fork_policy,
            "turn_index": reminder.turn_index,
            "reminder_text": reminder.text,
            "prefix_message_count": len(prefix_messages),
        }
        metadata["control"] = _run_metadata(control)
        metadata["branch"] = _run_metadata(branch)
        return AgentResult(
            response=branch.result.response,
            trajectory=branch.result.trajectory,
            trace_id=branch.result.trace_id,
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
            BeforeSendToLlmEvent,
            EventBus,
            ToolResultEvent,
        )
        from agentm.core.abi.messages import TextContent as _TextContent
        from agentm.core.abi.loop import LoopConfig
        from agentm.core.abi.session_config import AgentSessionConfig
        from agentm.core.runtime.session import AgentSession
        from agentm.core.runtime.session_factory import create_agent_session
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

        # Mirror ``agentm.cli._build_provider`` so the eval adapter honors
        # the same ``AGENTM_PROVIDER`` / ``OPENAI_*`` / ``ANTHROPIC_*``
        # convention as ``uv run agentm``. Previously we always pinned
        # the anthropic provider, which silently routed Doubao-Seed
        # requests through whatever ``ANTHROPIC_BASE_URL`` happened to
        # point at (e.g. the Kimi anthropic-compat gateway) instead of
        # the intended ``OPENAI_BASE_URL`` LiteLLM endpoint.
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
        extra_extensions: list[tuple[str, dict[str, Any]]] = []
        if "harness" in scenario:
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
            loop_config=LoopConfig(max_turns=max_turns),
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
            submission_dump = submission.model_dump(
                mode="json", by_alias=True
            )

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


def _default_control_scenario(scenario: str) -> str:
    """Pick a no-auditor control scenario for strict A/B fork mode."""
    if scenario in {
        "rca:harness.sync",
        "rca:harness.sync.opinions",
        "rca:harness.sync.opinions10",
    }:
        return "rca:harness.sync.extractor5"
    return scenario


def _coerce_schema_list(cls: Any, items: Any) -> list[Any]:
    out: list[Any] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            out.append(cls.from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return out


async def _run_offline_auditor_over_control(
    *,
    control: _SessionRun,
    cwd: str,
    provider: tuple[str, dict[str, Any]] | None,
    stop_on_first_surface: bool = True,
) -> _OfflineAuditRun:
    """Run auditor side-channel over the fixed control trajectory.

    Strict A/B means the control session must not mount the auditor. This
    helper reconstructs each auditor firing from the extractor-only replay
    sidecar, so the judgment is made against an immutable control prefix.
    """
    from pathlib import Path

    from llmharness.adapters.agentm import (
        _flatten_assistant_blocks,
        _serialize_full_trajectory,
    )
    from llmharness.audit.auditor.output import (
        AuditorOutputError,
        RawVerdictOutput,
    )
    from llmharness.audit.phase import merge_to_phases
    from llmharness.replay.record import ReplayRecord, iter_records, now_ns
    from llmharness.replay.runner import replay_auditor_record
    from llmharness.schema import Edge, Event

    replay_path = Path(control.audit_replay_path)
    if not replay_path.exists():
        return _OfflineAuditRun(reminder=None, records=[])

    events: list[Any] = []
    edges: list[Any] = []
    phases: list[Any] = []
    recent_verdicts: list[dict[str, Any]] = []
    continuation_notes: list[str] = []
    auditor_records: list[Any] = []

    extractor_records = [
        rec
        for rec in iter_records(replay_path)
        if rec.phase == "extractor" and rec.status == "ok" and rec.output
    ]
    extractor_records.sort(key=lambda rec: (rec.ts_ns, rec.turn_index))

    for extractor_record in extractor_records:
        output = extractor_record.output or {}
        new_events = _coerce_schema_list(Event, output.get("events") or [])
        new_edges = _coerce_schema_list(Edge, output.get("edges") or [])
        events.extend(new_events)
        edges.extend(new_edges)
        phases.extend(merge_to_phases(new_events))

        cut = min(max(extractor_record.turn_index + 1, 0), len(control.final_messages))
        trajectory_snapshot = _serialize_full_trajectory(control.final_messages[:cut])
        compose_kwargs = {
            "base_prompt": None,
            "cards_tools_config": {},
            "observability_config": {},
            "trajectory_snapshot": trajectory_snapshot,
            "events": [ev.to_dict() for ev in events],
            "edges": [ed.to_dict() for ed in edges],
            "phases": [ph.to_dict() for ph in phases],
            "findings": [],
            "check_errors": {},
            "continuation_notes": list(continuation_notes),
            "summary_threshold": 30,
            "tools": ["submit_verdict"],
        }
        payload = {
            "graph": [ev.to_dict() for ev in events],
            "recent_verdicts": list(recent_verdicts),
            "continuation_notes_from_prior_firing": list(continuation_notes),
        }
        replay_record = ReplayRecord(
            phase="auditor",
            turn_index=extractor_record.turn_index,
            root_session_id=control.session_log_id,
            ts_ns=(extractor_record.ts_ns or now_ns()) + 1,
            compose_kwargs=compose_kwargs,
            payload=payload,
            provider=None,
            output=None,
            status="ok",
        )
        phase_result = await replay_auditor_record(
            replay_record,
            cwd=cwd,
            provider_override=provider,
        )
        verdict_dict: dict[str, Any] | None = None
        if phase_result.status == "ok" and isinstance(phase_result.output, dict):
            try:
                verdict_dict = RawVerdictOutput.from_dict(
                    phase_result.output
                ).to_verdict().to_dict()
            except AuditorOutputError:
                verdict_dict = None
        auditor_record = ReplayRecord(
            phase="auditor",
            turn_index=extractor_record.turn_index,
            root_session_id=control.session_log_id,
            ts_ns=replay_record.ts_ns,
            compose_kwargs=compose_kwargs,
            payload=payload,
            provider=[provider[0], provider[1]] if provider else None,
            output=verdict_dict,
            status=phase_result.status if verdict_dict is not None else "no_call",
            error=phase_result.error,
            latency_ms=phase_result.latency_ms,
            raw_assistant_messages=_flatten_assistant_blocks(phase_result.messages),
        )
        auditor_records.append(auditor_record)

        verdict = verdict_dict
        if verdict is not None:
            recent_verdicts.append(verdict)
            raw_notes = verdict.get("continuation_notes")
            continuation_notes = [
                str(n) for n in raw_notes if isinstance(n, str)
            ] if isinstance(raw_notes, list) else []

        if not verdict or not verdict.get("surface_reminder"):
            continue
        text = verdict.get("reminder_text")
        if not isinstance(text, str) or not text.strip():
            continue
        reminder = _ReminderCandidate(
            turn_index=extractor_record.turn_index,
            text=text,
            record=auditor_record,
        )
        if stop_on_first_surface:
            return _OfflineAuditRun(reminder=reminder, records=auditor_records)

    first_reminder = next(
        (
            _ReminderCandidate(
                turn_index=int(record.turn_index),
                text=str((record.output or {}).get("reminder_text", "")),
                record=record,
            )
            for record in auditor_records
            if isinstance(record.output, dict)
            and record.output.get("surface_reminder")
            and str(record.output.get("reminder_text", "")).strip()
        ),
        None,
    )
    return _OfflineAuditRun(reminder=first_reminder, records=auditor_records)


def _clone_replay_record(
    record: Any,
    *,
    root_session_id: str,
    ts_ns: int | None = None,
) -> Any:
    from llmharness.replay.record import ReplayRecord

    provider = record.provider
    if isinstance(provider, list):
        provider = list(provider)
    return ReplayRecord(
        phase=record.phase,
        turn_index=int(record.turn_index),
        root_session_id=root_session_id,
        ts_ns=int(ts_ns if ts_ns is not None else record.ts_ns),
        compose_kwargs=dict(record.compose_kwargs or {}),
        payload=dict(record.payload or {}),
        provider=provider,
        output=dict(record.output) if isinstance(record.output, dict) else record.output,
        status=record.status,
        error=record.error,
        latency_ms=int(record.latency_ms or 0),
        extras=dict(record.extras or {}),
        raw_assistant_messages=list(record.raw_assistant_messages or []),
    )


def _write_strict_ab_replay(
    *,
    control: _SessionRun,
    branch: _SessionRun,
    offline_auditor_records: list[Any],
    branch_auditor_records: list[Any] | None = None,
    reminder: _ReminderCandidate,
    cwd: str,
) -> str:
    """Materialize the website sidecar for the with-auditor branch.

    The sidecar is intentionally composed as:
    control extractor/auditor prefix -> branch extractor/auditor tail. Silent
    auditor verdicts are persisted too, so the case viewer shows one auditor
    firing for every extractor firing while only the surfaced reminder changes
    the main-agent trajectory.
    """
    from pathlib import Path

    from llmharness.replay.record import iter_records, write_record

    out_path = (
        Path(cwd)
        / ".agentm"
        / "audit_replay"
        / f"{branch.session_log_id}.strict_ab.jsonl"
    )
    if out_path.exists():
        out_path.unlink()

    control_records = list(iter_records(Path(control.audit_replay_path)))
    branch_records = list(iter_records(Path(branch.audit_replay_path)))
    root_session_id = branch.session_log_id
    control_auditors_by_turn = {
        int(record.turn_index): record for record in offline_auditor_records
    }
    branch_auditors_by_turn = {
        int(record.turn_index): record for record in (branch_auditor_records or [])
    }

    for record in control_records:
        if record.phase != "extractor":
            continue
        if int(record.turn_index) > reminder.turn_index:
            continue
        write_record(
            out_path,
            _clone_replay_record(record, root_session_id=root_session_id),
        )
        auditor = control_auditors_by_turn.get(int(record.turn_index))
        if auditor is not None:
            write_record(
                out_path,
                _clone_replay_record(auditor, root_session_id=root_session_id),
            )

    branch_tail_written = False
    for record in branch_records:
        if record.phase != "extractor":
            continue
        if int(record.turn_index) <= reminder.turn_index:
            continue
        branch_tail_written = True
        write_record(
            out_path,
            _clone_replay_record(record, root_session_id=root_session_id),
        )
        auditor = branch_auditors_by_turn.get(int(record.turn_index))
        if auditor is not None:
            write_record(
                out_path,
                _clone_replay_record(auditor, root_session_id=root_session_id),
            )

    if not branch_tail_written:
        for record in branch_records:
            if record.phase != "extractor":
                continue
            write_record(
                out_path,
                _clone_replay_record(record, root_session_id=root_session_id),
            )
            auditor = branch_auditors_by_turn.get(int(record.turn_index))
            if auditor is not None:
                write_record(
                    out_path,
                    _clone_replay_record(auditor, root_session_id=root_session_id),
                )

    return str(out_path)


def _find_first_surface_reminder(audit_replay_path: str) -> _ReminderCandidate | None:
    """Pick the first auditor verdict that would have affected the agent.

    The baseline-fork experiment runs the baseline with reminders disabled,
    so the replay sidecar is the only place where auditor decisions live.
    """
    if not os.path.exists(audit_replay_path):
        return None

    with open(audit_replay_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("phase") != "auditor":
                continue
            if record.get("status") != "ok":
                continue
            output = record.get("output")
            if not isinstance(output, dict):
                continue
            if not output.get("surface_reminder"):
                continue
            text = output.get("reminder_text")
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                turn_index = int(record.get("turn_index"))
            except (TypeError, ValueError):
                continue
            return _ReminderCandidate(
                turn_index=turn_index,
                text=text,
                record=record,
            )
    return None


def _fork_prefix_messages(final_messages: list[Any], *, turn_index: int) -> list[Any]:
    """Return the main-agent prefix that matches live reminder delivery.

    ``llmharness`` records the auditor turn at ``TurnEndEvent`` time, which
    is after the assistant message but before tool execution. In the live
    path the reminder is injected only after the turn's tool results have
    been appended, so a prefix fork must include the paired ToolResultMessage
    when the turn's assistant message made tool calls.
    """
    if turn_index < 0:
        return []
    cut = min(turn_index + 1, len(final_messages))
    if cut >= len(final_messages):
        return list(final_messages[:cut])

    from agentm.core.abi.messages import (
        AssistantMessage,
        ToolCallBlock,
        ToolResultMessage,
    )

    current = final_messages[turn_index] if turn_index < len(final_messages) else None
    following = final_messages[cut]
    if isinstance(current, AssistantMessage) and isinstance(
        following, ToolResultMessage
    ):
        assistant_tool_call_ids = {
            block.id for block in current.content if isinstance(block, ToolCallBlock)
        }
        result_tool_call_ids = {
            block.tool_call_id
            for block in following.content
            if hasattr(block, "tool_call_id")
        }
        if assistant_tool_call_ids & result_tool_call_ids:
            cut += 1
    return list(final_messages[:cut])


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
    from agentm.core.abi.messages import (
        AssistantMessage,
        TextContent,
        ToolCallBlock,
        ToolResultMessage,
        UserMessage,
    )

    messages: list[Message] = []
    for msg in final_messages:
        if isinstance(msg, UserMessage):
            text = "".join(
                c.text for c in msg.content if isinstance(c, TextContent)
            )
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
                            arguments=json.dumps(
                                block.arguments, ensure_ascii=False
                            ),
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
                    c.text
                    for c in result_block.content
                    if isinstance(c, TextContent)
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
