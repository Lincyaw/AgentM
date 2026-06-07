"""Fork strategies for the replay-fork experiment.

A :class:`ForkStrategy` decides *when* and *what* to inject into the agent's
trajectory.  The driver delegates to the strategy for the fork logic and
handles judging, result formatting, and orchestration uniformly.

Two built-in strategies:

* :class:`HarnessStrategy` -- run the full extractor + auditor pipeline,
  fork wherever the auditor surfaces a reminder (the live-harness analogue).
* :class:`FixedInjectionStrategy` -- inject a fixed reminder at a computed
  fork point, bypassing the auditor entirely.  The ``turn_selector``
  parameter controls *where* in the backbone the injection lands.

Adding a new ablation is a new ``ForkStrategy`` subclass (or a new
``turn_selector`` for :class:`FixedInjectionStrategy`), not a flag on the
driver.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.core.abi.messages import AgentMessage
from llmharness.audit.triggers import TriggerRegistry

from .case_source import ReplayCase

_logger = logging.getLogger(__name__)

__all__ = [
    "FixedInjectionStrategy",
    "ForkStrategy",
    "HarnessStrategy",
    "after_submission",
    "before_submission",
]


# ---------------------------------------------------------------------------
# Turn selectors: backbone -> fork index
# ---------------------------------------------------------------------------

#: The tool names that terminate an RCA investigation. The orchestrator
#: submits with ``submit_final_report`` (see ``tools/finalize.py``);
#: ``submit_investigation`` is the legacy name kept so recordings made before
#: the rename still resolve. Single source of truth — ``cli.py`` reuses this
#: same set when wiring the on-submission trigger.
_SUBMISSION_TOOL_NAMES = frozenset({"submit_final_report", "submit_investigation"})


def _find_submission_turn(messages: list[AgentMessage]) -> int:
    """Index of the last assistant message that calls a submission tool.

    Scans backward so we find the *last* submission call (the final answer).
    Returns the index of that message, so callers can slice with
    ``messages[:index]`` to get the prefix just before the submission turn.

    Falls back to ``len(messages) - 1`` if no submission tool call is found
    (still forks before the very last message).
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if getattr(msg, "role", "") != "assistant":
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            # Dataclass ToolCallBlock: has .name attribute
            name = getattr(block, "name", None)
            if name in _SUBMISSION_TOOL_NAMES:
                return i
            # Dict-style block (rehydrated JSON)
            if isinstance(block, dict):
                if block.get("name") in _SUBMISSION_TOOL_NAMES:
                    return i
                # Fallback: check for root_causes in arguments/input
                args = block.get("input") or block.get("arguments")
                if isinstance(args, dict) and "root_causes" in args:
                    return i
    # No submission found; fork before the very last message.
    return max(len(messages) - 1, 0)


#: Turn selector: fork just before the agent's final submission call.
TurnSelector = Callable[[list[AgentMessage]], int]


def before_submission(messages: list[AgentMessage]) -> int:
    """Fork just before the agent's final submission-tool call."""
    return _find_submission_turn(messages)


def after_submission(messages: list[AgentMessage]) -> int:
    """Fork after the full trajectory, including the submission and its result.

    The agent sees its own final answer before receiving the reminder,
    so it can reflect on what it actually submitted.
    """
    return len(messages)


# ---------------------------------------------------------------------------
# Fork-execution result (strategy -> driver)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForkResult:
    """What a strategy hands back to the driver for judging.

    ``response`` is the raw JSON string the continuation submitted (or
    ``None`` if no continuation ran).  ``fired`` indicates whether the
    strategy produced an intervention at all.  ``intervention_path`` lists
    the reminder texts injected (one per fork depth).
    """

    fired: bool
    n_interventions: int
    intervention_path: list[str]
    response: str | None
    leaf_session_log_id: str | None


# ---------------------------------------------------------------------------
# Strategy protocol and implementations
# ---------------------------------------------------------------------------

class ForkStrategy:
    """Base class for fork strategies.  Subclasses implement :meth:`execute`."""

    @property
    def label(self) -> str:
        """Short human-readable name for logging and result tagging."""
        return type(self).__name__

    async def execute(
        self,
        case: ReplayCase,
        *,
        agent: Any,
        scenario: str,
    ) -> ForkResult:
        """Run the strategy on *case* and return the fork result.

        Parameters
        ----------
        case:
            The recorded baseline to replay-and-fork.
        agent:
            The ``AgentMAgent`` used for continuation rollouts.
        scenario:
            Scenario id passed to continuation sessions.
        """
        raise NotImplementedError


class HarnessStrategy(ForkStrategy):
    """Run the full extractor + auditor pipeline, fork on auditor surfaces.

    This is the default strategy and the live-harness analogue: the
    fork-tree engine re-audits the backbone, and wherever the auditor fires,
    a continuation is spawned with the surfaced reminder seeded.
    """

    def __init__(
        self,
        *,
        harness_provider: tuple[str, dict[str, Any]],
        max_depth: int = 3,
        extractor_interval: int = 5,
        audit_interval: int = 5,
        cwd: str | None = None,
        sidecar_dir: str | os.PathLike[str] | None = None,
        skip_extractor: bool = False,
        trigger_registry: TriggerRegistry | None = None,
        auditor_prompt: str | None = None,
    ) -> None:
        self._harness_provider = harness_provider
        self._max_depth = max_depth
        self._extractor_interval = extractor_interval
        self._audit_interval = audit_interval
        self._cwd = cwd or os.getcwd()
        self._sidecar_dir = Path(sidecar_dir) if sidecar_dir is not None else None
        self._skip_extractor = skip_extractor
        self._trigger_registry = trigger_registry
        self._auditor_prompt = auditor_prompt

    @property
    def label(self) -> str:
        sfx = ",skip_ext" if self._skip_extractor else ""
        return f"harness(depth={self._max_depth}{sfx})"

    async def execute(
        self,
        case: ReplayCase,
        *,
        agent: Any,
        scenario: str,
    ) -> ForkResult:
        from llmharness import (
            AuditorSettings,
            ExtractorSettings,
            SessionPayload,
            run_fork_tree_experiment,
        )

        session_runs: dict[str, Any] = {}
        control_id = f"{case.case_id}-control"

        async def factory(
            *,
            initial_messages: list[Any] | None,
            seed_reminder_text: str | None,
        ) -> SessionPayload:
            if initial_messages is None:
                # Control backbone: serve the recording, never re-run.
                return _RecordedBackbone(  # type: ignore[return-value]
                    session_log_id=control_id,
                    final_messages=case.backbone_messages,
                )
            run = await agent._execute_session(
                incident=None,
                data_dir=case.data_dir,
                scenario=scenario,
                initial_messages=initial_messages,
                seed_reminder_text=seed_reminder_text,
            )
            session_runs[run.session_log_id] = run
            return run  # type: ignore[return-value]

        out_path = None
        if self._sidecar_dir is not None:
            out_path = self._sidecar_dir / f"{case.case_id}.chained.jsonl"

        auditor_settings = AuditorSettings.default()
        if self._auditor_prompt is not None:
            from llmharness import load_auditor_prompt

            auditor_settings = AuditorSettings(
                base_prompt=load_auditor_prompt(self._auditor_prompt),
                observability_config=auditor_settings.observability_config,
                summary_threshold=auditor_settings.summary_threshold,
                tools=auditor_settings.tools,
            )

        experiment = await run_fork_tree_experiment(
            session_factory=factory,
            cwd=self._cwd,
            provider=self._harness_provider,
            extractor_settings=ExtractorSettings.default(),
            auditor_settings=auditor_settings,
            extractor_interval=self._extractor_interval,
            audit_interval=self._audit_interval,
            max_depth=self._max_depth,
            max_surfaces_per_node=1,
            out_path=out_path,
            skip_extractor=self._skip_extractor,
            trigger_registry=self._trigger_registry,
            trace_id=control_id,
        )

        fork_nodes = [n for n in experiment.nodes if n.parent_id is not None]
        if not fork_nodes:
            return ForkResult(
                fired=False,
                n_interventions=0,
                intervention_path=[],
                response=case.control_response,
                leaf_session_log_id=None,
            )

        leaf = max(fork_nodes, key=lambda n: n.depth)
        leaf_run = session_runs.get(leaf.backbone_session_id)
        return ForkResult(
            fired=True,
            n_interventions=leaf.depth,
            intervention_path=list(leaf.path),
            response=getattr(leaf_run, "response", None),
            leaf_session_log_id=leaf.backbone_session_id,
        )


class FixedInjectionStrategy(ForkStrategy):
    """Inject a fixed reminder at a computed fork point.

    Bypasses the auditor pipeline entirely.  Useful for ablations that test
    *what* to say independently of *when* the auditor decides to say it.

    Parameters
    ----------
    reminder:
        The reminder text to inject.
    turn_selector:
        A callable ``(messages) -> int`` returning the index of the message
        to fork *before*.  Defaults to :func:`before_submission`.
    """

    def __init__(
        self,
        *,
        reminder: str,
        turn_selector: TurnSelector | None = None,
    ) -> None:
        self._reminder = reminder
        self._turn_selector = turn_selector or before_submission

    @property
    def label(self) -> str:
        sel_name = getattr(self._turn_selector, "__name__", "custom")
        return f"fixed_injection(at={sel_name})"

    async def execute(
        self,
        case: ReplayCase,
        *,
        agent: Any,
        scenario: str,
    ) -> ForkResult:
        messages = case.backbone_messages
        fork_point = self._turn_selector(messages)
        _logger.info(
            "%s: case=%s fork_point=%d/%d",
            self.label,
            case.case_id,
            fork_point,
            len(messages),
        )

        run = await agent._execute_session(
            incident=None,
            data_dir=case.data_dir,
            scenario=scenario,
            initial_messages=messages[:fork_point],
            seed_reminder_text=self._reminder,
        )

        return ForkResult(
            fired=True,
            n_interventions=1,
            intervention_path=[self._reminder],
            response=getattr(run, "response", None),
            leaf_session_log_id=getattr(run, "session_log_id", None),
        )


# ---------------------------------------------------------------------------
# Shared internal dataclass (used by HarnessStrategy)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RecordedBackbone:
    """Control backbone served from a recording (no agent run).

    Structurally satisfies the engine's ``SessionPayload`` protocol
    (``session_log_id`` + ``final_messages``).
    """

    session_log_id: str
    final_messages: list[AgentMessage]


# ---------------------------------------------------------------------------
# Pre-built prompts for common ablations
# ---------------------------------------------------------------------------

UPPER_BOUND_REFLECTION = (
    "Hold on — before your answer is accepted, reconsider it. "
    "Look at the root causes you just submitted: are you confident you "
    "have identified ALL contributing faults, not just the most obvious one? "
    "Review your earlier queries — were there services with anomalous "
    "metrics, elevated error rates, or missing traces that you noticed but "
    "never fully investigated? Were there data sources you listed at the "
    "start but never queried? If you find gaps, resubmit with a more "
    "complete answer."
)
