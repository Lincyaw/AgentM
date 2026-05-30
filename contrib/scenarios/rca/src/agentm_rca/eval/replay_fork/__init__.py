"""Replay-fork counterfactual experiment for the RCA scenario.

Re-run the cognitive-audit harness (extractor + auditor) over a *recorded*
baseline trajectory with a chosen harness model, and -- wherever the
auditor surfaces a reminder -- fork the main agent at that point and let it
continue with the reminder seeded. The control arm is the recorded
baseline itself (the main agent is never re-run for control); only the
intervention continuations cost a fresh rollout. Comparing the judged
control answer against the judged intervention leaf answers measures
whether the harness's reminders actually raise the success rate.

The pieces are deliberately decoupled so the experiment is re-runnable as
storage / models / judges evolve:

* :class:`CaseSource` -- where recorded baselines come from. The eval.db
  schema is confined to :class:`EvalDbCaseSource`; the driver only ever
  sees :class:`ReplayCase`.
* :func:`openai_chat_to_agentm` -- rehydrate OpenAI-style chat messages
  into AgentM ``AgentMessage`` objects (inverse of the rca eval driver's
  trajectory serializer).
* :func:`build_profile_provider` -- build a provider tuple from a
  ``~/.agentm/config.toml`` profile, so the harness model can target a
  different endpoint than the agent model in the same process.
* :class:`ForkStrategy` -- pluggable fork logic.  Built-in strategies:
  :class:`HarnessStrategy` (extractor + auditor pipeline) and
  :class:`FixedInjectionStrategy` (inject a fixed reminder at a computed
  turn).

The driver, judge, and result sinks live in sibling modules and build on
this foundation.
"""

from __future__ import annotations

from .case_source import CaseSource, EvalDbCaseSource, ReplayCase
from .driver import (
    JsonlResultSink,
    ReplayCaseResult,
    ReplayForkDriver,
    ReplaySummary,
    ResultSink,
)
from .judge import JudgeOutcome, LeafJudge, RcabenchJudge
from .providers import build_profile_provider
from .strategy import (
    FixedInjectionStrategy,
    ForkStrategy,
    HarnessStrategy,
    UPPER_BOUND_REFLECTION,
    after_submission,
    before_submission,
)
from .trajectory import openai_chat_to_agentm

__all__ = [
    "CaseSource",
    "EvalDbCaseSource",
    "FixedInjectionStrategy",
    "ForkStrategy",
    "HarnessStrategy",
    "JsonlResultSink",
    "JudgeOutcome",
    "LeafJudge",
    "RcabenchJudge",
    "ReplayCase",
    "ReplayCaseResult",
    "ReplayForkDriver",
    "ReplaySummary",
    "ResultSink",
    "UPPER_BOUND_REFLECTION",
    "after_submission",
    "before_submission",
    "build_profile_provider",
    "openai_chat_to_agentm",
]
