"""Replayable benchmark harness (doc §6, §7, §13.1).

The RescueHarness machinery that turns recorded baseline trajectories into
``EvalUnit`` rows: trajectory corpus, prefix sampler, treatment factory + oracle,
branch runner, and the experiment orchestrator. Scoring and ground truth are not
here — they come from a scenario via the ``ScenarioAdapter`` seam (``adapter``),
so the harness stays benchmark-agnostic (DESIGN §9). For the read-only RCA data
plane the snapshot adapter is identity (conversation fork), so it has no module
of its own (DESIGN §1).
"""

from .adapter import (
    GroundTruth,
    ScenarioAdapter,
    ScoredOutcome,
    extract_tool_args,
    load_adapter,
)
from .corpus import TrajectoryRef, load_corpus, load_trajectory_messages
from .experiment import run_landscape
from .provider_profiles import build_profile_provider
from .runner import (
    RolloutConfig,
    continue_outcome,
    default_store,
    run_intervention_rollout,
)
from .sampler import PrefixSampler, SamplingPolicy
from .treatments import (
    CONTENT_LADDER,
    ORACLE_LANDSCAPE,
    PRESETS,
    OracleBuilder,
    StrongModelOracle,
    TreatmentFactory,
)

__all__ = [
    "CONTENT_LADDER",
    "ORACLE_LANDSCAPE",
    "PRESETS",
    "GroundTruth",
    "OracleBuilder",
    "PrefixSampler",
    "RolloutConfig",
    "SamplingPolicy",
    "ScenarioAdapter",
    "ScoredOutcome",
    "StrongModelOracle",
    "TrajectoryRef",
    "TreatmentFactory",
    "build_profile_provider",
    "continue_outcome",
    "default_store",
    "extract_tool_args",
    "load_adapter",
    "load_corpus",
    "load_trajectory_messages",
    "run_intervention_rollout",
    "run_landscape",
]
