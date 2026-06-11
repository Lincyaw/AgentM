"""Replay-fork: offline audit + fork counterfactual for RCA.

Run the cognitive-audit harness over a recorded baseline, fork at each
surface point, continue with the reminder, and judge the result.

Built on two primitives:
- ``SessionStore.fork()`` — create a new session from a prefix of an existing one
- ``llmharness.offline_audit()`` — find where the auditor would intervene
"""

from .judge import JudgeOutcome, RcabenchJudge
from .providers import build_profile_provider

__all__ = [
    "JudgeOutcome",
    "RcabenchJudge",
    "build_profile_provider",
]
