"""Lock the top-level ``llmharness`` public surface.

External consumers (rca-autorl, in particular the online-RL trainer) import
these names from the package root rather than digging into submodule paths.
A regression here breaks downstream installs silently — pin them.
"""

from __future__ import annotations


def test_replay_drivers_and_phase_types_are_public() -> None:
    """The four names needed to drive a single child rollout standalone
    must be reachable via the top-level ``llmharness`` namespace."""
    from llmharness import (
        PhaseResult,
        ReplayRecord,
        Status,
        replay_auditor_record,
        replay_extractor_record,
    )

    # Identity checks — confirm we re-export the canonical objects, not
    # accidental same-named shadows from another module.
    from llmharness.replay.record import ReplayRecord as _ReplayRecord
    from llmharness.replay.record import Status as _Status
    from llmharness.replay.runner import (
        replay_auditor_record as _replay_auditor_record,
    )
    from llmharness.replay.runner import (
        replay_extractor_record as _replay_extractor_record,
    )
    from llmharness.tools.engine import PhaseResult as _PhaseResult

    assert ReplayRecord is _ReplayRecord
    assert Status is _Status
    assert PhaseResult is _PhaseResult
    assert replay_extractor_record is _replay_extractor_record
    assert replay_auditor_record is _replay_auditor_record


def test_all_lists_the_new_names() -> None:
    import llmharness

    for name in (
        "PhaseResult",
        "Status",
        "replay_extractor_record",
        "replay_auditor_record",
    ):
        assert name in llmharness.__all__, name
