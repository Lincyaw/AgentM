"""Fail-stop: the submit_final_report wire schema must not drift from the
platform contract.

``finalize.py`` hand-lists ``_FAULT_KIND_ENUM`` (so the atom stays
import-light — it does not pull rcabench_platform at module load). That hand
list is only safe if it stays equal to the platform's ``FaultKind`` enum; a
silent divergence would let the model submit a fault_kind the platform
rejects, or hide a newly-added kind from the model. This test is the lock.
"""

from __future__ import annotations

import pytest

from agentm_rca.tools import finalize


def test_fault_kind_enum_matches_platform() -> None:
    FaultKind = pytest.importorskip(
        "rcabench_platform.v3.sdk.evaluation.v2"
    ).FaultKind
    assert finalize._FAULT_KIND_ENUM == [k.value for k in FaultKind]
