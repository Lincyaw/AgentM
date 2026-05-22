"""DPO pair builder is a Phase-3 sentinel — keep the contract honest.

Track B's DPO loader is being built against the schema documented in
:mod:`llmharness.distill.dpo_pairs`. Until Phase 3 fork-and-continue
lands, the builder must raise loudly rather than silently emit empty
output (which would look like "no preference signal" instead of
"feature not landed yet").
"""

from __future__ import annotations

import pytest

from llmharness.distill.dpo_pairs import dpo_pairs_from_outcomes


def test_dpo_pairs_from_outcomes_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError) as exc_info:
        next(dpo_pairs_from_outcomes(control_bundle={}, alternative_bundles=[]))
    msg = str(exc_info.value)
    assert msg.strip(), "NotImplementedError must carry a non-empty message"
    assert "Phase 3" in msg or "fork-and-continue" in msg
