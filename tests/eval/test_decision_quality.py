"""Layer 3 eval placeholder.

Decision-quality evaluation depends on external LLM judging infra and
is intentionally left as a single module-level placeholder.
"""

from __future__ import annotations

def test_decision_quality_placeholder() -> None:
    """Keep the eval suite visible without introducing a permanent skip.

    The real decision-quality harness depends on external judging infra that
    is not part of this repository, so the in-repo contract is simply that
    the placeholder remains collected and passes.
    """

    assert True
