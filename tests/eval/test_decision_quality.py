"""Layer 3 eval placeholder.

Decision-quality evaluation depends on external LLM judging infra and
is intentionally left as a single module-level placeholder.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.skip(reason="Layer 3 eval tests are not implemented yet")


def test_decision_quality_placeholder() -> None:
    """Sentinel to keep this suite visible in pytest collection."""
    raise NotImplementedError("Layer 3 eval not yet implemented")
