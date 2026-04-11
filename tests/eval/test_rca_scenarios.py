"""Layer 4 eval placeholder.

Full RCA scenario evaluation needs external infra and curated datasets,
so this module keeps a minimal placeholder only.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.skip(reason="Layer 4 eval tests are not implemented yet")


def test_rca_scenarios_placeholder() -> None:
    """Sentinel to keep this suite visible in pytest collection."""
    raise NotImplementedError("Layer 4 eval not yet implemented")
