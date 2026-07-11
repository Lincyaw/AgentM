"""TELBench span-level error localization evaluation."""

from __future__ import annotations

from pathlib import Path


def _tel_agent_dir() -> Path:
    import llmharness.agents.tel as _tel

    return Path(_tel.__file__).parent


__all__ = ["_tel_agent_dir"]
