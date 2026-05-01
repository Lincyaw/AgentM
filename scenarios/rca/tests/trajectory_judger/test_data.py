from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import pytest
from pydantic import ValidationError

from agentm_rca.trajectory_judger.data import TrajectoryLabel, TrajectoryStats


AllowedCategory = Literal[
    "success",
    "lucky_hit",
    "exploration_fail",
    "confirmation_fail",
    "judgment_fail",
]


@pytest.mark.parametrize(
    "category",  # type: ignore[misc]
    [
        "success",
        "lucky_hit",
        "exploration_fail",
        "confirmation_fail",
        "judgment_fail",
    ],
)
def test_trajectory_label_accepts_all_supported_categories(category: AllowedCategory) -> None:
    label = TrajectoryLabel(
        trajectory_id="traj-1",
        case_id="case-1",
        agent_conclusion=["checkout"],
        ground_truth=["checkout"],
        is_correct=True,
        is_partial=False,
        category=category,
        reasoning="The agent followed the available evidence long enough to satisfy the minimum reasoning length.",
        evidence=[],
        key_steps={"conclusion_step": 7},
        stats=TrajectoryStats(total_steps=7),
        analyzed_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    assert label.category == category


def test_trajectory_label_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError, match="category"):
        TrajectoryLabel(
            trajectory_id="traj-1",
            case_id="case-1",
            agent_conclusion=["checkout"],
            ground_truth=["payments"],
            is_correct=False,
            is_partial=False,
            category="wrong",  # type: ignore[arg-type]
            reasoning="The agent followed the available evidence long enough to satisfy the minimum reasoning length.",
        )
