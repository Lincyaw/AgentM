"""Trajectory judger scenario package."""

from .data import TrajectoryLabel
from .recipe import SCENARIO_NAME, load

__all__ = ["SCENARIO_NAME", "TrajectoryLabel", "load"]
