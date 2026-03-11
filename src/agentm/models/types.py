"""Shared typing aliases across AgentM."""

from __future__ import annotations

from typing import Literal, TypeAlias

TaskType: TypeAlias = Literal["scout", "verify", "deep_analyze"]
