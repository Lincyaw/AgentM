"""Phase 2 auditor: judges drift over the event graph (no raw trajectory)."""

from __future__ import annotations

from .extensions import compose_auditor_extensions
from .output import AuditorOutputError, RawVerdictOutput
from .prompt import AUDITOR_SYSTEM_PROMPT
from .submit_tool import SUBMIT_VERDICT_TOOL_NAME

__all__ = [
    "AUDITOR_SYSTEM_PROMPT",
    "SUBMIT_VERDICT_TOOL_NAME",
    "AuditorOutputError",
    "RawVerdictOutput",
    "compose_auditor_extensions",
]
