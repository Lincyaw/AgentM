"""Session cwd validation helpers, ported from pi-mono's session-cwd.ts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class SessionCwdSource(Protocol):
    def get_cwd(self) -> str: ...
    def get_session_file(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class SessionCwdIssue:
    session_file: str | None
    session_cwd: str
    fallback_cwd: str


class MissingSessionCwdError(RuntimeError):
    def __init__(self, issue: SessionCwdIssue) -> None:
        self.issue = issue
        session_file = (
            f"\nSession file: {issue.session_file}" if issue.session_file else ""
        )
        super().__init__(
            "Stored session working directory does not exist: "
            f"{issue.session_cwd}{session_file}\n"
            f"Current working directory: {issue.fallback_cwd}"
        )


def get_missing_session_cwd_issue(
    session_manager: SessionCwdSource,
    fallback_cwd: str,
) -> SessionCwdIssue | None:
    session_file = session_manager.get_session_file()
    if session_file is None:
        return None
    session_cwd = session_manager.get_cwd()
    if not session_cwd or Path(session_cwd).exists():
        return None
    return SessionCwdIssue(
        session_file=session_file,
        session_cwd=session_cwd,
        fallback_cwd=fallback_cwd,
    )


def assert_session_cwd_exists(
    session_manager: SessionCwdSource,
    fallback_cwd: str,
) -> None:
    issue = get_missing_session_cwd_issue(session_manager, fallback_cwd)
    if issue is not None:
        raise MissingSessionCwdError(issue)
