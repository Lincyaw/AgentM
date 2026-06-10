"""Prefix-replay from an auditor verdict.

Given a recorded auditor :class:`~llmharness.replay.record.ReplayRecord`
that surfaced a reminder at turn ``t``, materialise a new main-agent
session whose entry-tree mirrors the original up to turn ``t`` and is
ready to be resumed with the reminder seeded via
:mod:`llmharness.replay.reminder_seed`.

Why this lives next to the replay CLI: the operation is a developer
iteration tool for the auditor / reminder loop — we deliberately keep
all the orchestration outside ``agentm.core``. The only core surface we
touch is :meth:`SessionManager.create_branched_session`, which already
implements persisted truncation + entry-id rewrite + parent-session
linking.

Public entry points used by ``cli.py``:

* :func:`materialise_branched_session` — open the source JSONL, pick the
  leaf entry that ends turn ``t``, branch.
* :func:`build_prefix_replay_command` — compose the ``agentm`` command
  shape that resumes the branched session with the reminder seed atom
  mounted (and the live adapter's reminder leg disabled so the seed is
  the only reminder source).
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path

from agentm.core.abi.messages import AssistantMessage
from agentm.core.abi.session import ENTRY_TYPE_MESSAGE, SessionEntry
from agentm.core.runtime.session_manager import SessionManager

from llmharness.replay.record import ReplayRecord, iter_records


class PrefixReplayError(RuntimeError):
    """Domain error raised by helpers in this module.

    Wraps "no matching record" / "leaf not findable" / "session-file not
    found" failures so the CLI can map them to ``typer.BadParameter``
    uniformly without leaking ``RuntimeError`` to users.
    """


# --- record selection -------------------------------------------------------


def pick_auditor_reminder_record(audit_replay_path: Path, *, turn: int) -> ReplayRecord:
    """Return the auditor record at ``turn`` that carries a reminder.

    Selects the LAST matching auditor record (mirroring the existing
    ``_pick_record`` convention in :mod:`llmharness.replay.cli`). Raises
    :class:`PrefixReplayError` if no record matches or the matched
    record has no ``surface_reminder`` payload.
    """
    latest: ReplayRecord | None = None
    for rec in iter_records(audit_replay_path):
        if rec.phase == "auditor" and rec.turn_index == turn:
            latest = rec
    if latest is None:
        raise PrefixReplayError(f"no auditor record with turn_index={turn} in {audit_replay_path}")
    reminder_text = _extract_reminder_text(latest)
    if not reminder_text:
        raise PrefixReplayError(
            f"auditor record at turn={turn} did not surface a reminder "
            f"(output.surface_reminder missing or false)"
        )
    return latest


def _extract_reminder_text(record: ReplayRecord) -> str | None:
    output = record.output
    if not isinstance(output, dict):
        return None
    if not output.get("surface_reminder"):
        return None
    text = output.get("reminder_text")
    if not isinstance(text, str) or not text.strip():
        # The brief calls the field ``surface_reminder`` (boolean +
        # text); the Verdict.to_dict shape stores the text under
        # ``reminder_text``. Fall back to a string-typed
        # ``surface_reminder`` for defensive symmetry with hand-edited
        # records, but the canonical key is ``reminder_text``.
        sr = output.get("surface_reminder")
        if isinstance(sr, str) and sr.strip():
            return sr
        return None
    return text


# --- session location -------------------------------------------------------


def locate_source_session_file(*, session_dir: Path, session_id: str) -> Path:
    """Find the session JSONL whose id matches ``session_id``.

    Matches the ``JsonlSessionStore.open`` convention. Post single-event-
    log merge (.claude/designs/single-event-log.md) files are named
    ``<session_id>.jsonl``; the trailing glob still resolves legacy
    ``<timestamp>_<id>.jsonl`` artefacts for sessions written before the
    rename. Raises :class:`PrefixReplayError` if no match.
    """

    direct = session_dir / f"{session_id}.jsonl"
    if direct.is_file():
        return direct
    matches = sorted(session_dir.glob(f"*_{session_id}.jsonl"))
    if not matches:
        raise PrefixReplayError(f"no session file matching {session_id} under {session_dir}")
    return matches[0]


# --- leaf-entry selection ---------------------------------------------------


def find_leaf_entry_for_turn(manager: SessionManager, *, turn: int) -> SessionEntry:
    """Pick the entry that ends turn ``turn`` on the active branch.

    Counting rule matches :mod:`llmharness.atom`'s
    ``turn_index = len(messages) - 1``: the active branch's
    materialised message stream is the trajectory the audit pipeline
    saw, and the ``turn``-th message (0-indexed) is the boundary.

    We walk ``get_active_branch`` and pick the ``turn``-th ``message``
    entry. Non-message entries (audit_event, audit_edge, ... ) are
    skipped by the counter but copied into the branched session by
    ``create_branched_session`` — exactly what we want, the audit
    context up to that point is part of the prefix.
    """
    branch = manager.get_active_branch()
    if not branch:
        raise PrefixReplayError("source session has no entries on its active branch")
    message_count = 0
    for entry in branch:
        if entry.type != ENTRY_TYPE_MESSAGE:
            continue
        if message_count == turn:
            return entry
        message_count += 1
    raise PrefixReplayError(
        f"branch has only {message_count} message entries; cannot end turn {turn}"
    )


# --- command composition ----------------------------------------------------


@dataclass(frozen=True)
class PrefixReplayPlan:
    """Materialised plan: branched session on disk + command to run."""

    source_session_file: Path
    branched_session_file: Path
    branched_session_id: str
    reminder_text: str
    command: str


def materialise_branched_session(
    *,
    audit_replay_path: Path,
    turn: int,
    session_dir: Path,
) -> tuple[ReplayRecord, SessionManager, str]:
    """Pick record, open source session, branch at end of turn ``t``.

    Returns ``(record, branched_manager, leaf_entry_id)``. The branched
    manager's underlying file is already persisted on disk by
    :meth:`SessionManager.create_branched_session`. The caller still
    holds the source manager indirectly via the returned record's
    metadata if needed; for the CLI flow we only care about the branched
    side.
    """
    record = pick_auditor_reminder_record(audit_replay_path, turn=turn)
    source_file = locate_source_session_file(session_dir=session_dir, session_id=record.session_id)
    source_mgr = SessionManager.open(source_file)
    leaf_entry = find_leaf_entry_for_turn(source_mgr, turn=turn)
    branched_path = source_mgr.create_branched_session(leaf_entry.id)
    if branched_path is None:
        raise PrefixReplayError(
            "create_branched_session returned None — source session is not persisted"
        )
    branched_mgr = SessionManager.open(branched_path)
    return record, branched_mgr, leaf_entry.id


def build_prefix_replay_command(
    *,
    cwd: str,
    branched_session_id: str,
    reminder_text: str,
) -> str:
    """Compose the ``agentm`` command that resumes the branched session.

    The agentm CLI uses the unified ``--extension MODULE[:JSON]`` form
    (see ``src/agentm/cli.py::_parse_extensions``); there is no separate
    ``--extension-config`` flag. Each extension config travels inline as
    a JSON object after the first colon.

    ``enable_reminders:false`` + ``enable_auditor:true`` keeps the live
    auditor observing (verdicts still get persisted as evidence) while
    blocking a second reminder from masking the seed.
    """
    adapter_cfg = json.dumps(
        {"enable_reminders": False, "enable_auditor": True},
        separators=(",", ":"),
    )
    seed_cfg = json.dumps({"text": reminder_text}, separators=(",", ":"))
    parts = [
        "agentm",
        "--cwd",
        shlex.quote(cwd),
        "--resume",
        shlex.quote(branched_session_id),
        "--extension",
        shlex.quote(f"llmharness.atom:{adapter_cfg}"),
        "--extension",
        shlex.quote(f"llmharness.replay.reminder_seed:{seed_cfg}"),
    ]
    return " ".join(parts)


def make_plan(
    *,
    audit_replay_path: Path,
    turn: int,
    session_dir: Path,
) -> PrefixReplayPlan:
    """End-to-end: materialise + compose command. Used by the CLI."""
    record, branched_mgr, _leaf_id = materialise_branched_session(
        audit_replay_path=audit_replay_path,
        turn=turn,
        session_dir=session_dir,
    )
    branched_file_str = branched_mgr.get_session_file()
    if branched_file_str is None:
        raise PrefixReplayError("branched session has no session_file")
    branched_file = Path(branched_file_str)
    reminder_text = _extract_reminder_text(record) or ""
    cwd = branched_mgr.get_cwd()
    command = build_prefix_replay_command(
        cwd=cwd,
        branched_session_id=branched_mgr.get_session_id(),
        reminder_text=reminder_text,
    )
    source_file = locate_source_session_file(session_dir=session_dir, session_id=record.session_id)
    return PrefixReplayPlan(
        source_session_file=source_file,
        branched_session_file=branched_file,
        branched_session_id=branched_mgr.get_session_id(),
        reminder_text=reminder_text,
        command=command,
    )


# Re-export for convenience to keep ``cli.py`` imports tidy.
def assistant_message_count(branch: list[SessionEntry]) -> int:
    """Count assistant messages on ``branch`` — used by tests / introspection."""
    n = 0
    for entry in branch:
        if entry.type != ENTRY_TYPE_MESSAGE:
            continue
        if isinstance(entry.payload, AssistantMessage):
            n += 1
    return n


__all__ = [
    "PrefixReplayError",
    "PrefixReplayPlan",
    "assistant_message_count",
    "build_prefix_replay_command",
    "find_leaf_entry_for_turn",
    "locate_source_session_file",
    "make_plan",
    "materialise_branched_session",
    "pick_auditor_reminder_record",
]
