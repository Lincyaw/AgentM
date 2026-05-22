"""Fail-stop tests for :func:`llmharness.replay.write_strict_ab_replay`.

Why this is load-bearing: the strict-A/B sidecar is the only artifact
the case viewer and the downstream distillation pipeline read to compare
a control trajectory (no auditor) against a branch trajectory that was
seeded with the surfaced reminder. If the stitching contract drifts —
wrong cut point, wrong root_session_id, branch tail dropped — every
forked-eval case silently teaches the student model from misaligned
prefixes.

The function is pure (no LLM, no session spawn), so we exercise it
directly with hand-built ``ReplayRecord``s on disk.
"""

from __future__ import annotations

from pathlib import Path

from llmharness import (
    ReminderCandidate,
    ReplayRecord,
    iter_records,
    strict_ab_replay_path,
    write_record,
    write_strict_ab_replay,
)


def _ext(turn: int, root_id: str = "control-sid") -> ReplayRecord:
    return ReplayRecord(
        phase="extractor",
        turn_index=turn,
        root_session_id=root_id,
        ts_ns=1_000_000_000 + turn,
        compose_kwargs={},
        payload={"graph": []},
        provider=None,
        output={"events": [], "edges": []},
        status="ok",
        latency_ms=10,
    )


def _aud(
    turn: int,
    *,
    root_id: str = "control-sid",
    surface: bool = False,
    text: str = "",
) -> ReplayRecord:
    return ReplayRecord(
        phase="auditor",
        turn_index=turn,
        root_session_id=root_id,
        ts_ns=1_000_000_000 + turn + 10_000,
        compose_kwargs={},
        payload={"graph": []},
        provider=None,
        output={"surface_reminder": surface, "reminder_text": text},
        status="ok",
        latency_ms=15,
    )


def _seed_sidecar(path: Path, records: list[ReplayRecord]) -> None:
    for rec in records:
        write_record(path, rec)


def test_strict_ab_stitches_prefix_and_tail_keyed_by_branch(tmp_path: Path) -> None:
    """The output sidecar must contain:

    * control extractor + auditor for turns <= reminder.turn_index, and
    * branch  extractor + auditor for turns >  reminder.turn_index,

    with every record's ``root_session_id`` rewritten to the branch's id.
    This is the fail-stop position — if either side of the cut leaks
    the wrong session id, the case viewer joins records against the
    wrong trajectory and the distillation pair is corrupt.
    """

    control_path = tmp_path / "control.jsonl"
    branch_path = tmp_path / "branch.jsonl"
    _seed_sidecar(
        control_path,
        [_ext(0), _ext(1), _ext(2), _ext(3)],
    )
    _seed_sidecar(
        branch_path,
        [
            _ext(0, root_id="branch-sid"),
            _ext(1, root_id="branch-sid"),
            _ext(2, root_id="branch-sid"),
            _ext(3, root_id="branch-sid"),
            _ext(4, root_id="branch-sid"),
        ],
    )

    offline_auditor_records = [
        _aud(0),
        _aud(1, surface=True, text="check the upstream latency spike"),
        _aud(2),
    ]
    branch_auditor_records = [
        _aud(2, root_id="branch-sid"),
        _aud(3, root_id="branch-sid"),
        _aud(4, root_id="branch-sid"),
    ]
    reminder = ReminderCandidate(
        turn_index=1,
        text="check the upstream latency spike",
        record=offline_auditor_records[1],
    )

    out_path = strict_ab_replay_path(tmp_path, "branch-sid")
    returned = write_strict_ab_replay(
        control_replay_path=control_path,
        branch_replay_path=branch_path,
        branch_session_log_id="branch-sid",
        offline_auditor_records=offline_auditor_records,
        branch_auditor_records=branch_auditor_records,
        reminder=reminder,
        out_path=out_path,
    )

    assert returned == out_path
    assert out_path.exists()
    written = list(iter_records(out_path))

    # Every record must be keyed to the branch session id, regardless of
    # which side of the cut it came from. This is the fail-stop:
    # mis-keyed records misjoin in the case viewer.
    assert all(rec.root_session_id == "branch-sid" for rec in written)

    # Both phases must appear for turns 0..4 — one extractor + one
    # auditor per turn. Turns 0 and 1 come from the control side
    # (offline auditor records); turns 2..4 come from the branch side.
    written_turns_by_phase = {(rec.phase, rec.turn_index) for rec in written}
    for turn in range(5):
        assert ("extractor", turn) in written_turns_by_phase, written_turns_by_phase
        assert ("auditor", turn) in written_turns_by_phase, written_turns_by_phase


def test_strict_ab_drops_control_records_strictly_after_cut(tmp_path: Path) -> None:
    """Control extractor records *after* the reminder turn must not leak
    into the strict-A/B sidecar — the whole point of the branch is to
    replace the post-reminder trajectory."""

    control_path = tmp_path / "control.jsonl"
    branch_path = tmp_path / "branch.jsonl"
    _seed_sidecar(
        control_path,
        [_ext(0), _ext(1), _ext(2), _ext(3)],
    )
    _seed_sidecar(
        branch_path,
        [_ext(t, root_id="branch-sid") for t in range(4)],
    )

    reminder = ReminderCandidate(
        turn_index=1,
        text="...",
        record=_aud(1, surface=True, text="..."),
    )
    out_path = tmp_path / "strict_ab.jsonl"
    write_strict_ab_replay(
        control_replay_path=control_path,
        branch_replay_path=branch_path,
        branch_session_log_id="branch-sid",
        offline_auditor_records=[],
        branch_auditor_records=None,
        reminder=reminder,
        out_path=out_path,
    )

    written = list(iter_records(out_path))
    # Control records carry ts_ns close to 1_000_000_000 + turn;
    # branch records also do (we used same offset). Distinguish by the
    # rebinding: control records exist in the file only because they
    # are < reminder turn. We assert by turn_index since both sides use
    # the same ts.
    ext_turns = [rec.turn_index for rec in written if rec.phase == "extractor"]
    # Control side contributes turns 0, 1; branch side contributes
    # turns 2, 3. The full extractor stream should therefore be
    # exactly [0, 1, 2, 3] with no duplicates and no control turn > 1.
    assert ext_turns == [0, 1, 2, 3]


def test_strict_ab_no_branch_tail_means_no_branch_records(tmp_path: Path) -> None:
    """When the branch produces no post-reminder extractor records,
    the sidecar must carry only the control prefix. The previous
    'dump branch verbatim as a fallback' behavior duplicated
    turn_index keys with the control prefix and silently corrupted
    downstream ``{turn_index: record}`` views (the eval driver builds
    one in ``_run_baseline_fork``).

    Fail-stop: every extractor turn in the strict-A/B sidecar must
    appear exactly once, and every branch-side record must have
    ``turn_index > reminder.turn_index``."""

    control_path = tmp_path / "control.jsonl"
    branch_path = tmp_path / "branch.jsonl"
    _seed_sidecar(control_path, [_ext(0), _ext(1)])
    # Branch only produced an extractor at turn 0 — no turn-2 tail.
    _seed_sidecar(branch_path, [_ext(0, root_id="branch-sid")])

    reminder = ReminderCandidate(
        turn_index=1,
        text="...",
        record=_aud(1, surface=True, text="..."),
    )
    out_path = tmp_path / "strict_ab.jsonl"
    write_strict_ab_replay(
        control_replay_path=control_path,
        branch_replay_path=branch_path,
        branch_session_log_id="branch-sid",
        offline_auditor_records=[],
        branch_auditor_records=None,
        reminder=reminder,
        out_path=out_path,
    )
    written = list(iter_records(out_path))
    ext_turns = [rec.turn_index for rec in written if rec.phase == "extractor"]

    # Exactly the control prefix turns 0 and 1, no duplicates.
    assert ext_turns == [0, 1]

    # And there must be no record with turn_index <= reminder.turn_index
    # that was sourced from the branch — that would collide with the
    # control prefix's turn-keyed records.
    assert all(rec.turn_index <= reminder.turn_index for rec in written), (
        f"branch tail leaked records with turn_index > {reminder.turn_index}: "
        f"{[(r.phase, r.turn_index) for r in written]}"
    )


def test_strict_ab_replay_path_layout(tmp_path: Path) -> None:
    """Path convention: ``<cwd>/.agentm/audit_replay/<id>.strict_ab.jsonl``.

    The case viewer derives this path from the branch's
    ``audit_replay_path`` metadata. If the suffix / directory drift, the
    viewer falls back to the regular (non-strict) sidecar silently.
    """

    path = strict_ab_replay_path(tmp_path, "abc123")
    assert path == tmp_path / ".agentm" / "audit_replay" / "abc123.strict_ab.jsonl"
