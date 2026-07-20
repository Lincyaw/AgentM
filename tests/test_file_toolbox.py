from __future__ import annotations

from agentm_toolbox import FileToolbox


def test_edit_allows_line_range_after_covering_partial_read() -> None:
    toolbox = FileToolbox(cwd="/workspace")
    original = b"one\ntwo\nthree\n"

    read_result = toolbox.read_bytes("note.txt", original, offset=2, limit=1)
    assert not read_result.is_error

    result, updated = toolbox.plan_edit(
        "note.txt",
        original,
        start_line=2,
        end_line=2,
        new_string="TWO",
    )

    assert not result.is_error
    assert updated == b"one\nTWO\nthree\n"


def test_edit_rejects_line_range_not_covered_by_partial_read() -> None:
    toolbox = FileToolbox(cwd="/workspace")
    original = b"one\ntwo\nthree\n"

    toolbox.read_bytes("note.txt", original, offset=2, limit=1)
    result, updated = toolbox.plan_edit(
        "note.txt",
        original,
        start_line=3,
        end_line=3,
        new_string="THREE",
    )

    assert result.is_error
    assert updated is None
    assert "line 3" in result.text


def test_edit_allows_range_covered_by_multiple_partial_reads() -> None:
    toolbox = FileToolbox(cwd="/workspace")
    original = b"one\ntwo\nthree\nfour\n"

    toolbox.read_bytes("note.txt", original, offset=2, limit=1)
    toolbox.read_bytes("note.txt", original, offset=3, limit=1)
    result, updated = toolbox.plan_edit(
        "note.txt",
        original,
        start_line=2,
        end_line=3,
        new_string="middle",
    )

    assert not result.is_error
    assert updated == b"one\nmiddle\nfour\n"


def test_edit_rejects_partial_read_after_file_changes() -> None:
    toolbox = FileToolbox(cwd="/workspace")
    original = b"one\ntwo\nthree\n"
    changed = b"ONE\ntwo\nthree\n"

    toolbox.read_bytes("note.txt", original, offset=2, limit=1)
    result, updated = toolbox.plan_edit(
        "note.txt",
        changed,
        start_line=2,
        end_line=2,
        new_string="TWO",
    )

    assert result.is_error
    assert updated is None
    assert "modified since you last read it" in result.text


def test_edit_allows_string_match_after_covering_partial_read() -> None:
    toolbox = FileToolbox(cwd="/workspace")
    original = b"alpha\nneedle\nomega\n"

    toolbox.read_bytes("note.txt", original, offset=2, limit=1)
    result, updated = toolbox.plan_edit(
        "note.txt",
        original,
        old_string="needle",
        new_string="replacement",
    )

    assert not result.is_error
    assert updated == b"alpha\nreplacement\nomega\n"


def test_partial_edit_does_not_mark_unread_lines_as_read() -> None:
    toolbox = FileToolbox(cwd="/workspace")
    original = "\n".join(f"line-{index}" for index in range(1, 21)).encode()

    toolbox.read_bytes("note.txt", original, offset=10, limit=1)
    result, updated = toolbox.plan_edit(
        "note.txt",
        original,
        start_line=10,
        end_line=10,
        new_string="TWO",
    )
    assert not result.is_error
    assert updated is not None
    toolbox.accept_content("note.txt", updated, read_ranges=result.read_ranges)

    second_result, second_updated = toolbox.plan_edit(
        "note.txt",
        updated,
        start_line=20,
        end_line=20,
        new_string="THREE",
    )

    assert second_result.is_error
    assert second_updated is None
    assert "line 20" in second_result.text
