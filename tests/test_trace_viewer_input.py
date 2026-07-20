from __future__ import annotations

import os

from agentm.cli import _trace_viewer as viewer


def _read_key_from_bytes(data: bytes) -> str:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, data)
        os.close(write_fd)
        write_fd = -1
        return viewer._read_key(read_fd)
    finally:
        os.close(read_fd)
        if write_fd != -1:
            os.close(write_fd)


def test_trace_viewer_reads_supported_key_sequences() -> None:
    cases = {
        b"\x1b": viewer._KEY_ESC,
        b"\x1b[A": viewer._KEY_UP,
        b"\x1b[B": viewer._KEY_DOWN,
        b"\x1b[5~": viewer._KEY_PAGE_UP,
        b"\x1b[6~": viewer._KEY_PAGE_DOWN,
        b"\x1b[<64;10;20M": viewer._KEY_SCROLL_UP,
        b"\x1b[<65;10;20M": viewer._KEY_SCROLL_DOWN,
        b"\x1b[<0;10;20M": viewer._KEY_IGNORE,
        b"\x1b[?1006h": viewer._KEY_IGNORE,
    }

    for data, expected in cases.items():
        assert _read_key_from_bytes(data) == expected


def test_trace_viewer_mouse_wheel_scrolls_expanded_viewport() -> None:
    tui = viewer.TraceViewer.__new__(viewer.TraceViewer)
    tui._views = []
    tui._cursor = 0
    tui._items = [
        viewer._ViewItem(
            msg=viewer._Message(role="assistant", content=""),
            body_lines=[f"line {i}" for i in range(30)],
            expanded=True,
        )
    ]
    tui._scroll = 0
    tui._manual_scroll = False
    tui._term_size = lambda: (80, 10)

    tui._handle_expanded_key(viewer._KEY_SCROLL_DOWN)
    assert tui._scroll == viewer._WHEEL_SCROLL_LINES
    assert tui._manual_scroll is True

    tui._handle_expanded_key(viewer._KEY_SCROLL_UP)
    assert tui._scroll == 0
