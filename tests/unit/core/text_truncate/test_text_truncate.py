from __future__ import annotations

from agentm.core.text_truncate import format_size, truncate_head, truncate_line, truncate_tail


def test_truncate_head_reports_first_line_exceeding_byte_limit() -> None:
    result = truncate_head("abcdef\nrest", max_lines=10, max_bytes=3)

    assert result.truncated is True
    assert result.first_line_exceeds_limit is True
    assert result.content == ""


def test_truncate_tail_preserves_utf8_boundaries() -> None:
    result = truncate_tail("prefix\nhello😀", max_lines=10, max_bytes=5)

    assert result.truncated is True
    assert result.last_line_partial is True
    assert result.content == "o😀"


def test_truncate_line_adds_marker() -> None:
    text, truncated = truncate_line("abcdefghij", max_chars=8)

    assert truncated is True
    assert text == " ... [tr"


def test_format_size_uses_human_readable_units() -> None:
    assert format_size(12) == "12B"
    assert format_size(2048) == "2.0KB"
