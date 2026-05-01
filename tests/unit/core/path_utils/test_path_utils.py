from __future__ import annotations

import os
import unicodedata

from agentm.core.path_utils import expand_path, resolve_read_path, resolve_to_cwd


def test_expand_path_strips_at_prefix_and_unicode_spaces(tmp_path) -> None:
    value = expand_path("@foo\u00a0bar")

    assert value == "foo bar"
    assert resolve_to_cwd("notes.txt", str(tmp_path)) == os.path.join(str(tmp_path), "notes.txt")


def test_resolve_read_path_tries_nfd_and_curly_quote_variants(tmp_path) -> None:
    filename = unicodedata.normalize("NFD", "Capture d’écran 10.00.00 AM.png")
    path = tmp_path / filename
    path.write_text("x", encoding="utf-8")

    resolved = resolve_read_path("Capture d'écran 10.00.00 AM.png", str(tmp_path))

    assert resolved == str(path)
