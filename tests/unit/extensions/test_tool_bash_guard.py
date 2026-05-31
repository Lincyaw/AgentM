"""Unit tests for the tool_bash_guard extension's pattern matching."""

from __future__ import annotations

import re

from agentm.extensions.builtin.tool_bash_guard import _BLOCKED


def _is_blocked(cmd: str) -> str | None:
    for pattern, reason in _BLOCKED:
        if pattern.search(cmd):
            return reason
    return None


def test_blocks_sed_i():
    assert _is_blocked("sed -i 's/foo/bar/' file.py") is not None


def test_blocks_sed_i_with_backup():
    assert _is_blocked("sed -i.bak 's/foo/bar/' file.py") is not None


def test_blocks_sed_in_place():
    assert _is_blocked("sed --in-place 's/foo/bar/' file.py") is not None


def test_blocks_awk_inplace():
    assert _is_blocked("awk -i inplace '{print}' file.py") is not None


def test_allows_regular_sed():
    assert _is_blocked("sed 's/foo/bar/' file.py") is None


def test_allows_grep():
    assert _is_blocked("grep -r 'pattern' .") is None


def test_allows_sed_in_pipe():
    assert _is_blocked("cat file | sed 's/foo/bar/'") is None


def test_allows_ls():
    assert _is_blocked("ls -la") is None
