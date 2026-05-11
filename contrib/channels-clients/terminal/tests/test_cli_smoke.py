"""argparse-level smoke tests — no subprocess, no real socket."""

from __future__ import annotations

import io

import pytest

from agentm_terminal import cli as terminal_cli


def test_help_exits_zero(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit) as exc:
        terminal_cli._build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "agentm-terminal" in out
    assert "--connect" in out
    # Examples section is mandated by the brief — keeps the contract
    # discoverable from --help alone.
    assert "Examples" in out


def test_version_prints_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        terminal_cli._build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "0.1.0" in out


def test_bad_connect_scheme_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = terminal_cli.main(["--connect", "tcp://127.0.0.1:7000"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "agentm-terminal: error" in err
    assert "unix://" in err


def test_missing_connect_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = terminal_cli.main([])
    assert rc == 2


def test_resolve_format_defaults_text_for_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdout", _FakeStream(isatty=True))
    assert terminal_cli._resolve_format(None) == "text"


def test_resolve_format_defaults_json_for_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdout", _FakeStream(isatty=False))
    assert terminal_cli._resolve_format(None) == "json"


class _FakeStream(io.StringIO):
    def __init__(self, *, isatty: bool) -> None:
        super().__init__()
        self._isatty = isatty

    def isatty(self) -> bool:  # type: ignore[override]
        return self._isatty
