"""argparse-level smoke tests for ``agentm-worker``.

No subprocess, no real socket, no LLM. Mirrors the contract the
terminal/feishu CLIs already enforce in their own smoke suites.
"""

from __future__ import annotations

import pytest

from agentm_worker import cli as worker_cli


def test_help_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        worker_cli._build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "agentm-worker" in out
    assert "--connect" in out
    assert "--scenario" in out
    assert "--cwd" in out
    assert "--max-concurrency" in out
    # Examples section keeps the contract discoverable from --help alone.
    assert "Examples" in out


def test_version_prints_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        worker_cli._build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "0.1.0" in out


def test_missing_connect_uses_default_socket(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # --connect now defaults to the conventional gateway socket; with no
    # gateway running the CLI returns exit 7 (connect-failed), not 2.
    rc = worker_cli.main([])
    assert rc == 7
    err = capsys.readouterr().err
    assert "connect-failed" in err


def test_bad_connect_scheme_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = worker_cli.main(["--connect", "tcp://127.0.0.1:7000"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "agentm-worker: error" in err
    assert "unix://" in err


def test_empty_connect_path_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # ``unix://`` with no netloc and no path → bad-argument.
    rc = worker_cli.main(["--connect", "unix://"])
    assert rc == 2
