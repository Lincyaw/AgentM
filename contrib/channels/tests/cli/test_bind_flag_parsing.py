"""``--bind`` flag parsing + ``--check`` JSON output."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from agentm_channels.cli import _parse_args, _resolve_bind


def _ns(**kw: object) -> argparse.Namespace:
    """Build a minimal Namespace defaulting all bind-related fields."""
    return argparse.Namespace(
        bind=kw.get("bind"),
        bind_allow_uid=kw.get("bind_allow_uid"),
        bind_allow_any_uid=kw.get("bind_allow_any_uid", False),
    )


def test_resolve_bind_none_when_unset() -> None:
    spec = _resolve_bind(_ns(), cfg={})
    assert spec is None


def test_resolve_bind_tcp_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        _resolve_bind(_ns(bind="tcp://127.0.0.1:7000"), cfg={})
    assert "unix://" in str(exc.value)


def test_resolve_bind_mutually_exclusive_flags() -> None:
    with pytest.raises(SystemExit) as exc:
        _resolve_bind(
            _ns(
                bind="unix:///tmp/x.sock",
                bind_allow_uid=[1000],
                bind_allow_any_uid=True,
            ),
            cfg={},
        )
    assert "mutually exclusive" in str(exc.value)


def test_resolve_bind_default_uid_is_current_process() -> None:
    spec = _resolve_bind(_ns(bind="unix:///tmp/x.sock"), cfg={})
    assert spec is not None
    assert spec.allow_uids == frozenset({os.geteuid()})
    assert spec.socket_path == "/tmp/x.sock"


def test_resolve_bind_any_uid_means_None() -> None:
    spec = _resolve_bind(
        _ns(bind="unix:///tmp/x.sock", bind_allow_any_uid=True), cfg={}
    )
    assert spec is not None
    assert spec.allow_uids is None


def test_resolve_bind_explicit_uids() -> None:
    spec = _resolve_bind(
        _ns(bind="unix:///tmp/x.sock", bind_allow_uid=[1000, 1001]), cfg={}
    )
    assert spec is not None
    assert spec.allow_uids == frozenset({1000, 1001})


def test_resolve_bind_yaml_socket_and_allow_uids() -> None:
    spec = _resolve_bind(
        _ns(),
        cfg={"bind": {"socket": "unix:///var/run/x.sock", "allow_uids": [42]}},
    )
    assert spec is not None
    assert spec.socket_path == "/var/run/x.sock"
    assert spec.allow_uids == frozenset({42})


def test_resolve_bind_cli_overrides_yaml() -> None:
    spec = _resolve_bind(
        _ns(bind="unix:///cli.sock"),
        cfg={"bind": {"socket": "unix:///yaml.sock", "allow_uids": [42]}},
    )
    assert spec is not None
    assert spec.socket_path == "/cli.sock"
    # CLI overrides the URL; uid policy not set on CLI side, so the
    # YAML side wins (fine-grained precedence per axis).
    assert spec.allow_uids == frozenset({42})


def test_parse_args_appends_uids() -> None:
    ns = _parse_args(
        [
            "--bind",
            "unix:///tmp/x.sock",
            "--bind-allow-uid",
            "1000",
            "--bind-allow-uid",
            "1001",
        ]
    )
    assert ns.bind == "unix:///tmp/x.sock"
    assert ns.bind_allow_uid == [1000, 1001]
    assert ns.bind_allow_any_uid is False


def test_check_emits_json_to_stdout() -> None:
    """End-to-end: ``agentm-gateway --check --bind unix://...`` prints
    a JSON object on stdout describing the resolved config.
    """
    with tempfile.TemporaryDirectory(prefix="agentm-check-") as d:
        cfg_path = Path(d) / "gw.yaml"
        cfg_path.write_text(
            textwrap.dedent(
                f"""
                cwd: {d}
                channels:
                  stub:
                    enabled: true
                    allow_from: ['*']
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        sock = Path(d) / "g.sock"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "agentm_channels.cli",
                "--config",
                str(cfg_path),
                "--bind",
                f"unix://{sock}",
                "--bind-allow-any-uid",
                "--check",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        # Find the JSON payload line (logs go to stderr; stdout is one line).
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        assert payload["kind"] == "check"
        assert payload["bind"]["socket"] == f"unix://{sock}"
        assert payload["bind"]["allow_uids"] is None
        assert "stub" in payload["channels"]


def test_check_rejects_tcp_bind() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentm_channels.cli",
            "--bind",
            "tcp://127.0.0.1:7000",
            "--check",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 2
