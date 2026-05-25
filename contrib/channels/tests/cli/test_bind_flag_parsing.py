"""``--bind`` flag parsing + ``--check`` JSON output."""

from __future__ import annotations


import pytest
from typer.testing import CliRunner

from agentm_channels.cli import _resolve_bind

runner = CliRunner()




def test_resolve_bind_tcp_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        _resolve_bind(
            bind="tcp://127.0.0.1:7000",
            bind_allow_uid=None,
            bind_allow_any_uid=False,
            cfg={},
        )
    assert "unix://" in str(exc.value)
















