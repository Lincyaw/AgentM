"""Fail-stop: the systemd unit renderer must produce mode-correct, socket-paired
units without ever invoking real ``systemctl`` or writing system dirs.

The render step is a pure function of a resolved :class:`SystemdPlan`, so these
tests assert directly on the rendered text. A regression here means the operator
gets a gateway and a feishu client that disagree on the socket (broken deploy)
or a system unit that runs as the wrong user / target.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentm.gateway import systemd as gw_systemd
from agentm.gateway.systemd import (
    SystemdPlan,
    build_systemd_plan,
    render_feishu_unit,
    render_gateway_unit,
    systemd_action,
)


def _system_plan(*, feishu: bool = True) -> SystemdPlan:
    sock = "unix:///run/agentm/gw.sock"
    return SystemdPlan(
        system=True,
        unit_dir=Path("/etc/systemd/system"),
        socket_url=sock,
        gateway_exec_start=f"/opt/venv/bin/agentm gateway --bind {sock} --cwd /srv/ws",
        env_file=Path("/srv/ws/.env"),
        working_dir=Path("/srv"),
        path_env="/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
        run_as="agent",
        feishu_bin="/opt/venv/bin/agentm-feishu" if feishu else None,
    )


def _user_plan() -> SystemdPlan:
    sock = "unix://%t/agentm/gw.sock"
    return SystemdPlan(
        system=False,
        unit_dir=Path.home() / ".config" / "systemd" / "user",
        socket_url=sock,
        gateway_exec_start=f"/home/me/.venv/bin/agentm gateway --bind {sock}",
        env_file=Path("/home/me/ws/.env"),
        working_dir=Path("/home/me/ws"),
        path_env="/home/me/.venv/bin:/usr/local/bin:/usr/bin:/bin",
        run_as=None,
        feishu_bin="/home/me/.venv/bin/agentm-feishu",
    )


def test_system_mode_unit_pins_run_socket_and_user() -> None:
    plan = _system_plan()
    gw = render_gateway_unit(plan)
    assert "unix:///run/agentm/gw.sock" in gw
    assert "RuntimeDirectory=agentm" in gw
    assert "RuntimeDirectoryMode=0750" in gw
    assert "User=agent" in gw
    assert "Group=agent" in gw
    assert "WantedBy=multi-user.target" in gw


def test_user_mode_unit_uses_runtime_specifier_and_no_user() -> None:
    plan = _user_plan()
    gw = render_gateway_unit(plan)
    assert "unix://%t/agentm/gw.sock" in gw
    assert "WantedBy=default.target" in gw
    assert "User=" not in gw
    # RuntimeDirectory=agentm is required even for user units so systemd
    # creates %t/agentm/ before the gateway binds its socket there.
    assert "RuntimeDirectory=agentm" in gw


def test_gateway_and_feishu_share_the_same_socket() -> None:
    for plan in (_system_plan(), _user_plan()):
        gw = render_gateway_unit(plan)
        feishu = render_feishu_unit(plan)
        assert plan.socket_url in gw
        assert f"--connect {plan.socket_url}" in feishu


def test_feishu_unit_is_loosely_coupled_to_gateway() -> None:
    feishu = render_feishu_unit(_system_plan())
    assert "After=agentm-gateway.service" in feishu
    assert "Wants=agentm-gateway.service" in feishu
    assert "Requires=" not in feishu  # loose coupling: no restart cascade
    assert "--scenario" not in feishu  # defaults to chatbot in code


def test_units_carry_environmentfile_and_path() -> None:
    plan = _system_plan()
    gw = render_gateway_unit(plan)
    feishu = render_feishu_unit(plan)
    assert "EnvironmentFile=-/srv/ws/.env" in gw
    assert "EnvironmentFile=-/srv/ws/.env" in feishu
    assert f"Environment=PATH={plan.path_env}" in gw


def test_install_skips_feishu_unit_when_bin_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``agentm-feishu`` can't be resolved, the gateway unit is still
    written but the feishu unit is skipped (no broken half-deploy). systemctl
    is stubbed — this never touches real systemd."""
    unit_dir = tmp_path / "units"
    plan = SystemdPlan(
        system=False,
        unit_dir=unit_dir,
        socket_url="unix://%t/agentm/gw.sock",
        gateway_exec_start="/bin/agentm gateway --bind unix://%t/agentm/gw.sock",
        env_file=tmp_path / ".env",
        working_dir=tmp_path,
        path_env="/bin:/usr/bin",
        run_as=None,
        feishu_bin=None,
    )
    monkeypatch.setattr(gw_systemd, "build_systemd_plan", lambda: plan)
    calls: list[list[str]] = []

    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", lambda *a, **k: calls.append(list(a[0])))

    systemd_action(install=True)

    assert (unit_dir / "agentm-gateway.service").exists()
    assert not (unit_dir / "agentm-feishu.service").exists()
    # enable only the gateway unit; feishu absent from the enable call.
    enable_calls = [c for c in calls if "enable" in c]
    assert enable_calls, "expected an enable systemctl call"
    assert all("agentm-feishu.service" not in c for c in enable_calls)
    assert any("agentm-gateway.service" in c for c in enable_calls)


def test_build_plan_bakes_cwd_into_exec_and_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--cwd must survive into the gateway ExecStart, WorkingDirectory, and the
    EnvironmentFile, all on the same workspace. _baked_gateway_argv() strips
    --cwd to derive the workspace, so build_systemd_plan must re-add it — the
    gateway resolves its session cwd + state-dir from --cwd, not from the unit's
    WorkingDirectory."""
    import os
    import shutil
    import sys

    ws = "/srv/agent workspace"  # a space exercises the shlex.quote path too
    monkeypatch.setattr(
        sys, "argv", ["agentm gateway", "--cwd", ws, "--scenario", "chatbot"]
    )
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(shutil, "which", lambda name: f"/venv/bin/{name}")

    plan = build_systemd_plan()

    # --cwd round-trips into ExecStart (quoted because of the space).
    assert "--cwd" in plan.gateway_exec_start
    assert "'/srv/agent workspace'" in plan.gateway_exec_start
    # WorkingDirectory and EnvironmentFile agree on the same workspace.
    assert plan.working_dir == Path(ws)
    assert plan.env_file == Path(ws) / ".env"
    # The rendered gateway unit references <ws>/.env.
    gw = render_gateway_unit(plan)
    assert f"EnvironmentFile=-{Path(ws) / '.env'}" in gw
    # The forced --bind socket survives and pairs with feishu.
    assert plan.socket_url in plan.gateway_exec_start


def test_build_plan_defaults_workspace_to_cwd_when_no_cwd_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no --cwd, the workspace falls back to the current dir (unchanged
    behavior), and --cwd is still baked so ExecStart is explicit."""
    import os
    import shutil
    import sys

    monkeypatch.setattr(sys, "argv", ["agentm gateway"])
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(shutil, "which", lambda name: f"/venv/bin/{name}")
    monkeypatch.chdir(tmp_path)

    plan = build_systemd_plan()

    assert plan.working_dir == tmp_path
    assert plan.env_file == tmp_path / ".env"
    assert f"--cwd {tmp_path}" in plan.gateway_exec_start


def test_build_plan_absolutizes_relative_cwd_and_stays_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A relative --cwd must be absolutized — systemd rejects a relative
    WorkingDirectory ('bad unit file setting'). And even as root we install
    USER units, never system units."""
    import os
    import shutil
    import sys

    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agentm gateway", "--cwd", "work"])
    monkeypatch.setattr(os, "geteuid", lambda: 0)  # even root → user units
    monkeypatch.setattr(shutil, "which", lambda name: f"/venv/bin/{name}")

    plan = build_systemd_plan()

    assert not plan.system  # user units only, even as root
    assert plan.working_dir.is_absolute()
    assert plan.working_dir == (tmp_path / "work").resolve()
    assert plan.run_as is None
    gw = render_gateway_unit(plan)
    # The WorkingDirectory line is absolute (no relative path → no bad setting).
    assert f"WorkingDirectory={(tmp_path / 'work').resolve()}" in gw
    assert "WantedBy=default.target" in gw  # user target, not multi-user
    # RuntimeDirectory must be present even for user units, or systemd won't
    # create %t/agentm/ and the gateway's socket bind() fails (exit 1).
    assert "RuntimeDirectory=agentm" in gw
