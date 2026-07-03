"""Systemd unit rendering and installation for the gateway + Feishu client.

Pure-function render step (resolved inputs -> unit text) so tests can assert
on the rendered strings without touching real systemd or system dirs; the
install step writes the files and drives ``systemctl``.

Extracted from ``gateway.cli`` — zero dependency on the gateway runtime or
Typer CLI layer.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from agentm.env import resolve_cli_cwd

GATEWAY_UNIT = "agentm-gateway"
FEISHU_UNIT = "agentm-feishu"


@dataclass(frozen=True, slots=True)
class SystemdPlan:
    """Everything the render/install steps need, fully resolved up front.

    ``system`` selects system units (``/etc/systemd/system``, ``User=``,
    ``/run/agentm`` socket) vs user units (``~/.config/systemd/user``,
    ``%t/agentm`` socket, no ``User=``). ``feishu_bin`` is ``None`` when the
    ``agentm-feishu`` entry point cannot be resolved — the feishu unit is
    then skipped (the gateway unit is still rendered/installed).
    """

    system: bool
    unit_dir: Path
    socket_url: str
    gateway_exec_start: str
    env_file: Path
    working_dir: Path
    path_env: str
    run_as: str | None
    feishu_bin: str | None


def render_gateway_unit(plan: SystemdPlan) -> str:
    import textwrap

    user_lines = ""
    if plan.system and plan.run_as:
        user_lines = f"User={plan.run_as}\nGroup={plan.run_as}\n"
    runtime_lines = "RuntimeDirectory=agentm\nRuntimeDirectoryMode=0750\n"
    wanted_by = "multi-user.target" if plan.system else "default.target"
    return textwrap.dedent(f"""\
        [Unit]
        Description=AgentM gateway (single-process chat session host)
        After=network-online.target
        Wants=network-online.target

        [Service]
        Type=simple
        {user_lines}WorkingDirectory={plan.working_dir}
        EnvironmentFile=-{plan.env_file}
        {runtime_lines}Environment=AGENTM_SOCKET={plan.socket_url}
        Environment=PATH={plan.path_env}
        ExecStart={plan.gateway_exec_start}
        Restart=always
        RestartSec=2
        TimeoutStopSec=10

        [Install]
        WantedBy={wanted_by}
    """)


def render_feishu_unit(plan: SystemdPlan) -> str:
    import shlex
    import textwrap

    assert plan.feishu_bin is not None
    user_lines = ""
    if plan.system and plan.run_as:
        user_lines = f"User={plan.run_as}\nGroup={plan.run_as}\n"
    wanted_by = "multi-user.target" if plan.system else "default.target"
    exec_start = " ".join(
        shlex.quote(tok)
        for tok in [plan.feishu_bin, "--connect", plan.socket_url]
    )
    return textwrap.dedent(f"""\
        [Unit]
        Description=AgentM Feishu/Lark chat client
        After={GATEWAY_UNIT}.service
        Wants={GATEWAY_UNIT}.service

        [Service]
        Type=simple
        {user_lines}WorkingDirectory={plan.working_dir}
        EnvironmentFile=-{plan.env_file}
        Environment=PATH={plan.path_env}
        ExecStart={exec_start}
        Restart=always
        RestartSec=3
        TimeoutStopSec=10

        [Install]
        WantedBy={wanted_by}
    """)


def resolve_feishu_bin(agentm_bin: str) -> str | None:
    """Find ``agentm-feishu``: $PATH first, else next to the ``agentm`` bin."""
    import shutil

    found = shutil.which("agentm-feishu")
    if found:
        return found
    real_bin = Path(agentm_bin).resolve()
    sibling = real_bin.parent / "agentm-feishu"
    return str(sibling) if sibling.exists() else None


def baked_gateway_argv() -> tuple[list[str], str]:
    """Return (gateway-flags, workspace) from the current invocation.

    Strips the systemd flags. ``sys.argv[0]`` is ``"agentm gateway"`` (merged
    by the main CLI dispatcher); the rest are the gateway flags. The workspace
    is the ``--cwd`` value if present, else ``AGENTM_CWD`` / the current dir.
    """
    cleaned: list[str] = []
    workspace = str(resolve_cli_cwd())
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--install-systemd", "--uninstall-systemd"):
            i += 1
            continue
        if arg == "--cwd":
            if i + 1 < len(argv):
                workspace = str(resolve_cli_cwd(argv[i + 1]))
                i += 2
            else:
                i += 1
            continue
        if arg.startswith("--cwd="):
            workspace = str(resolve_cli_cwd(arg.split("=", 1)[1]))
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned, workspace


def build_systemd_plan() -> SystemdPlan:
    """Resolve a full :class:`SystemdPlan` from the live invocation/env."""
    import shlex
    import shutil

    agentm_bin = shutil.which("agentm")
    if agentm_bin is None:
        raise SystemExit(
            "Cannot find `agentm` on $PATH. Install it first or run from "
            "the venv (`uv run agentm gateway --install-systemd`)."
        )

    system = False
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    socket_url = "unix://%t/agentm/gw.sock"
    run_as = None

    cleaned, workspace = baked_gateway_argv()
    workspace_path = Path(workspace).expanduser().resolve()
    env_file = workspace_path / ".env"

    flags: list[str] = []
    skip_next = False
    for arg in cleaned:
        if skip_next:
            skip_next = False
            continue
        if arg == "--bind":
            skip_next = True
            continue
        if arg.startswith("--bind="):
            continue
        flags.append(arg)
    flags = ["--bind", socket_url, "--cwd", str(workspace_path), *flags]
    gateway_exec_start = " ".join(
        shlex.quote(tok) for tok in [agentm_bin, "gateway", *flags]
    )

    real_bin = Path(agentm_bin).resolve()
    bin_dir = str(real_bin.parent)
    path_env = f"{bin_dir}:/usr/local/bin:/usr/bin:/bin"

    return SystemdPlan(
        system=system,
        unit_dir=unit_dir,
        socket_url=socket_url,
        gateway_exec_start=gateway_exec_start,
        env_file=env_file,
        working_dir=workspace_path,
        path_env=path_env,
        run_as=run_as,
        feishu_bin=resolve_feishu_bin(agentm_bin),
    )


def _systemctl(plan: SystemdPlan) -> list[str]:
    return ["systemctl"] if plan.system else ["systemctl", "--user"]


def systemd_action(*, install: bool) -> None:
    """Install or uninstall the gateway + feishu systemd USER units.

    Always user units (``~/.config/systemd/user`` + ``systemctl --user``, socket
    ``%t/agentm/gw.sock``) — we do not install system units even as root. Both
    units share the same socket (gateway ``--bind`` == feishu ``--connect``).
    The feishu unit is skipped (with a note) when ``agentm-feishu`` is not on
    PATH (run ``uv sync --all-packages``).
    """
    import subprocess

    plan = build_systemd_plan()
    sctl = _systemctl(plan)
    units = [f"{GATEWAY_UNIT}.service", f"{FEISHU_UNIT}.service"]

    if not install:
        for unit in units:
            subprocess.run([*sctl, "stop", unit], check=False)
            subprocess.run([*sctl, "disable", unit], check=False)
        removed = False
        for unit in units:
            path = plan.unit_dir / unit
            if path.exists():
                path.unlink()
                sys.stdout.write(f"Removed {path}\n")
                removed = True
        if removed:
            subprocess.run([*sctl, "daemon-reload"], check=False)
        else:
            sys.stdout.write(
                f"No agentm units in {plan.unit_dir}, nothing to remove.\n"
            )
        return

    plan.unit_dir.mkdir(parents=True, exist_ok=True)
    gateway_path = plan.unit_dir / f"{GATEWAY_UNIT}.service"
    gateway_path.write_text(render_gateway_unit(plan), encoding="utf-8")
    sys.stdout.write(f"Wrote {gateway_path}\n")

    enable_units = [f"{GATEWAY_UNIT}.service"]
    if plan.feishu_bin is not None:
        feishu_path = plan.unit_dir / f"{FEISHU_UNIT}.service"
        feishu_path.write_text(render_feishu_unit(plan), encoding="utf-8")
        sys.stdout.write(f"Wrote {feishu_path}\n")
        enable_units.append(f"{FEISHU_UNIT}.service")
    else:
        sys.stdout.write(
            "NOTE: agentm-feishu not found; skipping the feishu unit. Run "
            "`uv sync --all-packages` then re-run --install-systemd to add it.\n"
        )

    if not plan.env_file.exists():
        sys.stdout.write(
            f"NOTE: {plan.env_file} is missing — the feishu client needs "
            "LARK_APP_ID / LARK_APP_SECRET there (and model creds for the "
            "gateway). Run `agentm onboard` or create it, then restart.\n"
        )

    subprocess.run([*sctl, "daemon-reload"], check=True)
    subprocess.run([*sctl, "enable", "--now", *enable_units], check=True)

    j = "journalctl --user" if not plan.system else "journalctl"
    sys.stdout.write(
        f"\nInstalled and started: {', '.join(enable_units)}\n"
        f"  {j} -u {GATEWAY_UNIT} -f    # follow gateway logs\n"
        f"  {' '.join(sctl)} status {GATEWAY_UNIT}\n"
        f"  contrib/gateway-peers/deploy/update.sh   # pull latest + restart\n"
        f"  agentm gateway --uninstall-systemd       # remove\n"
    )
    if not plan.system:
        sys.stdout.write(
            "NOTE: user units stop at logout unless lingering is enabled:\n"
            f"  sudo loginctl enable-linger {os.environ.get('USER', '$USER')}\n"
        )
