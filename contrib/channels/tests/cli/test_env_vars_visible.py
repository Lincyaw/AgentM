"""Fail-stop tests: env vars must be visible at typer parse time.

Each of the four CLIs (``agentm-gateway``, ``agentm-terminal``,
``agentm-feishu``, ``agentm-worker``) accepts a unified ``AGENTM_*``
prefix for its environment-variable defaults via typer's
``envvar=...``. If a future refactor reorders module imports or moves
``autoload_dotenv()`` past argv parsing, those defaults silently fall
back to built-ins and the test catches it.

These tests run the CLI's ``--help`` (or a structural inspection) under
a controlled environment and assert that the env var is reflected — the
strongest contract guarantee we can make without spinning up a real
gateway.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _run_help(prog: str, *, env_vars: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run ``<prog> --help`` with a controlled env. Skip dotenv autoload
    so the repo's own ``.env`` doesn't bleed real keys into the test."""
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("AGENTM_")},
        "AGENTM_SKIP_DOTENV": "1",
        **env_vars,
    }
    return subprocess.run(
        [sys.executable, "-m", prog, "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )


def test_gateway_envvar_appears_in_help() -> None:
    """``--help`` should mention the env var names so the contract is
    discoverable. Pin them here so a future rename breaks loudly."""
    result = _run_help("agentm_channels.cli", env_vars={})
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for name in (
        "AGENTM_SOCKET",
        "AGENTM_CWD",
        "AGENTM_SCENARIO",
        "AGENTM_PROVIDER",
        "AGENTM_MODEL",
        "AGENTM_LOG_LEVEL",
        "AGENTM_CONFIG",
        "AGENTM_STATE_DIR",
    ):
        assert name in out, f"{name} not surfaced in agentm-gateway --help"


def test_terminal_envvar_appears_in_help() -> None:
    result = _run_help("agentm_terminal.cli", env_vars={})
    assert result.returncode == 0, result.stderr
    assert "AGENTM_SOCKET" in result.stdout
    assert "AGENTM_FORMAT" in result.stdout


def test_worker_envvar_appears_in_help() -> None:
    result = _run_help("agentm_worker.cli", env_vars={})
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for name in (
        "AGENTM_SOCKET",
        "AGENTM_SCENARIO",
        "AGENTM_CWD",
        "AGENTM_PROVIDER",
        "AGENTM_MODEL",
    ):
        assert name in out, f"{name} not surfaced in agentm-worker --help"


def test_feishu_envvar_appears_in_help() -> None:
    result = _run_help("agentm_feishu.cli", env_vars={})
    assert result.returncode == 0, result.stderr
    assert "AGENTM_SOCKET" in result.stdout
    # LARK_APP_ID is vendor-named (lark-oapi convention); keep as-is.
    assert "LARK_APP_ID" in result.stdout


def test_gateway_env_resolves_at_runtime() -> None:
    """Set ``AGENTM_SOCKET`` then run ``--check`` and assert the
    resolved bind URL comes from the env (not the built-in default).

    This is the actual fail-stop: if ``autoload_dotenv()`` moves after
    typer parse, OR a future refactor caches the default at import
    time, the resolved socket would be the conventional path instead of
    the env value, and this test fails.
    """
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="agentm-env-") as d:
        sock = Path(d) / "from-env.sock"
        env = {
            **{k: v for k, v in os.environ.items() if not k.startswith("AGENTM_")},
            "AGENTM_SKIP_DOTENV": "1",
            "AGENTM_SOCKET": f"unix://{sock}",
        }
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "agentm_channels.cli",
                "--bind-allow-any-uid",
                "--check",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        assert payload["bind"]["socket"] == f"unix://{sock}"
