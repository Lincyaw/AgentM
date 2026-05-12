"""Fail-stop: env vars must resolve at runtime, not just appear in --help.

If a future refactor moves ``autoload_dotenv()`` past argv parsing, OR caches
the default at module-import time, ``AGENTM_SOCKET`` from the environment
would be silently ignored and the gateway would fall back to its built-in
default path. This test sets ``AGENTM_SOCKET``, runs ``agentm-gateway
--check``, and asserts the resolved bind URL comes from the env value.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_gateway_env_resolves_at_runtime() -> None:
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
