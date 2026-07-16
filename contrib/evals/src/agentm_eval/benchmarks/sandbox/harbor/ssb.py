"""senior-swe-bench adapter (github.com/snorkel-ai/senior-swe-bench-v2026.06).

Harbor layout, but task.toml declares no registry image — environments are
built locally from ``environment/Dockerfile`` and pushed under the
``{registry}/{prefix}-{task}:{tag}`` convention (``--source-images`` is
unsupported). The verifier is a multi-stage pipeline (tests + LLM rubric,
taste, and validation-agent judges).

How to run::

    uv run agentm-eval sandbox batch \\
      --bench senior-swe \\
      --repo <repo>/tasks \\
      --model azure-gpt \\
      -j 5 -n 1

Images: ``pair-cn-guangzhou.cr.volces.com/opspai/ssb-{task}:v1``.
"""

from __future__ import annotations

import os
import shlex

from ..bench import TaskSpec
from .base import HarborAdapter


class SeniorSweAdapter(HarborAdapter):
    """senior-swe-bench adapter."""

    DEFAULT_REGISTRY = "pair-cn-guangzhou.cr.volces.com/opspai"
    DEFAULT_PREFIX = "ssb"
    DEFAULT_TAG = "v1"

    EXTRA_VERIFIER_ENV = (
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    )

    _JUDGE_CREDENTIAL_VARS = ("PORTKEY_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    _MIN_EVAL_TIMEOUT = 2400

    def get_source_image(self, task: TaskSpec) -> str | None:
        return None

    def eval_timeout_for(self, task: TaskSpec, timeout: int) -> int:
        return max(super().eval_timeout_for(task, timeout), self._MIN_EVAL_TIMEOUT)

    def _verifier_env(self, task: TaskSpec) -> dict[str, str]:
        env = super()._verifier_env(task)
        for key in self.EXTRA_VERIFIER_ENV:
            if not env.get(key) and os.environ.get(key):
                env[key] = os.environ[key]
        if not self._fill_model_overrides(env):
            if env.get("DEEPSEEK_API_KEY") and not any(
                env.get(k) for k in self._JUDGE_CREDENTIAL_VARS
            ):
                env["OPENAI_API_KEY"] = env["DEEPSEEK_API_KEY"]
        return env

    @staticmethod
    def _fill_model_overrides(env: dict[str, str]) -> bool:
        """Set SSB_OVERRIDE_*_MODEL + OPENAI creds from config.toml."""
        try:
            import tomllib
            from agentm.core.lib.user_config import agentm_home_dir
            cfg_path = agentm_home_dir() / "config.toml"
            if not cfg_path.is_file():
                return False
            with open(cfg_path, "rb") as fh:
                cfg = tomllib.load(fh)
        except Exception:  # noqa: BLE001
            return False
        default_name = cfg.get("default_model", "")
        profile = cfg.get("models", {}).get(default_name, {})
        if not profile:
            return False
        model_slug = profile.get("model", "")
        base_url = profile.get("base_url", "")
        api_key = profile.get("api_key", "")
        if not model_slug or not base_url or not api_key:
            return False
        if not env.get("SSB_OVERRIDE_VA_MODEL"):
            env["SSB_OVERRIDE_VA_MODEL"] = f"openai/{model_slug}"
        if not env.get("SSB_OVERRIDE_ALL_JUDGE_MODEL"):
            env["SSB_OVERRIDE_ALL_JUDGE_MODEL"] = f"openai/{model_slug}"
        env["OPENAI_BASE_URL"] = base_url
        env["OPENAI_API_KEY"] = api_key
        return True

    VERIFIER_PIP_DEPS = ("litellm[proxy]",)
    _PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        if self.VERIFIER_PIP_DEPS:
            deps = " ".join(shlex.quote(d) for d in self.VERIFIER_PIP_DEPS)
            mirror = self._PIP_MIRROR
            session.execute([{  # type: ignore[attr-defined]
                "name": "pip-mirror",
                "command": ["bash", "-lc",
                    f"python3 -m pip config set global.index-url {mirror} 2>/dev/null "
                    f"|| mkdir -p ~/.config/pip "
                    f"&& printf '[global]\\nindex-url = {mirror}\\n"
                    f"trusted-host = pypi.tuna.tsinghua.edu.cn\\n' > ~/.config/pip/pip.conf "
                    f"|| true"],
                "work_dir": "/app",
            }], recover_timeout=30)
            session.execute([{  # type: ignore[attr-defined]
                "name": "verifier-deps",
                "command": ["bash", "-lc",
                    f"python3 -m pip install --break-system-packages -q {deps} "
                    f"2>/dev/null || python3 -m pip install -q {deps} 2>/dev/null || true"],
                "work_dir": "/app",
                "timeoutSeconds": 600,
            }], recover_timeout=720)
        result = super().evaluate(session, task, timeout=timeout)
        if result.get("reward") is None:
            result["invalid_trial"] = True
        return result
