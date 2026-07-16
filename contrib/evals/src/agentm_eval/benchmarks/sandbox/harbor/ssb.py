"""senior-swe-bench adapter (github.com/snorkel-ai/senior-swe-bench-v2026.06).

Harbor layout, but task.toml declares no registry image — environments are
built locally from ``environment/Dockerfile`` and pushed under the
``{registry}/{prefix}-{task}:{tag}`` convention (``--source-images`` is
unsupported). The verifier is a multi-stage pipeline (tests + LLM rubric,
taste, and validation-agent judges).

How to run::

    uv run agentm-eval sandbox batch --bench senior-swe --model azure-gpt -j 5

Repo auto-clones to ``$AGENTM_HOME/bench-repos/senior-swe`` on first run.
Override with ``--repo <path>`` or ``$SSB_REPO``.
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
    DEFAULT_REPO_URL = "https://github.com/snorkel-ai/senior-swe-bench-v2026.06.git"
    DEFAULT_REPO_SUBDIR = "tasks"
    DEFAULT_REPO_ENV = "SSB_REPO"

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
        env.setdefault("UV_DEFAULT_INDEX", self._PIP_MIRROR)
        env.setdefault("NPM_CONFIG_REGISTRY", self._NPM_MIRROR)
        no_proxy = "localhost,127.0.0.1,.svc,.svc.cluster.local,10.0.0.0/8,172.16.0.0/12"
        env.setdefault("HTTPS_PROXY", self._PROXY_URL)
        env.setdefault("HTTP_PROXY", self._PROXY_URL)
        env.setdefault("https_proxy", self._PROXY_URL)
        env.setdefault("http_proxy", self._PROXY_URL)
        env.setdefault("NO_PROXY", no_proxy)
        env.setdefault("no_proxy", no_proxy)
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
        if not env.get("SSB_OVERRIDE_CLASSIFIER_MODEL"):
            env["SSB_OVERRIDE_CLASSIFIER_MODEL"] = f"openai/{model_slug}"
        env.setdefault("OPENAI_BASE_URL", base_url)
        env.setdefault("OPENAI_API_KEY", api_key)
        return True

    VERIFIER_PIP_DEPS = ("litellm[proxy]",)
    _PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
    _NPM_MIRROR = "https://registry.npmmirror.com/"
    _PROXY_URL = "http://sing-box.arl1.svc:7890"

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        mirror = self._PIP_MIRROR
        npm_mirror = self._NPM_MIRROR
        proxy = self._PROXY_URL
        no_proxy = "localhost,127.0.0.1,.svc,.svc.cluster.local,10.0.0.0/8,172.16.0.0/12"
        session.execute([{  # type: ignore[attr-defined]
            "name": "setup-mirrors",
            "command": ["bash", "-lc",
                # build tools for C extensions (uvloop etc.)
                f"HTTPS_PROXY={proxy} HTTP_PROXY={proxy} "
                f"https_proxy={proxy} http_proxy={proxy} "
                "apt-get update -qq && "
                f"HTTPS_PROXY={proxy} HTTP_PROXY={proxy} "
                f"https_proxy={proxy} http_proxy={proxy} "
                "apt-get install -y -qq gcc python3-dev 2>/dev/null || true; "
                # pip mirror
                f"mkdir -p ~/.config/pip && "
                f"printf '[global]\\nindex-url = {mirror}\\n"
                f"trusted-host = pypi.tuna.tsinghua.edu.cn\\n' > ~/.config/pip/pip.conf && "
                # npm / pnpm mirror
                f"npm config set registry {npm_mirror} 2>/dev/null; "
                # persist env vars for all subprocesses
                f"printf '"
                f"UV_DEFAULT_INDEX={mirror}\\n"
                f"HTTPS_PROXY={proxy}\\n"
                f"HTTP_PROXY={proxy}\\n"
                f"https_proxy={proxy}\\n"
                f"http_proxy={proxy}\\n"
                f"no_proxy={no_proxy}\\n"
                f"NO_PROXY={no_proxy}\\n"
                f"' >> /etc/environment "
                "|| true"],
            "work_dir": "/app",
        }], recover_timeout=120)
        if self.VERIFIER_PIP_DEPS:
            deps = " ".join(shlex.quote(d) for d in self.VERIFIER_PIP_DEPS)
            pip_idx = f"-i {mirror} --trusted-host pypi.tuna.tsinghua.edu.cn"
            pip_env = (
                f"HTTPS_PROXY={proxy} HTTP_PROXY={proxy} "
                f"https_proxy={proxy} http_proxy={proxy} "
                f"NO_PROXY={no_proxy} no_proxy={no_proxy} "
            )
            session.execute([{  # type: ignore[attr-defined]
                "name": "verifier-deps",
                "command": ["bash", "-lc",
                    f"{pip_env}"
                    f"python3 -m pip install --break-system-packages -q {pip_idx} {deps} "
                    f"|| python3 -m pip install -q {pip_idx} {deps} "
                    f"|| true"],
                "work_dir": "/app",
                "timeoutSeconds": 600,
            }], recover_timeout=720)
        result = super().evaluate(session, task, timeout=timeout)
        if result.get("reward") is None:
            result["invalid_trial"] = True
        return result
