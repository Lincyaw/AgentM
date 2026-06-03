"""Harbor agent adapter for AgentM to run Terminal-Bench evaluations."""

import json
import os
import shlex
import subprocess
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, CliFlag, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths


class AgentMAgent(BaseInstalledAgent):

    CLI_FLAGS = [
        CliFlag(
            "max_turns",
            cli="--max-turns",
            type="int",
            env_fallback="AGENTM_MAX_TURNS",
        ),
        CliFlag(
            "max_tool_calls",
            cli="--max-tool-calls",
            type="int",
            env_fallback="AGENTM_MAX_TOOL_CALLS",
        ),
    ]

    @staticmethod
    def name() -> str:
        return "agentm"

    def __init__(self, logs_dir: Path, *args, **kwargs):
        self._scenario = kwargs.pop("scenario", "terminal_bench")
        super().__init__(logs_dir, *args, **kwargs)

    def get_version_command(self) -> str | None:
        return "agentm --help 2>&1 | head -1 || true"

    @staticmethod
    def _find_repo_root() -> Path | None:
        """Locate the AgentM repo root (contains pyproject.toml with [project.scripts] agentm)."""
        candidates: list[Path] = []

        # 1. Walk up from this source file (works for editable / source installs)
        p = Path(__file__).resolve().parent
        for _ in range(8):
            candidates.append(p)
            p = p.parent

        # 2. Resolve via dist-info direct_url.json (works for non-editable local installs)
        dist_infos = list(
            Path(__file__).resolve().parent.parent.glob("agentm_terminal_bench*.dist-info")
        )
        for di in dist_infos:
            du = di / "direct_url.json"
            if du.is_file():
                try:
                    data = json.loads(du.read_text())
                    url = data.get("url", "")
                    if url.startswith("file://"):
                        scenario_dir = Path(url.removeprefix("file://"))
                        for _ in range(5):
                            candidates.append(scenario_dir)
                            scenario_dir = scenario_dir.parent
                except (json.JSONDecodeError, OSError):
                    pass

        for c in candidates:
            toml = c / "pyproject.toml"
            if toml.is_file():
                try:
                    text = toml.read_text()
                    if 'agentm = "agentm:main"' in text:
                        return c
                except OSError:
                    pass
        return None

    def _find_wheel(self) -> Path:
        """Find the agentm wheel to install in the container.

        Resolution order:
        1. AGENTM_WHEEL_PATH env var (explicit path)
        2. Pre-built wheel in the repo's dist/ directory
        3. Build one on the fly via ``uv build --wheel``
        """
        explicit = os.environ.get("AGENTM_WHEEL_PATH")
        if explicit:
            p = Path(explicit)
            if p.is_file():
                return p
            raise FileNotFoundError(f"AGENTM_WHEEL_PATH={explicit} does not exist")

        repo_root = self._find_repo_root()
        if repo_root is None:
            raise FileNotFoundError(
                "Cannot locate AgentM repo root. "
                "Set AGENTM_WHEEL_PATH to a pre-built .whl file."
            )

        dist_dir = repo_root / "dist"
        wheels = sorted(dist_dir.glob("agentm-*.whl")) if dist_dir.is_dir() else []
        if wheels:
            return wheels[-1]

        self.logger.info("No pre-built wheel found, building via `uv build --wheel`")
        subprocess.run(
            ["uv", "build", "--wheel"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        wheels = sorted(dist_dir.glob("agentm-*.whl"))
        if wheels:
            return wheels[-1]
        raise FileNotFoundError("Failed to build agentm wheel")

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                "apt-get update && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y "
                "python3 python3-pip python3-venv curl git"
            ),
        )

        await self.exec_as_agent(
            environment,
            command="curl -LsSf https://astral.sh/uv/install.sh | sh",
        )

        wheel_path = self._find_wheel()
        container_wheel = f"/tmp/{wheel_path.name}"
        await environment.upload_file(wheel_path, container_wheel)

        await self.exec_as_agent(
            environment,
            command=(
                'export PATH="$HOME/.local/bin:$PATH" && '
                f"uv pip install --system --break-system-packages {container_wheel} && "
                "agentm --help > /dev/null"
            ),
        )

        repo_root = self._find_repo_root()
        if repo_root:
            manifest = repo_root / "contrib" / "scenarios" / "terminal_bench" / "manifest.yaml"
        else:
            manifest = Path(__file__).resolve().parent.parent.parent / "manifest.yaml"
        if manifest.is_file():
            await self.exec_as_agent(environment, command="mkdir -p /tmp/terminal_bench")
            await environment.upload_file(manifest, "/tmp/terminal_bench/manifest.yaml")

        config_toml = Path(
            os.environ.get("AGENTM_HOME", Path.home() / ".agentm")
        ) / "config.toml"
        if config_toml.is_file():
            await self.exec_as_agent(environment, command="mkdir -p $HOME/.agentm")
            await environment.upload_file(config_toml, "/root/.agentm/config.toml")
            await self.exec_as_agent(
                environment,
                command=(
                    "cp /root/.agentm/config.toml $HOME/.agentm/config.toml "
                    "2>/dev/null || true"
                ),
            )

    def _is_profile_name(self, model_name: str) -> bool:
        return "/" not in model_name and "." not in model_name

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        escaped_instruction = shlex.quote(instruction)

        env: dict[str, str] = {}

        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "AGENTM_API_KEY",
            "AGENTM_PROVIDER",
            "AGENTM_MODEL",
            "AGENTM_BASE_URL",
        ]:
            val = self._get_env(key)
            if val:
                env[key] = val

        model_args = ""
        if self.model_name:
            if self._is_profile_name(self.model_name):
                model_args = f"--model {shlex.quote(self.model_name)}"
            else:
                if "/" in self.model_name:
                    provider, model = self.model_name.split("/", 1)
                else:
                    provider, model = "openai", self.model_name
                model_args = f"--provider {shlex.quote(provider)} --model {shlex.quote(model)}"

                provider_key_map = {
                    "anthropic": "ANTHROPIC_API_KEY",
                    "openai": "OPENAI_API_KEY",
                }
                key_env = provider_key_map.get(provider)
                if key_env and key_env in os.environ and key_env not in env:
                    env[key_env] = os.environ[key_env]

        cli_flags = self.build_cli_flags()
        extra_flags = f" {cli_flags}" if cli_flags else ""

        output_dir = EnvironmentPaths.agent_dir
        trace_output = output_dir / "agentm-trace.jsonl"

        scenario_flag = ""
        if self._scenario:
            scenario_flag = " --scenario /tmp/terminal_bench"

        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {output_dir}",
            env=env,
        )

        result = await environment.exec(
            command=(
                f"set -o pipefail; "
                f'export PATH="$HOME/.local/bin:$PATH"; '
                f"agentm -p {escaped_instruction}"
                f" {model_args}"
                f"{scenario_flag}"
                f"{extra_flags}"
                f" 2>&1 | tee {trace_output}"
                f"; exit 0"
            ),
            env=env,
        )
        self.logger.debug(
            f"agentm exited with code {result.return_code}",
            extra={"stdout": (result.stdout or "")[-500:]},
        )

        await environment.exec(
            command=f"cp -r .agentm/observability/ {output_dir}/observability/ 2>/dev/null || true",
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        trace_dir = self.logs_dir
        jsonl_files = list(trace_dir.rglob("*.jsonl"))

        if not jsonl_files:
            self.logger.debug("No AgentM trace files found")
            return

        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_tokens = 0

        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if not isinstance(event, dict):
                            continue

                        if event.get("kind") == "turn.summary":
                            attrs = event.get("attributes", {})
                            if not isinstance(attrs, dict):
                                continue
                            total_input_tokens += attrs.get("input_tokens", 0)
                            total_output_tokens += attrs.get("output_tokens", 0)
                            total_cache_tokens += attrs.get("cache_read", 0)
                            total_cache_tokens += attrs.get("cache_write", 0)
            except OSError:
                continue

        context.n_input_tokens = total_input_tokens
        context.n_output_tokens = total_output_tokens
        context.n_cache_tokens = total_cache_tokens
