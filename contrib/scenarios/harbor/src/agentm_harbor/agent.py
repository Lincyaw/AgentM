"""Harbor agent adapter — installs agentm from PyPI and runs it."""

from __future__ import annotations

import json
import os
import shlex
import tempfile

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from loguru import logger

SCENARIO_MANIFEST = """\
name: harbor_bench
description: |
  Minimal tool set for Harbor benchmark evaluation.
  Bash, file I/O, observability, compaction.
  No memory or skill loader (ephemeral container).

extensions:
  - module: agentm.extensions.builtin.operations
    config:
      backend: local
  - module: agentm.extensions.builtin.tool_result_cap
    config:
      max_tokens: 50000
      preview_tokens: 50000
  - module: agentm.extensions.builtin.file_tools
  - module: agentm.extensions.builtin.tool_bash
  - module: agentm.extensions.builtin.observability
  - module: agentm.extensions.builtin.system_prompt
    config:
      prompt: ""
  - module: agentm.extensions.builtin.runtime_context
  - module: agentm.extensions.builtin.llm_compaction
    config:
      tool_result_max_tokens: 8000
  - module: agentm.extensions.builtin.read_history
    config:
      tool_result_max_tokens: 20000
      total_max_tokens: 50000
"""

SCENARIO_MANIFEST_HARNESS = """\
name: harbor_bench_harness
description: |
  Harbor benchmark + llmharness cognitive audit.
  Adds extractor + auditor that surface reasoning drift as reminders.

extensions:
  - module: agentm.extensions.builtin.operations
    config:
      backend: local
  - module: agentm.extensions.builtin.tool_result_cap
    config:
      max_tokens: 50000
      preview_tokens: 50000
  - module: agentm.extensions.builtin.file_tools
  - module: agentm.extensions.builtin.tool_bash
  - module: agentm.extensions.builtin.observability
  - module: agentm.extensions.builtin.system_prompt
    config:
      prompt: ""
  - module: agentm.extensions.builtin.runtime_context
  - module: agentm.extensions.builtin.llm_compaction
    config:
      tool_result_max_tokens: 8000
  - module: agentm.extensions.builtin.read_history
    config:
      tool_result_max_tokens: 20000
      total_max_tokens: 50000
  - module: llmharness.atom
    config:
      mode: sync
      extractor_interval_turns: 5
      audit_interval_turns: 10
      enable_reminders: true
"""

_SCENARIO_DIR = "/tmp/harbor_bench"
_HARNESS_SCENARIO_DIR = "/tmp/harbor_bench_harness"


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
        CliFlag(
            "reasoning_effort",
            cli="--reasoning-effort",
            type="str",
            env_fallback="AGENTM_REASONING_EFFORT",
        ),
    ]

    def __init__(self, logs_dir, *args, **kwargs):
        self._extra_wheels: list[str] = []
        raw = kwargs.pop("extra_wheels", "")
        if raw:
            self._extra_wheels = [p.strip() for p in raw.split(",") if p.strip()]
        harness_val = kwargs.pop("harness", False)
        self._use_harness = harness_val is True or str(harness_val).lower() in (
            "true", "1", "yes",
        )
        super().__init__(logs_dir, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "agentm"

    def get_version_command(self) -> str | None:
        return "agentm --help 2>&1 | head -1 || true"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                "apt-get update -qq && "
                "DEBIAN_FRONTEND=noninteractive "
                "apt-get install -y -qq python3 python3-pip python3-venv curl git"
            ),
        )

        version_spec = f"=={self._version}" if self._version else ""
        await self.exec_as_agent(
            environment,
            command=(
                "curl -LsSf https://astral.sh/uv/install.sh | sh && "
                'export PATH="$HOME/.local/bin:$PATH" && '
                f"uv tool install agentm{version_spec} && "
                "agentm --help > /dev/null"
            ),
        )

        for wheel_path in self._extra_wheels:
            remote = f"/tmp/{os.path.basename(wheel_path)}"
            await environment.upload_file(wheel_path, remote)
            await self.exec_as_agent(
                environment,
                command=(
                    'export PATH="$HOME/.local/bin:$PATH" && '
                    "uv pip install --python "
                    "$HOME/.local/share/uv/tools/agentm/bin/python3 "
                    f"{remote}"
                ),
            )

        if self._use_harness:
            scenario_dir = _HARNESS_SCENARIO_DIR
            manifest_content = SCENARIO_MANIFEST_HARNESS
        else:
            scenario_dir = _SCENARIO_DIR
            manifest_content = SCENARIO_MANIFEST

        self._active_scenario_dir = scenario_dir

        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {scenario_dir}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(manifest_content)
            local_manifest = f.name
        try:
            await environment.upload_file(
                local_manifest, f"{scenario_dir}/manifest.yaml"
            )
        finally:
            os.unlink(local_manifest)

    def _build_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for key in (
            "AGENTM_API_KEY",
            "AGENTM_PROVIDER",
            "AGENTM_MODEL",
            "AGENTM_BASE_URL",
            "AGENTM_REASONING_EFFORT",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
        ):
            val = self._get_env(key)
            if val:
                env[key] = val
        api_key = env.get("AGENTM_API_KEY", "")
        if api_key and "OPENAI_API_KEY" not in env:
            env["OPENAI_API_KEY"] = api_key
        if api_key and "ANTHROPIC_API_KEY" not in env:
            env["ANTHROPIC_API_KEY"] = api_key
        base_url = env.get("AGENTM_BASE_URL", "")
        if base_url and "OPENAI_BASE_URL" not in env:
            env["OPENAI_BASE_URL"] = base_url
        return env

    def _build_model_args(self) -> str:
        if not self.model_name:
            return ""
        return f"--model {shlex.quote(self.model_name)}"

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped = shlex.quote(instruction)
        env = self._build_env()
        model_args = self._build_model_args()
        cli_flags = self.build_cli_flags()
        extra = f" {cli_flags}" if cli_flags else ""
        agent_dir = EnvironmentPaths.agent_dir
        agentm_home = agent_dir / "agentm-home"
        env["AGENTM_HOME"] = str(agentm_home)
        scenario_dir = getattr(self, "_active_scenario_dir", _SCENARIO_DIR)

        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {shlex.quote(str(agent_dir))} "
                f"{shlex.quote(str(agentm_home))}"
            ),
            env=env,
        )

        try:
            await self.exec_as_agent(
                environment,
                command=(
                    'export PATH="$HOME/.local/bin:$PATH"; '
                    f"agentm -p {escaped}"
                    f" {model_args}"
                    f" --scenario {scenario_dir}"
                    f"{extra}"
                    f" 2>&1 | tee {agent_dir / 'agentm-trace.jsonl'}"
                    "; exit 0"
                ),
                env=env,
            )
        finally:
            try:
                obs_dest = agent_dir / "observability"
                obs_dest_q = shlex.quote(str(obs_dest))
                await self.exec_as_agent(
                    environment,
                    command=(
                        'obs_dir="${AGENTM_OBSERVABILITY_DIR:-'
                        '${AGENTM_HOME:-$HOME/.agentm}/observability}"; '
                        f"mkdir -p {obs_dest_q}; "
                        f'if [ -d "$obs_dir" ]; then cp -R "$obs_dir"/. {obs_dest_q}/; fi'
                    ),
                    env=env,
                )
            except Exception as exc:  # noqa: BLE001
                # Best-effort copy of the observability dir out of the sandbox.
                logger.debug("harbor: observability copy-out failed: {}", exc)

    def populate_context_post_run(self, context: AgentContext) -> None:
        obs_dir = self.logs_dir / "observability"
        if not obs_dir.exists():
            obs_dir = self.logs_dir

        total_input = 0
        total_output = 0
        total_cache = 0

        for jsonl_file in obs_dir.rglob("*.jsonl"):
            try:
                for line in jsonl_file.read_text().splitlines():
                    if not line.strip():
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
                        total_input += attrs.get("input_tokens", 0)
                        total_output += attrs.get("output_tokens", 0)
                        total_cache += attrs.get("cache_read", 0)
                        total_cache += attrs.get("cache_write", 0)
            except OSError:
                continue

        context.n_input_tokens = total_input
        context.n_output_tokens = total_output
        context.n_cache_tokens = total_cache
