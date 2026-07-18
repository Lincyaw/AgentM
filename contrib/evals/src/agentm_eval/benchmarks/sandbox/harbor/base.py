"""Harbor base adapter: task.toml + instruction.md + tests/test.sh."""

from __future__ import annotations

import os
import re
import shlex
import tomllib
from pathlib import Path

from ..bench import TaskSpec, image_name


def _source_image_from_dockerfile(task_dir: Path) -> str:
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return ""
    for raw_line in dockerfile.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line, comments=True)
        if not parts or parts[0].upper() != "FROM":
            continue
        image_idx = 1
        while image_idx < len(parts) and parts[image_idx].startswith("--"):
            image_idx += 1
        if image_idx < len(parts):
            return parts[image_idx]
    return ""


_ENV_REF = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def expand_host_env(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` against the host environment."""
    return _ENV_REF.sub(
        lambda m: os.environ.get(m.group("name")) or (m.group("default") or ""),
        value,
    )


class HarborAdapter:
    """Harbor format: task.toml + instruction.md + environment/ dir."""

    def discover_tasks(self, source: str) -> list[TaskSpec]:
        repo = Path(source).expanduser().resolve()
        tasks: list[TaskSpec] = []
        for task_dir in sorted(repo.iterdir()):
            if not task_dir.is_dir():
                continue
            task_toml = task_dir / "task.toml"
            if not task_toml.is_file():
                continue

            meta = tomllib.loads(task_toml.read_text())

            prompt = ""
            instruction_file = task_dir / "instruction.md"
            if instruction_file.is_file():
                prompt = instruction_file.read_text().strip()

            image = ""
            env_section = meta.get("environment", {})
            if isinstance(env_section, dict):
                image = env_section.get("docker_image", "")
            if not image:
                image = _source_image_from_dockerfile(task_dir)

            metadata = meta.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            tasks.append(TaskSpec(
                name=task_dir.name,
                prompt=prompt,
                image=image,
                path=str(task_dir),
                difficulty=metadata.get("difficulty", meta.get("difficulty", "")),
                category=metadata.get("category", meta.get("category", "")),
                extra=meta,
            ))
        return tasks

    def get_image(self, task: TaskSpec, registry: str, prefix: str, tag: str) -> str:
        return image_name(task.name, registry, prefix, tag)

    def get_source_image(self, task: TaskSpec) -> str | None:
        return task.image or None

    def supports_build(self) -> bool:
        return True

    def _verifier_env(self, task: TaskSpec) -> dict[str, str]:
        """Env vars exported to test.sh: task.toml ``[verifier.env]``."""
        raw = task.extra.get("verifier", {})
        declared = raw.get("env", {}) if isinstance(raw, dict) else {}
        env: dict[str, str] = {}
        for key, value in declared.items():
            expanded = expand_host_env(str(value))
            if expanded:
                env[key] = expanded
        return env

    def eval_timeout_for(self, task: TaskSpec, timeout: int) -> int:
        """Effective verifier timeout: task.toml ``[verifier] timeout_sec``
        extends the CLI value when larger."""
        verifier = task.extra.get("verifier", {})
        task_timeout = verifier.get("timeout_sec") if isinstance(verifier, dict) else None
        if isinstance(task_timeout, (int, float)) and task_timeout > timeout:
            return int(task_timeout)
        return timeout

    def get_sidecar_containers(
        self, task: TaskSpec, registry: str, prefix: str, tag: str
    ) -> list[dict]:
        return []

    def get_main_env(self, task: TaskSpec) -> dict[str, str]:
        return {}

    def get_resources(self, task: TaskSpec) -> dict[str, str]:
        """Operations-config resource fields from task.toml ``[environment]``."""
        env_section = task.extra.get("environment", {})
        if not isinstance(env_section, dict):
            return {}
        fields: dict[str, str] = {}
        cpus = env_section.get("cpus")
        if isinstance(cpus, (int, float)) and cpus > 0:
            fields["cpu_request"] = str(int(cpus))
            fields["cpu_limit"] = str(int(cpus))
        memory_mb = env_section.get("memory_mb")
        if isinstance(memory_mb, (int, float)) and memory_mb > 0:
            fields["memory_request"] = f"{int(memory_mb)}Mi"
            fields["memory_limit"] = f"{int(memory_mb)}Mi"
        return fields

    def is_pass(self, result: dict) -> bool:
        reward = result.get("reward")
        return isinstance(reward, (int, float)) and reward >= 1.0

    def format_score_line(self, r: dict) -> str:
        name = r.get("task", "?")
        tools = r.get("tools", "?")
        status = r.get("status", "?").upper()
        reward = r.get("reward")
        reward_str = f"{reward:.2f}" if isinstance(reward, (int, float)) else "-"
        sid = str(r.get("session_id", "-"))[:12]
        return f"  [{status}] {name} tools={tools} reward={reward_str} session={sid}"

    def summary_header(self) -> str:
        return (
            f"  {'Task':<30} {'Reward':<12}\n"
            f"  {'-' * 45}"
        )

    def summary_row(self, name: str, r: dict) -> str:
        reward = r.get("reward")
        reward_str = f"{reward:.2f}" if isinstance(reward, (int, float)) else "-"
        return f"  {name:<30} {reward_str:<12}"

    def summary_footer(self, results: dict[str, dict]) -> str:
        scored: list[float] = [
            float(r["reward"]) for r in results.values()
            if isinstance(r.get("reward"), (int, float))
        ]
        if scored:
            avg = sum(scored) / len(scored)
            return f"\nAvg reward: {avg:.3f} ({len(scored)}/{len(results)} scored)"
        return ""

    def pass_at_k_header(self) -> str:
        return (
            f"  {'Task':<30} {'Best':<8} {'Pass':<8} {'Avg reward'}\n"
            f"  {'-' * 60}"
        )

    def pass_at_k_row(self, name: str, runs: list[dict]) -> tuple[str, dict]:
        rewards = [
            float(r["reward"]) for r in runs
            if isinstance(r.get("reward"), (int, float))
        ]
        best = max(rewards) if rewards else None
        avg = sum(rewards) / len(rewards) if rewards else None
        any_pass = any(self.is_pass(r) for r in runs)

        best_str = f"{best:.2f}" if best is not None else "-"
        pass_str = "YES" if any_pass else "no"
        avg_str = f"{avg:.3f}" if avg is not None else "-"
        line = f"  {name:<30} {best_str:<8} {pass_str:<8} {avg_str}"
        stats = {"any_pass": any_pass, "avg_reward": avg, "best_reward": best}
        return line, stats

    def pass_at_k_footer(self, all_stats: list[dict], n_tasks: int) -> str:
        pass_count = sum(1 for s in all_stats if s.get("any_pass"))
        avg_rewards = [s["avg_reward"] for s in all_stats if s.get("avg_reward") is not None]
        lines = [f"\n  Overall pass@k: {pass_count}/{n_tasks} = {pass_count / n_tasks:.1%}"]
        if avg_rewards:
            lines.append(f"  Avg reward:     {sum(avg_rewards) / len(avg_rewards):.3f}")
        return "\n".join(lines)
