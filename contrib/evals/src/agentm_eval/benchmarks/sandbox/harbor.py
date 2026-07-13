"""Harbor adapter (formerly Terminal Bench 2.0): task.toml + instruction.md + tests/test.sh."""

from __future__ import annotations

import shlex
import tomllib
from pathlib import Path

from loguru import logger

from .bench import TaskSpec, image_name, upload_file_to_sandbox


def _patch_test_sh(content: bytes) -> bytes:
    """No-op: sandbox has network access, let test.sh install its own deps."""
    return content


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

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        task_dir = Path(task.path)

        tests_dir = task_dir / "tests"
        if tests_dir.is_dir():
            session.execute([{  # type: ignore[attr-defined]
                "name": "prep",
                "command": ["bash", "-lc", "mkdir -p /tests /logs/verifier /app/_eval_staging"],
                "work_dir": "/app",
            }])
            for f in tests_dir.rglob("*"):
                if not f.is_file():
                    continue
                content = f.read_bytes()
                if f.name == "test.sh":
                    content = _patch_test_sh(content)
                rel = str(f.relative_to(tests_dir))
                upload_file_to_sandbox(session, f"_eval_staging/{rel}", content)
            session.execute([{  # type: ignore[attr-defined]
                "name": "mv-tests",
                "command": ["bash", "-lc",
                    'cd /app/_eval_staging && find . -type f | while read f; do '
                    'mkdir -p "/tests/$(dirname "$f")" && '
                    'mv "$f" "/tests/$f" && chmod +x "/tests/$f"; done && '
                    'rm -rf /app/_eval_staging'],
                "work_dir": "/app",
            }])

        session.execute([{  # type: ignore[attr-defined]
            "name": "prep-eval",
            "command": ["bash", "-lc", "mkdir -p /logs/verifier"],
            "work_dir": "/app",
        }])

        r = session.execute([{  # type: ignore[attr-defined]
            "name": "eval",
            "command": ["bash", "-lc",
                f"timeout {timeout} bash /tests/test.sh 2>&1"],
            "work_dir": "/app",
        }])
        eval_out = r.results[0].output.stdout

        reward = None
        try:
            r2 = session.execute([{  # type: ignore[attr-defined]
                "name": "read-reward",
                "command": ["bash", "-lc", "cat /logs/verifier/reward.txt 2>/dev/null"],
                "work_dir": "/app",
            }])
            txt = r2.results[0].output.stdout.strip()
            if txt:
                reward = float(txt)
        except Exception as exc:  # noqa: S110
            logger.debug("Failed to read reward.txt: {}", exc)

        return {
            "reward": reward,
            "eval_output": eval_out or "",
        }

    def is_pass(self, result: dict) -> bool:
        reward = result.get("reward")
        return isinstance(reward, (int, float)) and reward >= 1.0

    def format_score_line(self, r: dict) -> str:
        name = r.get("task", "?")
        tools = r.get("tools", "?")
        status = r.get("status", "?").upper()
        reward = r.get("reward")
        reward_str = f"{reward:.2f}" if isinstance(reward, (int, float)) else "-"
        return f"  [{status}] {name} tools={tools} reward={reward_str}"

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


class LhtbAdapter(HarborAdapter):
    """LHTB (Long-Horizon Terminal-Bench, github.com/zli12321/LHTB).

    Standard Harbor task layout with two bench-specific conventions:
    solved means reward >= 0.95 (not 1.0), and each task.toml declares a
    ``[verifier] timeout_sec`` that can exceed the CLI default (some
    verifiers replay hundreds of authenticated moves). Point ``--repo``
    at the checkout's ``tasks/`` directory and run with
    ``--source-images`` — all 46 tasks ship prebuilt docker.io images.
    """

    SOLVED_THRESHOLD = 0.95

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        verifier = task.extra.get("verifier", {})
        task_timeout = verifier.get("timeout_sec") if isinstance(verifier, dict) else None
        if isinstance(task_timeout, (int, float)) and task_timeout > timeout:
            timeout = int(task_timeout)
        return super().evaluate(session, task, timeout=timeout)

    def is_pass(self, result: dict) -> bool:
        reward = result.get("reward")
        return isinstance(reward, (int, float)) and reward >= self.SOLVED_THRESHOLD
