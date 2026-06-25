"""Harbor adapter (formerly Terminal Bench 2.0): task.toml + instruction.md + tests/test.sh."""

from __future__ import annotations

import tomllib
from pathlib import Path

from .base import TaskSpec, image_name, upload_file_to_sandbox


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

            tasks.append(TaskSpec(
                name=task_dir.name,
                prompt=prompt,
                image=image,
                path=str(task_dir),
                difficulty=meta.get("difficulty", ""),
                category=meta.get("category", ""),
                extra=meta,
            ))
        return tasks

    def get_image(self, task: TaskSpec, registry: str, prefix: str, tag: str) -> str:
        if task.image:
            return task.image
        return image_name(task.name, registry, prefix, tag)

    def supports_build(self) -> bool:
        return True

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        task_dir = Path(task.path)

        test_sh = task_dir / "tests" / "test.sh"
        if test_sh.is_file():
            session.execute([{  # type: ignore[attr-defined]
                "name": "prep",
                "command": ["bash", "-lc", "mkdir -p /tests /logs/verifier"],
                "work_dir": "/app",
            }])
            upload_file_to_sandbox(session, "_eval_staging_test.sh", test_sh.read_bytes())
            session.execute([{  # type: ignore[attr-defined]
                "name": "mv-test",
                "command": ["bash", "-lc",
                    "mv /app/_eval_staging_test.sh /tests/test.sh && "
                    "chmod +x /tests/test.sh"],
                "work_dir": "/app",
            }])

        r = session.execute([{  # type: ignore[attr-defined]
            "name": "eval",
            "command": ["bash", "-lc",
                f"mkdir -p /logs/verifier && "
                f"timeout {timeout} bash /tests/test.sh 2>&1"],
            "work_dir": "/app",
        }])
        eval_out = r.results[0].output.stdout

        reward = None
        try:
            reward_data = session._client.download_file(  # type: ignore[attr-defined]
                session._session_id, "../logs/verifier/reward.txt"  # type: ignore[attr-defined]
            )
            reward = float(reward_data.decode().strip())
        except Exception:  # noqa: S110
            pass

        if reward is None:
            try:
                r2 = session.execute([{  # type: ignore[attr-defined]
                    "name": "read-reward",
                    "command": ["bash", "-lc", "cat /logs/verifier/reward.txt 2>/dev/null"],
                    "work_dir": "/app",
                }])
                txt = r2.results[0].output.stdout.strip()
                if txt:
                    reward = float(txt)
            except Exception:  # noqa: S110
                pass

        return {
            "reward": reward,
            "eval_output": eval_out or "",
        }

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
