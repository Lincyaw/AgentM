"""SWE-bench adapter (Verified + Pro): HuggingFace dataset + pre-built images."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .base import TaskSpec

try:
    import datasets as _datasets
except ImportError:  # noqa: S110
    _datasets = None  # type: ignore[assignment]


class SWEBenchAdapter:
    """SWE-bench Verified / Pro: HuggingFace dataset + pre-built Docker images.

    Does NOT run gold tests inside the agent sandbox. Instead, extracts the
    agent's git diff as model_patch and writes predictions JSONL for
    upstream SWE-bench harness scoring.
    """

    def __init__(self, *, variant: str = "verified") -> None:
        self.variant = variant

    def discover_tasks(self, source: str) -> list[TaskSpec]:
        if _datasets is None:
            typer.echo("Error: 'datasets' library required for SWE-bench. "
                       "Install with: pip install datasets", err=True)
            raise typer.Exit(1)

        ds = _datasets.load_dataset(source, split="test")
        tasks: list[TaskSpec] = []
        for row in ds:
            instance_id = row["instance_id"]
            tasks.append(TaskSpec(
                name=instance_id,
                prompt=row.get("problem_statement", ""),
                image=self._image_for_instance(instance_id, row),
                instance_id=instance_id,
                base_commit=row.get("base_commit", ""),
                repo=row.get("repo", ""),
                extra={k: v for k, v in row.items()
                       if k not in ("problem_statement", "instance_id",
                                    "base_commit", "repo")},
            ))
        return tasks

    def _image_for_instance(self, instance_id: str, row: dict) -> str:
        if self.variant == "pro":
            docker_tag = row.get("dockerhub_tag", instance_id)
            return f"jefzda/sweap-images:{docker_tag}"
        return f"swebench/sweb.eval.x86_64.{instance_id}:latest"

    def get_image(self, task: TaskSpec, registry: str, prefix: str, tag: str) -> str:
        return task.image

    def supports_build(self) -> bool:
        return False

    def evaluate(
        self, session: object, task: TaskSpec, *, timeout: int = 300
    ) -> dict:
        """Extract git diff from sandbox as model_patch. No test execution."""
        r = session.execute([{  # type: ignore[attr-defined]
            "name": "extract-patch",
            "command": ["bash", "-lc",
                "cd /app && git diff HEAD 2>/dev/null || "
                "git diff 2>/dev/null || echo ''"],
            "work_dir": "/app",
        }])
        model_patch = r.results[0].output.stdout

        return {
            "instance_id": task.instance_id,
            "model_patch": model_patch,
            "has_patch": bool(model_patch.strip()),
        }

    def format_score_line(self, r: dict) -> str:
        name = r.get("task", "?")
        tools = r.get("tools", "?")
        status = r.get("status", "?").upper()
        has_patch = r.get("has_patch", False)
        return f"  [{status}] {name} tools={tools} patch={'yes' if has_patch else 'no'}"

    def summary_header(self) -> str:
        return (
            f"  {'Task':<50} {'Patch'}\n"
            f"  {'-' * 70}"
        )

    def summary_row(self, name: str, r: dict) -> str:
        has_patch = r.get("has_patch", False)
        return f"  {name:<50} {'yes' if has_patch else 'no'}"

    def summary_footer(self, results: dict[str, dict]) -> str:
        patched = sum(1 for r in results.values() if r.get("has_patch"))
        return f"\n{patched}/{len(results)} produced patches"


def write_predictions(
    results: dict[str, dict], out: Path, model: str
) -> None:
    """Write SWE-bench predictions JSONL for upstream harness scoring."""
    pred_file = out / "predictions.jsonl"
    with open(pred_file, "w") as f:
        for name in sorted(results):
            r = results[name]
            if r.get("status") != "done":
                continue
            f.write(json.dumps({
                "instance_id": r.get("instance_id", name),
                "model_patch": r.get("model_patch", ""),
                "model_name_or_path": model,
            }, ensure_ascii=False) + "\n")
    typer.echo(f"\nPredictions written to: {pred_file}")
