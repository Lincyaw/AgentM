"""TELBench adapter — wraps llmharness-eval for span-level error localization.

Delegates to the llmharness.eval.telbench package; the adapter provides
experiment lifecycle and unified result recording.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated, Optional

import typer

from agentm_eval.experiment import experiment_context
from agentm_eval.registry import register
from agentm_eval.result import TaskResult


class TelBenchAdapter:
    name = "telbench"
    description = "TELBench span-level error localization evaluation"

    def create_cli(self) -> typer.Typer:
        cli = typer.Typer(
            name="telbench",
            help="Run TELBench evaluation via llmharness.",
            add_completion=False,
        )

        @cli.command()
        def run(
            data: Annotated[Path, typer.Option(help="TELBench data directory")],
            model: Annotated[Optional[str], typer.Option(help="Model profile")] = None,
            concurrency: Annotated[int, typer.Option("-j")] = 4,
            limit: Annotated[Optional[int], typer.Option("-n")] = None,
            exp_id: Annotated[Optional[str], typer.Option(help="Override experiment ID")] = None,
        ) -> None:
            """Run TELBench evaluation."""
            with experiment_context(
                "telbench", model=model, exp_id=exp_id,
            ) as exp:
                output_file = exp.artifacts_dir / "results.jsonl"
                cmd = ["llmharness-eval", "--data", str(data), "--output", str(output_file)]
                if model:
                    cmd.extend(["--model", model])
                cmd.extend(["-j", str(concurrency)])
                if limit:
                    cmd.extend(["-n", str(limit)])

                result = subprocess.run(cmd)
                if result.returncode != 0:
                    exp.finish(status="failed", summary={"error": f"exit code {result.returncode}"})
                    raise typer.Exit(result.returncode)

                if output_file.is_file():
                    import json

                    for line in output_file.read_text().splitlines():
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        exp.record_result(TaskResult(
                            task_id=record.get("instance_id", "unknown"),
                            status="pass" if record.get("score", 0) > 0 else "fail",
                            score={k: v for k, v in record.items()
                                   if k in ("score", "precision", "recall", "f1")},
                            metadata=record,
                        ).to_dict())

        @cli.command()
        def iterate(
            data: Annotated[Path, typer.Option(help="TELBench data directory")],
            model: Annotated[Optional[str], typer.Option(help="Model profile")] = None,
            iterations: Annotated[int, typer.Option("-i")] = 5,
            exp_id: Annotated[Optional[str], typer.Option(help="Override experiment ID")] = None,
        ) -> None:
            """Run iterative eval→reflect→evolve loop."""
            with experiment_context(
                "telbench-iterate", model=model, exp_id=exp_id,
                iterations=iterations,
            ) as exp:
                runs_dir = exp.artifacts_dir / "iterations"
                cmd = [
                    "llmharness-iterate",
                    "--data", str(data),
                    "--runs-dir", str(runs_dir),
                    "--iterations", str(iterations),
                ]
                if model:
                    cmd.extend(["--model", model])

                result = subprocess.run(cmd)
                if result.returncode != 0:
                    exp.finish(status="failed")
                    raise typer.Exit(result.returncode)

        return cli


register("telbench", TelBenchAdapter.description, TelBenchAdapter)
