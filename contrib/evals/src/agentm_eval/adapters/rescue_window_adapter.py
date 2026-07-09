"""Rescue-window adapter — wraps the rescue-window package for branching experiments.

Delegates to the rescue_window CLI; the adapter provides experiment lifecycle.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated, Optional

import typer

from agentm_eval.experiment import experiment_context
from agentm_eval.registry import register
from agentm_eval.result import TaskResult


class RescueWindowAdapter:
    name = "rescue-window"
    description = "Fork-at-prefix branching experiments"

    def create_cli(self) -> typer.Typer:
        cli = typer.Typer(
            name="rescue-window",
            help="Run rescue-window branching experiments.",
            add_completion=False,
        )

        @cli.command()
        def run(
            corpus: Annotated[Path, typer.Option(help="Corpus manifest JSON")],
            oracle_model: Annotated[Optional[str], typer.Option(help="config.toml profile for ORACLE_GROUNDED")] = None,
            actor_model: Annotated[Optional[str], typer.Option(help="config.toml profile overriding stored provider")] = None,
            adapter: Annotated[str, typer.Option(help="Scenario adapter name")] = "rca",
            preset: Annotated[str, typer.Option(help="Condition preset")] = "oracle-landscape",
            k: Annotated[int, typer.Option(help="Rollouts per intervention cell")] = 3,
            concurrency: Annotated[int, typer.Option("-j")] = 1,
            limit: Annotated[Optional[int], typer.Option(help="Max trajectories")] = None,
            exp_id: Annotated[Optional[str], typer.Option(help="Override experiment ID")] = None,
        ) -> None:
            """Run rescue-window experiments."""
            with experiment_context(
                "rescue-window", model=oracle_model, exp_id=exp_id,
                corpus=str(corpus), adapter=adapter, preset=preset, k=k,
            ) as exp:
                output_file = exp.artifacts_dir / "eval_units.jsonl"
                cmd = [
                    "rescue-window", "run",
                    "--corpus", str(corpus),
                    "--out", str(output_file),
                    "--adapter", adapter,
                    "--preset", preset,
                    "--k", str(k),
                    "-j", str(concurrency),
                ]
                if oracle_model:
                    cmd.extend(["--oracle-model", oracle_model])
                if actor_model:
                    cmd.extend(["--actor-model", actor_model])
                if limit is not None:
                    cmd.extend(["--limit", str(limit)])

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
                            task_id=record.get("unit_id", "unknown"),
                            status="pass" if record.get("outcome") == "rescued" else "fail",
                            score={k_: v for k_, v in record.items()
                                   if k_ in ("outcome", "score", "n_turns")},
                            session_ids=[record["session_id"]] if record.get("session_id") else [],
                            metadata=record,
                        ).to_dict())

        @cli.command()
        def aggregate(
            exp_id: Annotated[str, typer.Argument(help="Experiment ID")],
        ) -> None:
            """Aggregate results from a rescue-window experiment."""
            from agentm_eval.experiment import Experiment

            exp = Experiment.load(exp_id)
            eval_units = exp.artifacts_dir / "eval_units.jsonl"
            if not eval_units.is_file():
                typer.echo("No eval_units.jsonl found in artifacts.", err=True)
                raise typer.Exit(1)

            cmd = [
                "rescue-window", "aggregate",
                "--store", str(eval_units),
                "--out-prefix", str(exp.output_dir / "report"),
            ]
            subprocess.run(cmd, check=True)

        return cli


register("rescue-window", RescueWindowAdapter.description, RescueWindowAdapter)
