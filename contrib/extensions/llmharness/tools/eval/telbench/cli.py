"""TELBench evaluation CLI.

Usage::

    llmharness-eval telbench --data datasets/TELBench/TELBench.jsonl \\
        --mode posthoc --limit 10
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from .adapter import load_telbench
from .runner import evaluate_instance
from .scoring import AggregateScores, SpanScores, aggregate_scores

app = typer.Typer(help="TELBench span-level error localization evaluation.")

def _resolve_provider(
    provider_spec: str | None,
    model_name: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Resolve a provider from a ``--model`` profile name, explicit spec, or env.

    Resolution order:

    1. ``model_name`` — matched against ``~/.agentm/config.toml`` profiles
       (e.g. ``litellm-dsv4flash``).
    2. ``provider_spec`` with ``"module:{json}"`` — explicit config.
    3. ``provider_spec`` bare name — registry lookup.
    4. ``None`` — fall back to ``AGENTM_PROVIDER`` / ``AGENTM_MODEL`` env vars.
    """
    if model_name:
        try:
            from agentm.ai import DEFAULT_PROVIDER_REGISTRY
            from agentm.core.lib import resolve_model_profile

            profile = resolve_model_profile(model_name)
            if profile is not None:
                registry = DEFAULT_PROVIDER_REGISTRY
                return registry.build(
                    profile.provider, profile.to_build_config(), env=os.environ
                )
        except ImportError:
            pass

    if not provider_spec:
        from llmharness.replay.cli import (  # type: ignore[import-untyped]
            resolve_default_provider_spec,
        )

        return resolve_default_provider_spec()  # type: ignore[no-any-return]

    if ":" in provider_spec:
        module, payload = provider_spec.split(":", 1)
        cfg = json.loads(payload)
        if not isinstance(cfg, dict):
            raise typer.BadParameter("--provider config JSON must be an object")
        return module.strip(), cfg

    # Bare provider id — try registry resolve.
    try:
        from agentm.ai import DEFAULT_PROVIDER_REGISTRY

        registry = DEFAULT_PROVIDER_REGISTRY
        descriptor = registry.resolve(provider_spec)
        if descriptor.extension_module is None:
            return provider_spec, {}
        model = os.environ.get("AGENTM_MODEL") or descriptor.default_model
        return registry.build(provider_spec, {"model": model}, env=os.environ)
    except (KeyError, ImportError):
        return provider_spec, {}

def _format_scores(scores: SpanScores) -> str:
    fea = "Y" if scores.first_error_accurate else "N"
    return f"P={scores.precision:.3f}  R={scores.recall:.3f}  F1={scores.f1:.3f}  FEA={fea}"

def _format_aggregate(agg: AggregateScores) -> str:
    return (
        f"  macro_precision:      {agg.macro_precision:.4f}\n"
        f"  macro_recall:         {agg.macro_recall:.4f}\n"
        f"  macro_f1:             {agg.macro_f1:.4f}\n"
        f"  first_error_accuracy: {agg.first_error_accuracy:.4f}\n"
        f"  n_instances:          {agg.n_instances}"
    )

@app.command()
def telbench(
    data: Annotated[Path, typer.Option("--data", help="Path to TELBench JSONL")],
    mode: Annotated[str, typer.Option("--mode", help="posthoc | online")] = "posthoc",
    extractor_interval: Annotated[
        int, typer.Option("--extractor-interval", help="Extractor firing interval in spans")
    ] = 5,
    audit_interval: Annotated[
        int, typer.Option("--audit-interval", help="Auditor firing interval in spans")
    ] = 5,
    limit: Annotated[int | None, typer.Option("--limit", help="Max instances to evaluate")] = None,
    difficulty: Annotated[str, typer.Option("--difficulty", help="easy | hard | all")] = "all",
    answer_status: Annotated[
        str, typer.Option("--answer-status", help="correct | incorrect | all")
    ] = "all",
    offset: Annotated[int, typer.Option("--offset", help="Skip first N instances")] = 0,
    output: Annotated[
        Path | None, typer.Option("--output", help="Write per-instance results as JSONL")
    ] = None,
    cwd: Annotated[Path, typer.Option("--cwd", help="Working directory for child sessions")] = Path(
        "."
    ),
    provider: Annotated[str | None, typer.Option("--provider", help="LLM provider spec")] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="config.toml profile name")
    ] = None,
    auditor_prompt: Annotated[
        str, typer.Option("--auditor-prompt", help="Auditor prompt variant name")
    ] = "telbench",
) -> None:
    """Run TELBench evaluation with the cognitive-audit pipeline."""
    if mode not in ("posthoc", "online"):
        typer.echo(f"Error: --mode must be 'posthoc' or 'online', got '{mode}'", err=True)
        raise typer.Exit(1)

    instances = load_telbench(data)
    typer.echo(f"Loaded {len(instances)} instances from {data}")

    if difficulty != "all":
        instances = [i for i in instances if i.meta.get("difficulty") == difficulty]
        typer.echo(f"Filtered to {len(instances)} instances (difficulty={difficulty})")

    if answer_status != "all":
        instances = [i for i in instances if i.meta.get("answer_status") == answer_status]
        typer.echo(f"Filtered to {len(instances)} instances (answer_status={answer_status})")

    if offset > 0:
        instances = instances[offset:]
        typer.echo(f"Skipped {offset}, {len(instances)} remaining")

    if limit is not None:
        instances = instances[:limit]
        typer.echo(f"Limited to {len(instances)} instances")

    resolved_provider = _resolve_provider(provider, model_name=model)
    resolved_cwd = str(cwd.resolve())

    # Validate mode literal for the type checker.
    eval_mode: Any = mode

    instance_scores: list[SpanScores] = []
    output_lines: list[str] = []

    for idx, inst in enumerate(instances):
        typer.echo(
            f"\n[{idx + 1}/{len(instances)}] {inst.id} ({len(inst.spans)} spans) ...", nl=False
        )
        sys.stdout.flush()
        try:
            result = asyncio.run(
                evaluate_instance(
                    inst,
                    mode=eval_mode,
                    provider=resolved_provider,
                    cwd=resolved_cwd,
                    extractor_interval=extractor_interval,
                    audit_interval=audit_interval,
                    auditor_prompt=auditor_prompt,
                )
            )
            instance_scores.append(result.scores)
            typer.echo(f"  {_format_scores(result.scores)}")

            if output is not None:
                record = {
                    "instance_id": result.instance_id,
                    "predicted_error_indices": sorted(result.predicted_error_indices),
                    "gold_error_indices": sorted(result.gold_error_indices),
                    "n_spans": result.n_spans,
                    "precision": result.scores.precision,
                    "recall": result.scores.recall,
                    "f1": result.scores.f1,
                    "first_error_accurate": result.scores.first_error_accurate,
                    "n_verdicts": len(result.verdicts),
                }
                output_lines.append(json.dumps(record, ensure_ascii=False))
        except Exception as exc:
            typer.echo(f"  ERROR: {exc}", err=True)

    if output is not None and output_lines:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(line + "\n")
        typer.echo(f"\nPer-instance results written to {output}")

    agg = aggregate_scores(instance_scores)
    typer.echo(f"\n--- Aggregate ({agg.n_instances} instances, mode={mode}) ---")
    typer.echo(_format_aggregate(agg))

def main() -> None:
    """Entry point for the ``llmharness-eval`` script."""
    app()

__all__ = ["app", "main"]
