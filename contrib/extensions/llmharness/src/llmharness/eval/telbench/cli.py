"""TELBench evaluation CLI.

Usage::

    llmharness-eval --data datasets/TELBench/TELBench.jsonl \\
        --mode posthoc --model litellm-dsv4flash --limit 10
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger

from .adapter import TelBenchInstance, load_telbench
from .runner import EvalResult, evaluate_instance, evaluate_instance_tel, reflect_on_result
from .scoring import AggregateScores, SpanScores, aggregate_scores

try:
    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.lib import resolve_model_profile
    from agentm.env import autoload_dotenv
    _HAS_AGENTM = True
except ImportError:
    DEFAULT_PROVIDER_REGISTRY = None  # type: ignore[assignment]
    resolve_model_profile = None  # type: ignore[assignment]
    autoload_dotenv = None  # type: ignore[assignment]
    _HAS_AGENTM = False
    logger.debug("telbench.cli: agentm SDK not available; model profile resolution disabled")

app = typer.Typer(help="TELBench span-level error localization evaluation.")


def _resolve_provider(
    provider_spec: str | None,
    model_name: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Resolve a provider from a ``--model`` profile name, explicit spec, or env."""
    if model_name and _HAS_AGENTM:
        profile = resolve_model_profile(model_name)
        if profile is not None:
            return DEFAULT_PROVIDER_REGISTRY.build(  # noqa: F823 — guarded by _HAS_AGENTM
                profile.provider, profile.to_build_config(), env=os.environ
            )

    if not provider_spec:
        if _HAS_AGENTM:
            profile = resolve_model_profile(None)
            if profile is not None:
                return DEFAULT_PROVIDER_REGISTRY.build(
                    profile.provider, profile.to_build_config(), env=os.environ
                )
        return None

    if ":" in provider_spec:
        module, payload = provider_spec.split(":", 1)
        cfg = json.loads(payload)
        if not isinstance(cfg, dict):
            raise typer.BadParameter("--provider config JSON must be an object")
        return module.strip(), cfg

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
    model: Annotated[str | None, typer.Option("--model", help="config.toml profile name")] = None,
    auditor_prompt: Annotated[
        str, typer.Option("--auditor-prompt", help="Auditor prompt variant name")
    ] = "minimal_index",
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-j", help="Max parallel instances")
    ] = 1,
    instance_ids: Annotated[
        Path | None,
        typer.Option("--instance-ids", help="JSON file with list of instance IDs to run"),
    ] = None,
    reflect: Annotated[
        bool,
        typer.Option("--reflect", help="Run reflection on wrong cases after eval"),
    ] = False,
) -> None:
    """Run TELBench evaluation with the cognitive-audit pipeline or TEL agent."""
    # Load .env the same way the ``agentm`` CLI does, so child sessions inherit
    # OTEL_EXPORTER_OTLP_ENDPOINT (trajectories ship to ClickHouse) and the
    # model/provider profile without the caller having to ``source .env`` first.
    if _HAS_AGENTM:
        autoload_dotenv()

    if mode not in ("posthoc", "online", "tel"):
        typer.echo(
            f"Error: --mode must be 'posthoc', 'online', or 'tel', got '{mode}'",
            err=True,
        )
        raise typer.Exit(1)

    instances = load_telbench(data)
    typer.echo(f"Loaded {len(instances)} instances from {data}")

    if difficulty != "all":
        instances = [i for i in instances if i.meta.get("difficulty") == difficulty]
        typer.echo(f"Filtered to {len(instances)} instances (difficulty={difficulty})")

    if answer_status != "all":
        instances = [i for i in instances if i.meta.get("answer_status") == answer_status]
        typer.echo(f"Filtered to {len(instances)} instances (answer_status={answer_status})")

    if instance_ids is not None:
        with open(instance_ids, encoding="utf-8") as f:
            ids_to_run = set(json.load(f))
        instances = [i for i in instances if i.id in ids_to_run]
        typer.echo(f"Filtered to {len(instances)} instances by --instance-ids")

    if offset > 0:
        instances = instances[offset:]
        typer.echo(f"Skipped {offset}, {len(instances)} remaining")

    if limit is not None:
        instances = instances[:limit]
        typer.echo(f"Limited to {len(instances)} instances")

    resolved_provider = _resolve_provider(provider, model_name=model)
    resolved_cwd = str(cwd.resolve())
    total = len(instances)
    typer.echo(f"Running {total} instances with concurrency={concurrency}")

    eval_mode: Any = mode

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")  # truncate; lines appended live

    async def _run_all() -> tuple[list[SpanScores], list[tuple[EvalResult, TelBenchInstance]]]:
        sem = asyncio.Semaphore(concurrency)
        scores: list[SpanScores | None] = [None] * total
        wrong_cases: list[tuple[EvalResult, TelBenchInstance]] = []
        write_lock = asyncio.Lock()
        done_count = 0
        shutting_down = False

        async def _run_one(idx: int, inst: Any) -> None:
            nonlocal done_count, shutting_down
            if shutting_down:
                return
            async with sem:
                if shutting_down:
                    return
                inst_cwd = str(Path(resolved_cwd) / f"inst-{inst.id}")
                Path(inst_cwd).mkdir(parents=True, exist_ok=True)
                try:
                    if mode == "tel":
                        result = await evaluate_instance_tel(
                            inst,
                            provider=resolved_provider,
                            cwd=inst_cwd,
                            prompt_name=auditor_prompt
                            if auditor_prompt != "telbench"
                            else "default",
                        )
                    else:
                        result = await evaluate_instance(
                            inst,
                            mode=eval_mode,
                            provider=resolved_provider,
                            cwd=inst_cwd,
                            extractor_interval=extractor_interval,
                            audit_interval=audit_interval,
                            auditor_prompt=auditor_prompt,
                        )
                    scores[idx] = result.scores
                    if result.scores.f1 < 1.0:
                        async with write_lock:
                            wrong_cases.append((result, inst))
                    done_count += 1
                    fea = "Y" if result.scores.first_error_accurate else "N"
                    typer.echo(
                        f"  [{done_count}/{total}] {inst.id} "
                        f"P={result.scores.precision:.3f} R={result.scores.recall:.3f} "
                        f"F1={result.scores.f1:.3f} FEA={fea}"
                    )
                    if output is not None:
                        line = json.dumps(
                            {
                                "instance_id": result.instance_id,
                                "predicted_error_indices": sorted(result.predicted_error_indices),
                                "gold_error_indices": sorted(result.gold_error_indices),
                                "n_spans": result.n_spans,
                                "precision": result.scores.precision,
                                "recall": result.scores.recall,
                                "f1": result.scores.f1,
                                "first_error_accurate": result.scores.first_error_accurate,
                                "n_verdicts": len(result.verdicts),
                            },
                            ensure_ascii=False,
                        )
                        async with write_lock:
                            with open(output, "a", encoding="utf-8") as fh:
                                fh.write(line + "\n")
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    logger.debug("cli: caught exception: {}", exc)
                    done_count += 1
                    typer.echo(f"  [{done_count}/{total}] {inst.id} ERROR: {exc}", err=True)

        tasks = [asyncio.create_task(_run_one(i, inst)) for i, inst in enumerate(instances)]
        try:
            await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            shutting_down = True
            typer.echo("\n⏹ Interrupted — cancelling pending tasks…", err=True)
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            typer.echo(f"  {done_count}/{total} instances completed before interrupt.", err=True)
        return [s for s in scores if s is not None], wrong_cases

    async def _reflect_all(
        wrong: list[tuple[EvalResult, TelBenchInstance]],
    ) -> None:
        reflect_dir = Path(resolved_cwd) / "reflections"
        reflect_dir.mkdir(parents=True, exist_ok=True)
        total_wrong = len(wrong)
        typer.echo(f"\n--- Reflection ({total_wrong} wrong cases, concurrency={concurrency}) ---")

        reflect_sem = asyncio.Semaphore(concurrency)
        done_count = 0

        async def _reflect_one(result: EvalResult, inst: TelBenchInstance) -> None:
            nonlocal done_count
            async with reflect_sem:
                try:
                    text = await reflect_on_result(
                        result, inst,
                        provider=resolved_provider,
                        cwd=resolved_cwd,
                    )
                    fea_tag = "Y" if result.scores.first_error_accurate else "N"
                    score_header = (
                        f"> **Case score**: F1={result.scores.f1:.3f}"
                        f"  P={result.scores.precision:.3f}"
                        f"  R={result.scores.recall:.3f}"
                        f"  FEA={fea_tag}\n\n"
                    )
                    out_path = reflect_dir / f"{result.instance_id}.md"
                    out_path.write_text(score_header + text, encoding="utf-8")
                    done_count += 1
                    typer.echo(f"  [{done_count}/{total_wrong}] {result.instance_id} → {out_path}")
                except Exception as exc:
                    logger.debug("cli: caught exception: {}", exc)
                    done_count += 1
                    typer.echo(f"  [{done_count}/{total_wrong}] {result.instance_id} ERROR: {exc}", err=True)

        await asyncio.gather(*[_reflect_one(r, i) for r, i in wrong])
        typer.echo(f"Reflections written to {reflect_dir}")

    try:
        instance_scores, wrong_cases = asyncio.run(_run_all())
    except KeyboardInterrupt:
        typer.echo("\n⏹ Aborted.", err=True)
        instance_scores, wrong_cases = [], []

    if output is not None:
        typer.echo(f"\nPer-instance results written to {output}")

    agg = aggregate_scores(instance_scores)
    typer.echo(f"\n--- Aggregate ({agg.n_instances} instances, mode={mode}) ---")
    typer.echo(_format_aggregate(agg))

    if reflect and wrong_cases:
        try:
            asyncio.run(_reflect_all(wrong_cases))
        except KeyboardInterrupt:
            typer.echo("\n⏹ Reflection aborted.", err=True)


def main() -> None:
    """Entry point for the ``llmharness-eval`` script."""
    app()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
