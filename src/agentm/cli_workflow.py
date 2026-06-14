"""``agentm workflow`` subcommand — run or validate workflow scripts.

``agentm workflow run <script> [--args JSON] [--model M]``
  Create a minimal workflow session (operations + observability +
  artifact_store + workflow atoms), invoke WorkflowRunner.run_file,
  print the JSON result. No orchestrator manifest needed.

``agentm workflow validate <script>``
  Parse the script and report validation issues without executing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

app = typer.Typer(
    name="workflow",
    help="Run or validate workflow scripts.",
    no_args_is_help=True,
)

_WORKFLOW_EXTENSIONS: list[tuple[str, dict[str, object]]] = [
    ("agentm.extensions.builtin.operations", {"backend": "local"}),
    ("agentm.extensions.builtin.observability", {}),
    ("agentm.extensions.builtin.artifact_store", {}),
    ("agentm.extensions.builtin.workflow", {}),
]


@app.command()
def run(
    script: Annotated[
        Path,
        typer.Argument(
            help="Path to the workflow script (.py).",
            exists=True,
            dir_okay=False,
        ),
    ],
    args_json: Annotated[
        str | None,
        typer.Option(
            "--args",
            help="JSON object passed as ctx.args / the args global.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name or config.toml profile."),
    ] = None,
    cwd: Annotated[
        str | None,
        typer.Option("--cwd", help="Working directory for child agents."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress progress output."),
    ] = False,
) -> None:
    """Run a workflow script."""
    import asyncio

    workflow_args: dict[str, object] = {}
    if args_json is not None:
        try:
            parsed = json.loads(args_json)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"--args is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise typer.BadParameter("--args must be a JSON object")
        workflow_args = parsed

    resolved_cwd = cwd or os.environ.get("AGENTM_CWD") or os.getcwd()
    script_path = script.resolve()

    exit_code = asyncio.run(
        _run_async(script_path, workflow_args, resolved_cwd, model, quiet)
    )
    raise SystemExit(exit_code)


async def _run_async(
    script_path: Path,
    workflow_args: dict[str, object],
    resolved_cwd: str,
    model_flag: str | None,
    quiet: bool,
) -> int:
    from agentm.ai import DEFAULT_PROVIDER_REGISTRY
    from agentm.core.abi.events import EventBus
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.lib.user_config import (
        apply_reasoning_effort,
        resolve_provider_model,
    )
    from agentm.core.runtime.session import AgentSession
    from agentm.extensions.builtin.workflow import WorkflowPhaseEvent

    provider_id, model_name, profile = resolve_provider_model(
        model_flag=model_flag,
    )
    if profile is not None:
        build_config = profile.to_build_config()
    else:
        build_config = {"model": model_name}
    apply_reasoning_effort(build_config, None)

    provider_spec = DEFAULT_PROVIDER_REGISTRY.build(provider_id, build_config)

    bus = EventBus()

    if not quiet:
        def _on_phase(event: WorkflowPhaseEvent) -> None:
            logger.info("[{kind}] {text}", kind=event.kind, text=event.text)

        bus.on(WorkflowPhaseEvent.CHANNEL, _on_phase)

    config = AgentSessionConfig(
        cwd=resolved_cwd,
        provider=provider_spec,
        extensions=[(m, dict(c)) for m, c in _WORKFLOW_EXTENSIONS],
        auto_commit=False,
        bus=bus,
    )
    session = await AgentSession.create(config)
    try:
        runner = session.get_service("workflow_runner")
        if runner is None:
            logger.error("workflow_runner service not found in session")
            return 1

        result = await runner.run_file(script_path, workflow_args)
        output = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        print(output)

        if not quiet:
            summary = runner.last_run_summary
            logger.info("--- workflow summary ---")
            logger.info(
                "agents: {spawned} spawned, {ok} ok, {failed} failed, {retried} retried",
                spawned=summary["agents_spawned"],
                ok=summary["agents_succeeded"],
                failed=summary["agents_failed"],
                retried=summary["agents_retried"],
            )
            logger.info(
                "tokens: {spent} (in={inp}, out={out})",
                spent=summary["budget"]["spent"],
                inp=summary["budget"]["input_tokens"],
                out=summary["budget"]["output_tokens"],
            )
            logger.info("wall clock: {t:.1f}s", t=summary["wall_clock_s"])

        return 0

    except Exception as exc:
        logger.error("{exc_type}: {exc}", exc_type=type(exc).__name__, exc=exc)
        return 1
    finally:
        await session.shutdown()


@app.command()
def validate(
    script: Annotated[
        Path,
        typer.Argument(
            help="Path to the workflow script (.py).",
            exists=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """Validate a workflow script without executing it."""
    from agentm.extensions.builtin.workflow import (
        _detect_script_mode,
        _validate_script,
    )

    source = script.read_text(encoding="utf-8")
    mode = _detect_script_mode(source)
    issues = _validate_script(source, mode)

    print(f"mode: {mode}")
    if not issues:
        print("no issues found")
        raise SystemExit(0)

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    for issue in issues:
        marker = "ERROR" if issue.severity == "error" else "WARN"
        print(f"  {marker} line {issue.line}: {issue.message}")

    if errors:
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
        raise SystemExit(1)
    print(f"\n{len(warnings)} warning(s), no errors")
    raise SystemExit(0)


def main() -> None:
    app()
