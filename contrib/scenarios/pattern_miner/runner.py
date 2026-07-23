# code-health: ignore-file[AM025] -- host program validates untyped session messages and JSON rows
"""Batch runner for the pattern miner (host program).

Discovers Harbor trials in one or more jobs directories, exports each into a
case bundle, then runs one miner session per (case, dimension) plus a
root-attribution session per case. Findings land in
``<out>/patterns.jsonl``; per-case detail in ``<out>/cases/<trial>/``.

The batch trajectory store is read through --dsn/--schema; the miner's own
sessions journal to the same Postgres under --miner-schema (default
``pattern_miner``) so the batch schema stays pure case data.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import typer
import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from export import export_case, read_reward  # noqa: E402
from prompts import (  # noqa: E402
    ANGLES,
    build_angle_prompt,
    build_attribution_prompt,
    build_distill_prompt,
)
from schema import (  # noqa: E402
    AngleReport,
    ChecklistDraft,
    RootAttribution,
)

from agentm import AgentSession, AgentSessionConfig, LoopConfig  # noqa: E402
from agentm.core.abi import (  # noqa: E402
    AssistantMessage,
    FunctionTool,
    JsonValue,
    ToolCallBlock,
    ToolResult,
)
from agentm.core.abi.messages import thaw_json  # noqa: E402
from agentm.core.lib import (  # noqa: E402
    error_result,
    pydantic_to_tool_schema,
    text_result,
)
from agentm.scenarios import builtin_scenario_loader  # noqa: E402
from agentm.storage.trajectory.resolve import resolve_trajectory_store  # noqa: E402

app = typer.Typer(add_completion=False)


def _extract_submission(messages: list[object]) -> dict[str, object] | None:
    for message in reversed(messages):
        if not isinstance(message, AssistantMessage):
            continue
        for block in message.content:
            if isinstance(block, ToolCallBlock) and block.name == "submit_result":
                result = thaw_json(block.arguments.get("result"))
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError:
                        return None
                return result if isinstance(result, dict) else None
    return None


async def _run_structured(
    *,
    bundle_dir: Path,
    prompt: str,
    result_schema: dict[str, object],
    provider_config: dict[str, object],
    max_tool_calls: int,
    purpose: str,
    miner_store_env: dict[str, str],
    extra_tools: list[FunctionTool] | None = None,
    scenario: str = "pattern_miner",
) -> dict[str, object] | None:
    # The session journals to the miner schema via an explicitly resolved
    # store — no process-global env mutation; the host owns the lifecycle.
    resolved = resolve_trajectory_store(env=miner_store_env)
    try:
        session = await AgentSession.create(
            AgentSessionConfig(
                cwd=str(bundle_dir),
                scenario=scenario,
                scenario_loader=builtin_scenario_loader,
                provider=("agentm.extensions.builtin.llm_openai", provider_config),
                extra_extensions=[
                    (
                        "agentm.extensions.builtin.structured_output",
                        {"schema": result_schema},
                    )
                ],
                extra_tools=list(extra_tools or []),
                purpose=purpose,
                loop_config=LoopConfig(max_tool_calls=max_tool_calls),
                trajectory_store=resolved.store if resolved is not None else None,
            )
        )
        try:
            messages = await session.run(prompt)
        finally:
            await session.shutdown()
    finally:
        if resolved is not None:
            resolved.close()
    return _extract_submission(list(messages))


async def _run_validated(
    model_cls: type[BaseModel], *, label: str, **run_kwargs: object
) -> dict[str, object]:
    """Run one structured session; normalize the outcome to a plain dict.

    Returns the validated ``model_dump()``, or ``{"error": ...}`` (with the
    raw submission attached on schema mismatch).
    """

    try:
        submission = await _run_structured(**run_kwargs)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        logger.error("{}: session failed: {}", label, exc)
        return {"error": str(exc)}
    if submission is None:
        logger.warning("{}: ended without submit_result", label)
        return {"error": "no submit_result call in session"}
    try:
        return model_cls.model_validate(submission).model_dump()
    except ValidationError as exc:
        logger.warning("{}: schema mismatch", label)
        return {"error": f"schema mismatch: {exc}", "raw": submission}


def _ensure_pg_schema(dsn: str, schema: str) -> None:
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )


def _discover_trials(jobs_dirs: list[Path]) -> list[Path]:
    trials: list[Path] = []
    for jobs_dir in jobs_dirs:
        if not jobs_dir.is_dir():
            raise typer.BadParameter(f"not a directory: {jobs_dir}")
        for child in sorted(jobs_dir.iterdir()):
            if child.is_dir() and (child / "result.json").is_file():
                trials.append(child)
    return trials


class _EnvBashArgs(BaseModel):
    cmd: str = Field(description="Shell command to run inside the task environment")
    timeout: int = Field(
        default=300, description="Seconds before the command is killed"
    )
    cwd: str | None = Field(
        default=None, description="Working directory; defaults to the task workdir"
    )


def _env_bash_tool(env: object) -> FunctionTool:
    async def _run(args: dict[str, JsonValue]) -> ToolResult:
        try:
            parsed = _EnvBashArgs.model_validate(dict(args))
        except ValidationError as exc:
            return error_result(f"bad arguments: {exc}")
        try:
            result = await env.exec(  # pyright: ignore[reportAttributeAccessIssue]
                parsed.cmd, cwd=parsed.cwd, timeout_sec=parsed.timeout
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("env_bash failed: {}", exc)
            return error_result(f"environment error: {exc}")
        parts = [f"exit={result.return_code}"]
        if result.stdout:
            parts.append(f"stdout (tail):\n{result.stdout[-8000:]}")
        if result.stderr:
            parts.append(f"stderr (tail):\n{result.stderr[-4000:]}")
        return text_result("\n".join(parts))

    return FunctionTool(
        name="env_bash",
        description=(
            "Run a shell command inside a live copy of the task environment: "
            "the repository in its pre-fix state (what the original agent "
            "started from), with the agent's final patch at /tmp/agent.patch "
            "and the oracle patch at /tmp/oracle.patch. Apply or revert "
            "patches, build, run tests, and probe behavior yourself."
        ),
        parameters=pydantic_to_tool_schema(_EnvBashArgs),
        fn=_run,
    )


async def _start_case_env(
    *,
    trial_name: str,
    task_path: Path | None,
    bundle_dir: Path,
    gateway_url: str,
    image_registry: str,
    image_tag: str,
    work_dir: Path,
) -> object:
    import tomllib
    import uuid

    from arl.harbor import ArlEnvironment
    from harbor.models.task.config import TaskConfig
    from harbor.models.trial.paths import TrialPaths

    if task_path is None or not (task_path / "task.toml").is_file():
        raise RuntimeError(f"task path unavailable for {trial_name}")
    task_config = TaskConfig.model_validate(
        tomllib.loads((task_path / "task.toml").read_text(encoding="utf-8"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    env = ArlEnvironment(
        environment_dir=task_path / "environment",
        environment_name=task_path.name,
        session_id=f"miner-{trial_name}-{uuid.uuid4().hex[:6]}",
        trial_paths=TrialPaths(trial_dir=work_dir),
        task_env_config=task_config.environment,
        gateway_url=gateway_url,
        image_registry=image_registry,
        image_tag=image_tag,
    )
    await env.start(force_build=False)
    for name, target in (
        ("agent.patch", "/tmp/agent.patch"),
        ("oracle.patch", "/tmp/oracle.patch"),
    ):
        source = bundle_dir / name
        if source.is_file():
            await env.upload_file(source, target)
    return env


def _run_case(
    *,
    trial_dir: Path,
    out_dir: Path,
    batch_store_env: dict[str, str],
    miner_store_env: dict[str, str],
    angles: list[str],
    provider_config: dict[str, object],
    max_tool_calls: int,
    refresh_bundle: bool,
    angle_concurrency: int,
    with_env: bool,
    gateway_url: str,
    image_registry: str,
    image_tag: str,
) -> dict[str, object]:
    case_dir = out_dir / "cases" / trial_dir.name
    bundle_dir = case_dir / "bundle"
    meta_path = bundle_dir / "meta.json"
    if refresh_bundle or not meta_path.is_file():
        resolved = resolve_trajectory_store(env=batch_store_env)
        try:
            store = resolved.store if resolved is not None else None
            meta = export_case(trial_dir, bundle_dir, store=store)
        finally:
            if resolved is not None:
                resolved.close()
    else:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = loaded if isinstance(loaded, dict) else {}

    # Merge with any previous run of this case: angle runs are incremental,
    # a fresh run of an angle overwrites only that angle's report.
    reports: dict[str, object] = {}
    report_path = case_dir / "report.json"
    if report_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cannot merge previous report {}: {}", report_path, exc)
        else:
            if isinstance(previous, dict) and isinstance(previous.get("reports"), dict):
                reports.update(previous["reports"])
    record: dict[str, object] = {"meta": meta, "reports": reports, "attribution": None}

    if meta.get("trajectory_error"):
        record["skipped"] = f"trajectory unavailable: {meta['trajectory_error']}"
        return record

    g0_invalid = bool(meta.get("g0_reasons"))
    case_angles = ["label_validity"] if g0_invalid else angles
    if g0_invalid:
        record["skipped"] = (
            "label invalid (mechanical G0); only label_validity angle run: "
            + "; ".join(str(reason) for reason in meta.get("g0_reasons", []))
        )

    async def _mine() -> None:
        env: object | None = None
        if with_env and not g0_invalid:
            try:
                raw_task_path = meta.get("task_path")
                env = await _start_case_env(
                    trial_name=trial_dir.name,
                    task_path=Path(str(raw_task_path)) if raw_task_path else None,
                    bundle_dir=bundle_dir,
                    gateway_url=gateway_url,
                    image_registry=image_registry,
                    image_tag=image_tag,
                    work_dir=case_dir / "env-work",
                )
                logger.info("{} | live task environment attached", trial_dir.name)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "{} | environment start failed; mining without it: {}",
                    trial_dir.name,
                    exc,
                )
        extra_tools = [_env_bash_tool(env)] if env is not None else None
        env_attached = env is not None
        # A live environment is shared, stateful, and mutated by probes;
        # angle sessions must not interleave inside it.
        semaphore = asyncio.Semaphore(1 if env_attached else angle_concurrency)

        async def _one_angle(position: int, angle: str) -> None:
            async with semaphore:
                logger.info(
                    "{} | angle {}/{} {}: session starting",
                    trial_dir.name,
                    position,
                    len(case_angles),
                    angle,
                )
                report = await _run_validated(
                    AngleReport,
                    label=f"{trial_dir.name} | angle {angle}",
                    bundle_dir=bundle_dir,
                    prompt=build_angle_prompt(angle, meta, env_attached=env_attached),
                    result_schema=pydantic_to_tool_schema(AngleReport),
                    provider_config=provider_config,
                    max_tool_calls=max_tool_calls,
                    purpose=f"pattern-miner:{angle}",
                    miner_store_env=miner_store_env,
                    extra_tools=extra_tools,
                )
                reports[angle] = report
                if "error" not in report:
                    logger.info(
                        "{} | angle {}: {} finding(s)",
                        trial_dir.name,
                        angle,
                        len(report.get("findings", [])),
                    )

        try:
            await asyncio.gather(
                *(
                    _one_angle(position, angle)
                    for position, angle in enumerate(case_angles, start=1)
                )
            )

            if g0_invalid:
                return
            fired = [
                angle
                for angle, report in reports.items()
                if isinstance(report, dict) and report.get("findings")
            ]
            if not fired:
                return
            logger.info(
                "{} | attribution over {} fired dimension(s)",
                trial_dir.name,
                len(fired),
            )
            reports_json = json.dumps(
                {angle: reports[angle] for angle in fired},
                ensure_ascii=False,
                indent=1,
            )
            record["attribution"] = await _run_validated(
                RootAttribution,
                label=f"{trial_dir.name} | attribution",
                bundle_dir=bundle_dir,
                prompt=build_attribution_prompt(
                    meta, reports_json, env_attached=env_attached
                ),
                result_schema=pydantic_to_tool_schema(RootAttribution),
                provider_config=provider_config,
                max_tool_calls=max_tool_calls,
                purpose="pattern-miner:attribution",
                miner_store_env=miner_store_env,
                extra_tools=extra_tools,
            )
        finally:
            if env is not None:
                try:
                    await env.stop(  # pyright: ignore[reportAttributeAccessIssue]
                        delete=True
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "{} | environment stop failed: {}", trial_dir.name, exc
                    )

    asyncio.run(_mine())

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "report.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return record


def _load_case_records(out: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for report_path in sorted((out / "cases").glob("*/report.json")):
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cannot read case report {}: {}", report_path, exc)
            continue
        if isinstance(loaded, dict):
            records.append(loaded)
    return records


def _iter_findings(
    record: dict[str, object],
) -> Iterator[tuple[dict[str, object], str, object, dict[str, object]]]:
    """Yield (meta, angle, root_dimension, finding) with the guards applied."""

    meta = record.get("meta")
    reports = record.get("reports")
    if not isinstance(meta, dict) or not isinstance(reports, dict):
        return
    attribution = record.get("attribution")
    root = (
        attribution.get("root_dimension")
        if isinstance(attribution, dict) and "error" not in attribution
        else None
    )
    for angle, report in reports.items():
        if not isinstance(report, dict):
            continue
        findings = report.get("findings")
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if isinstance(finding, dict):
                yield meta, angle, root, finding


def _pattern_rows(record: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "task": meta.get("task_name"),
            "trial": meta.get("trial_name"),
            "agent_model": meta.get("agent_model"),
            "reward": meta.get("reward"),
            "dimension": angle,
            "root_dimension": root,
            **finding,
        }
        for meta, angle, root, finding in _iter_findings(record)
    ]


@app.command()
def mine(
    jobs_dirs: Annotated[
        list[Path],
        typer.Argument(help="Harbor jobs batch directories (jobs/<timestamp>)"),
    ],
    out: Annotated[Path, typer.Option("--out", help="Output directory")],
    dsn: Annotated[
        str, typer.Option("--dsn", help="Trajectory store DSN of the batch")
    ],
    schema_name: Annotated[
        str, typer.Option("--schema", help="Trajectory store schema of the batch")
    ],
    model: Annotated[
        str, typer.Option("--model", help="Miner model (OpenAI-compatible name)")
    ],
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Miner model endpoint")
    ] = None,
    angle: Annotated[
        list[str] | None,
        typer.Option("--angle", help="Subset of angles; default all"),
    ] = None,
    include_passes: Annotated[
        bool, typer.Option("--include-passes", help="Also mine passing trials")
    ] = False,
    fail_below: Annotated[
        float, typer.Option("--fail-below", help="Reward below this counts as failed")
    ] = 1.0,
    concurrency: Annotated[int, typer.Option("--concurrency", "-n", min=1)] = 2,
    max_tool_calls: Annotated[int, typer.Option("--max-tool-calls", min=1)] = 60,
    limit: Annotated[
        int, typer.Option("--limit", min=0, help="Mine at most N cases; 0 = all")
    ] = 0,
    trial_filter: Annotated[
        str, typer.Option("--trial", help="Substring filter on trial names")
    ] = "",
    refresh_bundles: Annotated[
        bool, typer.Option("--refresh-bundles", help="Re-export existing bundles")
    ] = False,
    miner_schema: Annotated[
        str,
        typer.Option(
            "--miner-schema",
            help="Postgres schema where the miner's own sessions journal",
        ),
    ] = "pattern_miner",
    angle_concurrency: Annotated[
        int,
        typer.Option(
            "--angle-concurrency",
            min=1,
            help="Parallel angle sessions per case (angles are mutually blind)",
        ),
    ] = 4,
    with_env: Annotated[
        bool,
        typer.Option(
            "--with-env",
            help=(
                "Attach a live ARL task environment per case so the miner can "
                "probe (forces serial angles within the case)"
            ),
        ),
    ] = False,
    gateway_url: Annotated[
        str,
        typer.Option("--gateway-url", help="ARL gateway; defaults to $ARL_GATEWAY_URL"),
    ] = "",
    image_registry: Annotated[
        str,
        typer.Option(
            "--image-registry",
            help="Task image registry; defaults to $ARL_IMAGE_REGISTRY",
        ),
    ] = "",
    image_tag: Annotated[
        str, typer.Option("--image-tag", help="Task image tag")
    ] = "v2026.06",
) -> None:
    """Mine failure patterns from Harbor trials along the feedback dimensions."""

    angles = list(angle) if angle else list(ANGLES)
    unknown = sorted(set(angles) - set(ANGLES))
    if unknown:
        raise typer.BadParameter(f"unknown angles: {', '.join(unknown)}")
    if with_env and not (gateway_url or os.environ.get("ARL_GATEWAY_URL")):
        raise typer.BadParameter("--with-env needs --gateway-url or $ARL_GATEWAY_URL")

    # Miner sessions journal to the same Postgres but in their own schema —
    # never the batch schema, whose sessions schema-enumerating consumers
    # (policy backfill, future mining rounds) treat as case data.
    if miner_schema == schema_name:
        raise typer.BadParameter("--miner-schema must differ from --schema")
    _ensure_pg_schema(dsn, miner_schema)
    batch_store_env = {
        "AGENTM_TRAJECTORY_DSN": dsn,
        "AGENTM_TRAJECTORY_SCHEMA": schema_name,
    }
    miner_store_env = {
        "AGENTM_TRAJECTORY_DSN": dsn,
        "AGENTM_TRAJECTORY_SCHEMA": miner_schema,
    }

    provider_config: dict[str, object] = {"model": model, "name": model}
    if base_url:
        provider_config["base_url"] = base_url

    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "mine.log", level="INFO", enqueue=True)
    logger.info(
        "logs: {} | miner sessions journal to schema {!r} at the batch DSN",
        out / "mine.log",
        miner_schema,
    )
    trials = _discover_trials(jobs_dirs)
    if trial_filter:
        trials = [t for t in trials if trial_filter in t.name]

    # Selection needs only the reward; the expensive export (trajectory
    # load + bundle write) happens in _run_case, concurrently, for the
    # cases that were actually selected.
    selected: list[Path] = []
    skipped_rows: list[dict[str, object]] = []
    for trial_dir in trials:
        reward, _ = read_reward(trial_dir)
        is_failed = reward is None or reward < fail_below
        if not is_failed and not include_passes:
            skipped_rows.append(
                {"trial": trial_dir.name, "reason": f"pass (reward={reward})"}
            )
            continue
        selected.append(trial_dir)
    if limit:
        selected = selected[:limit]

    logger.info(
        "mining {} cases ({} skipped) along {} angles with {}",
        len(selected),
        len(skipped_rows),
        len(angles),
        model,
    )

    def _one(trial_dir: Path) -> dict[str, object]:
        return _run_case(
            trial_dir=trial_dir,
            out_dir=out,
            batch_store_env=batch_store_env,
            miner_store_env=miner_store_env,
            angles=angles,
            provider_config=provider_config,
            max_tool_calls=max_tool_calls,
            refresh_bundle=refresh_bundles,
            angle_concurrency=angle_concurrency,
            with_env=with_env,
            gateway_url=gateway_url,
            image_registry=image_registry,
            image_tag=image_tag,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(_one, selected))

    # Rebuild the aggregate from every case report on disk, so incremental
    # runs never shrink patterns.jsonl to the current selection.
    all_records = _load_case_records(out)
    rows = [row for record in all_records for row in _pattern_rows(record)]
    with (out / "patterns.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for record in records:
        skipped = record.get("skipped")
        meta = record.get("meta")
        if skipped and isinstance(meta, dict):
            skipped_rows.append({"trial": meta.get("trial_name"), "reason": skipped})
    (out / "skipped.json").write_text(
        json.dumps(skipped_rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info(
        "done: {} findings across {} case report(s) ({} mined this run) -> {}",
        len(rows),
        len(all_records),
        len(records),
        out / "patterns.jsonl",
    )


def _collect_findings(out: Path) -> dict[str, list[dict[str, object]]]:
    by_dimension: dict[str, list[dict[str, object]]] = {}
    for record in _load_case_records(out):
        for meta, angle, root, finding in _iter_findings(record):
            by_dimension.setdefault(angle, []).append(
                {
                    "trial": meta.get("trial_name"),
                    "task": meta.get("task_name"),
                    "agent_model": meta.get("agent_model"),
                    "case_root_dimension": root,
                    **finding,
                }
            )
    return by_dimension


@app.command()
def distill(
    out: Annotated[
        Path,
        typer.Option("--out", help="Mining output directory (cases/*/report.json)"),
    ],
    model: Annotated[
        str, typer.Option("--model", help="Distiller model (OpenAI-compatible name)")
    ],
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Distiller model endpoint")
    ] = None,
    checklist: Annotated[
        Path | None,
        typer.Option(
            "--checklist",
            help="Checklist to merge into and rewrite; defaults to <out>/checklist.yaml",
        ),
    ] = None,
    dsn: Annotated[
        str,
        typer.Option("--dsn", help="Postgres DSN for the distiller's own sessions"),
    ] = "",
    miner_schema: Annotated[
        str, typer.Option("--miner-schema", help="Schema for distiller sessions")
    ] = "pattern_miner",
    concurrency: Annotated[int, typer.Option("--concurrency", "-n", min=1)] = 2,
) -> None:
    """Distill a batch's angle reports into the merged checklist."""

    out = out.resolve()
    checklist_path = (checklist or out / "checklist.yaml").resolve()
    logger.add(out / "distill.log", level="INFO", enqueue=True)

    miner_store_env: dict[str, str] = {}
    if dsn:
        _ensure_pg_schema(dsn, miner_schema)
        miner_store_env = {
            "AGENTM_TRAJECTORY_DSN": dsn,
            "AGENTM_TRAJECTORY_SCHEMA": miner_schema,
        }

    by_dimension = _collect_findings(out)
    if not by_dimension:
        raise typer.BadParameter(f"no findings under {out / 'cases'}")

    existing_items: list[dict[str, object]] = []
    if checklist_path.is_file():
        loaded = yaml.safe_load(checklist_path.read_text(encoding="utf-8")) or {}
        raw_items = loaded.get("items") if isinstance(loaded, dict) else None
        if isinstance(raw_items, list):
            existing_items = [item for item in raw_items if isinstance(item, dict)]

    provider_config: dict[str, object] = {"model": model, "name": model}
    if base_url:
        provider_config["base_url"] = base_url

    logger.info(
        "distilling {} dimension(s) ({} findings, {} existing items) -> {}",
        len(by_dimension),
        sum(len(v) for v in by_dimension.values()),
        len(existing_items),
        checklist_path,
    )

    results: dict[str, dict[str, object]] = {}

    async def _distill_all() -> None:
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(dimension: str, findings: list[dict[str, object]]) -> None:
            async with semaphore:
                current = [
                    item
                    for item in existing_items
                    if item.get("dimension") == dimension
                ]
                existing_json = (
                    json.dumps(current, ensure_ascii=False, indent=1)
                    if current
                    else None
                )
                logger.info(
                    "distill {}: {} findings, {} existing items",
                    dimension,
                    len(findings),
                    len(current),
                )
                result = await _run_validated(
                    ChecklistDraft,
                    label=f"distill {dimension}",
                    bundle_dir=out,
                    prompt=build_distill_prompt(
                        dimension,
                        json.dumps(findings, ensure_ascii=False, indent=1),
                        existing_json,
                    ),
                    result_schema=pydantic_to_tool_schema(ChecklistDraft),
                    provider_config=provider_config,
                    max_tool_calls=6,
                    purpose=f"pattern-miner:distill:{dimension}",
                    miner_store_env=miner_store_env,
                    scenario="pattern_miner:distill",
                )
                results[dimension] = result
                if "error" not in result:
                    logger.info(
                        "distill {}: {} item(s), dropped: {}",
                        dimension,
                        len(result.get("items", [])),
                        result.get("dropped") or "-",
                    )

        await asyncio.gather(
            *(_one(dimension, findings) for dimension, findings in by_dimension.items())
        )

    asyncio.run(_distill_all())

    merged: list[dict[str, object]] = [
        item
        for item in existing_items
        if item.get("dimension") not in results
        or "error" in results[str(item.get("dimension"))]
    ]
    for _dimension, result in sorted(results.items()):
        items = result.get("items")
        if isinstance(items, list):
            merged.extend(item for item in items if isinstance(item, dict))
    merged.sort(key=lambda item: (str(item.get("dimension")), str(item.get("check"))))

    checklist_path.parent.mkdir(parents=True, exist_ok=True)
    checklist_path.write_text(
        yaml.safe_dump(
            {"items": merged}, allow_unicode=True, sort_keys=False, width=88
        ),
        encoding="utf-8",
    )
    (out / "distill-report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    failed = sorted(d for d, r in results.items() if "error" in r)
    if failed:
        logger.warning("distill failed for: {}", ", ".join(failed))
    logger.info("checklist: {} item(s) -> {}", len(merged), checklist_path)


if __name__ == "__main__":
    app()
