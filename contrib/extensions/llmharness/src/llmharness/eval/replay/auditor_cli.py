"""CLI: offline auditor replay with trajectory_index.

Run the auditor over a recorded trajectory, feeding it a pre-built symbol
table from the trajectory_index extension.

Usage::

    uv run python -m llmharness.eval.replay.auditor_cli \\
        --session <session_id> \\
        --index /tmp/rca_nav_full/index.json \\
        --model doubao \\
        --audit-interval 3

    # Multiple sessions:
    uv run python -m llmharness.eval.replay.auditor_cli \\
        --session-file sessions.txt \\
        --index /tmp/rca_nav_full/index.json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger

app = typer.Typer(pretty_exceptions_enable=False)


def _load_messages(session_id: str) -> list[Any]:
    """Load trajectory messages from ClickHouse and deserialize."""
    from agentm.core.lib.message_codec import deserialize_payload
    from agentm.core.observability.clickhouse import get_url, session_entries

    url = get_url()
    if not url:
        logger.error("ClickHouse unavailable")
        return []
    entries = session_entries(url, session_id)
    if not entries:
        return []

    messages = []
    for entry in entries:
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        try:
            msg = deserialize_payload(payload)
        except Exception:
            continue
        from agentm.core.abi import AgentMessage
        if isinstance(msg, AgentMessage):
            messages.append(msg)
    return messages


def _load_index(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load symbols and references from a trajectory_index JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("symbols", []), data.get("references", [])


def _resolve_provider(model_name: str) -> tuple[str, dict[str, Any]] | None:
    from agentm.ai import DEFAULT_PROVIDER_DESCRIPTORS
    from agentm.core.lib import resolve_model_profile

    profile = resolve_model_profile(model_name)
    if profile is None:
        logger.error(f"model profile {model_name!r} not found")
        return None
    for desc in DEFAULT_PROVIDER_DESCRIPTORS:
        if desc.id == profile.provider:
            return (desc.extension_module, dict(profile.to_build_config()))
    logger.error(f"no extension module for provider {profile.provider!r}")
    return None


async def _run_one(
    session_id: str,
    symbols: list[dict[str, Any]],
    references: list[dict[str, Any]],
    provider: tuple[str, dict[str, Any]],
    audit_interval: int,
    cwd: str,
) -> dict[str, Any]:
    """Run offline auditor pipeline for one session. Returns summary dict."""
    messages = _load_messages(session_id)
    if not messages:
        return {"session_id": session_id, "error": "no messages", "verdicts": []}

    from llmharness.agents.auditor.context import load_auditor_prompt
    from llmharness.eval.replay.offline_driver import replay_pipeline_over_trajectory
    from llmharness.eval.replay.runner import AuditorSettings

    result = await replay_pipeline_over_trajectory(
        messages=messages,
        cwd=cwd,
        session_id=session_id,
        provider=provider,
        auditor_settings=AuditorSettings(base_prompt=load_auditor_prompt("minimal_index")),
        audit_interval=audit_interval,
        enable_auditor=True,
        stop_on_first_surface=False,
        symbols=symbols,
        references=references,
    )

    verdicts = []
    for v in result.state.recent_verdicts:
        verdicts.append({
            "surface_reminder": v.get("surface_reminder", False),
            "reminder_text": v.get("reminder_text", ""),
            "continuation_notes": v.get("continuation_notes", []),
        })

    surfaces = [
        {"turn_index": s.turn_index, "reminder_text": s.reminder_text}
        for s in result.surfaces
    ]

    return {
        "session_id": session_id,
        "message_count": len(messages),
        "auditor_steps": len(result.all_step_results),
        "surface_count": len(result.surfaces),
        "verdict_count": len(verdicts),
        "verdicts": verdicts,
        "surfaces": surfaces,
    }


@app.command()
def run(
    session: Annotated[list[str] | None, typer.Option(help="Session ID (repeatable)")] = None,
    session_file: Annotated[Path | None, typer.Option("--session-file", help="File with one session ID per line")] = None,
    index: Annotated[Path | None, typer.Option(help="trajectory_index JSON file")] = None,
    model: Annotated[str, typer.Option(help="Auditor model profile")] = "doubao",
    audit_interval: Annotated[int, typer.Option(help="Fire auditor every N turns")] = 3,
    output: Annotated[Path | None, typer.Option(help="Write results JSONL to file")] = None,
    concurrency: Annotated[int, typer.Option(help="Max concurrent sessions")] = 1,
) -> None:
    """Run offline auditor with trajectory_index on one or more sessions."""
    all_sessions: list[str] = list(session or [])
    if session_file:
        all_sessions.extend(
            line.strip() for line in session_file.read_text().splitlines() if line.strip()
        )
    if not all_sessions:
        logger.error("no sessions specified (use --session or --session-file)")
        raise typer.Exit(1)

    symbols: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    if index and index.exists():
        symbols, references = _load_index(index)
        logger.info(f"loaded index: {len(symbols)} symbols, {len(references)} references")
    else:
        logger.info("no index provided; auditor will use trajectory-only context")

    provider = _resolve_provider(model)
    if provider is None:
        raise typer.Exit(1)

    cwd = "/tmp/llmharness_auditor_replay"
    Path(cwd).mkdir(parents=True, exist_ok=True)

    async def _run_all() -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)

        async def _guarded(sid: str) -> dict[str, Any]:
            async with sem:
                logger.info(f"starting {sid}")
                try:
                    result = await _run_one(sid, symbols, references, provider, audit_interval, cwd)
                except Exception:
                    logger.exception(f"{sid}: auditor failed")
                    result = {"session_id": sid, "error": "auditor_failed", "verdicts": []}
                sc = result.get("surface_count", 0)
                vc = result.get("verdict_count", 0)
                logger.info(f"done {sid}: {vc} verdicts, {sc} surfaces")
                return result

        tasks = [_guarded(sid) for sid in all_sessions]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run_all())

    # Report
    total_verdicts = sum(r.get("verdict_count", 0) for r in results)
    total_surfaces = sum(r.get("surface_count", 0) for r in results)
    errors = sum(1 for r in results if r.get("error"))
    logger.info(
        f"complete: {len(results)} sessions, {total_verdicts} verdicts, "
        f"{total_surfaces} surfaces, {errors} errors"
    )

    for r in results:
        if r.get("error"):
            continue
        sid = r["session_id"]
        for i, v in enumerate(r.get("verdicts", [])):
            surface = v.get("surface_reminder", False)
            text = v.get("reminder_text", "")
            if surface and text:
                print(f"\n[{sid}] verdict {i+1} (surface=True):")
                print(f"  {text[:500]}")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"results written to {output}")


if __name__ == "__main__":
    app()
