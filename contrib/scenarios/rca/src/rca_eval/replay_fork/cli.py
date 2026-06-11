"""Replay-fork CLI.

Example::

    uv run python -m rca_eval.replay_fork.cli \
        --session abc123 --session def456 \
        --data-dir datasets/ops-lite-clean/cases/batch-XXX \
        --harness-model doubao --agent-model doubao \
        --out runs/replay-fork/results.jsonl
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="replay-fork",
    help=__doc__,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.callback(invoke_without_command=True)
def run(
    session: Annotated[
        list[str],
        typer.Option("--session", help="Baseline session id(s) (repeatable)"),
    ],
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", help="Case data directory"),
    ],
    harness_model: Annotated[
        str, typer.Option("--harness-model", help="config.toml profile for extractor+auditor"),
    ] = "doubao",
    agent_model: Annotated[
        str, typer.Option("--agent-model", help="config.toml profile for continuation agent"),
    ] = "doubao",
    scenario: Annotated[
        str, typer.Option("--scenario", help="scenario for continuation sessions"),
    ] = "rca:baseline",
    extractor_interval: Annotated[
        int, typer.Option("--extractor-interval"),
    ] = 5,
    audit_interval: Annotated[
        int, typer.Option("--audit-interval"),
    ] = 5,
    auditor_prompt: Annotated[
        str, typer.Option("--auditor-prompt", help="auditor prompt variant"),
    ] = "minimal",
    max_turns: Annotated[
        int, typer.Option("--max-turns", help="max turns per fork continuation"),
    ] = 60,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-j"),
    ] = 1,
    out: Annotated[
        Path, typer.Option("--out", help="results JSONL path"),
    ] = Path("runs/replay-fork/results.jsonl"),
    cwd: Annotated[
        Path, typer.Option("--cwd"),
    ] = Path("."),
) -> None:
    """Run replay-fork over baseline sessions."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from agentm.core.runtime.session_manager import JsonlSessionStore

    from .api import replay_batch
    from .providers import build_profile_provider

    harness_provider = build_profile_provider(harness_model)
    agent_provider = build_profile_provider(agent_model)
    resolved_cwd = str(cwd.resolve())
    resolved_data = str(data_dir.resolve())

    obs_dir = Path(resolved_cwd) / ".agentm" / "observability"
    store = JsonlSessionStore(session_dir=obs_dir)

    cases = [(sid, resolved_data) for sid in session]
    typer.echo(
        f"# harness: {harness_provider[1].get('model')}\n"
        f"# agent:   {agent_provider[1].get('model')}\n"
        f"# auditor_prompt: {auditor_prompt}\n"
        f"# cases: {len(cases)}  concurrency: {concurrency}"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")

    def _on_result(result, done, total):  # type: ignore[no-untyped-def]
        fh.write(json.dumps(asdict(result), ensure_ascii=False, default=str) + "\n")
        fh.flush()
        tag = "fired" if result.fired else "silent"
        typer.echo(f"  [{done}/{total}] {result.case_id} {tag} ctrl={result.control_correct} iv={result.intervene_correct}")

    try:
        summary = asyncio.run(
            replay_batch(
                cases,
                store=store,
                harness_provider=harness_provider,
                agent_provider=agent_provider,
                scenario=scenario,
                cwd=resolved_cwd,
                extractor_interval=extractor_interval,
                audit_interval=audit_interval,
                auditor_prompt=auditor_prompt,
                max_turns=max_turns,
                concurrency=concurrency,
                on_result=_on_result,
            )
        )
    finally:
        fh.close()

    typer.echo(f"\n=== replay-fork ===\n{summary.format()}\n# results: {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
