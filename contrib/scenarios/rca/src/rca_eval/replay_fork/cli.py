"""Replay-fork CLI.

Session config (scenario, provider, data_dir) is auto-restored from
the source session. Only the harness model needs to be specified.

Example::

    uv run python -m rca_eval.replay_fork.cli \
        --session abc123 --session def456 \
        --harness-model doubao \
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
    harness_model: Annotated[
        str, typer.Option("--harness-model", help="config.toml profile for extractor+auditor"),
    ] = "doubao",
    auditor_prompt: Annotated[
        str, typer.Option("--auditor-prompt", help="auditor prompt variant"),
    ] = "minimal",
    extractor_interval: Annotated[
        int, typer.Option("--extractor-interval"),
    ] = 5,
    audit_interval: Annotated[
        int, typer.Option("--audit-interval"),
    ] = 5,
    max_turns: Annotated[
        int, typer.Option("--max-turns"),
    ] = 60,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-j"),
    ] = 1,
    out: Annotated[
        Path, typer.Option("--out", help="results JSONL path"),
    ] = Path("runs/replay-fork/results.jsonl"),
    obs_dir: Annotated[
        Path, typer.Option("--obs-dir", help="observability directory"),
    ] = Path(".agentm/observability"),
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

    harness_prov = build_profile_provider(harness_model)
    store = JsonlSessionStore(session_dir=obs_dir.resolve())

    typer.echo(
        f"# harness: {harness_prov[1].get('model')}\n"
        f"# auditor_prompt: {auditor_prompt}\n"
        f"# sessions: {len(session)}  concurrency: {concurrency}"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")

    def _on_result(result, done, total):  # type: ignore[no-untyped-def]
        fh.write(json.dumps(asdict(result), ensure_ascii=False, default=str) + "\n")
        fh.flush()
        tag = "FIRE" if result.fired else "----"
        ctrl = "Y" if result.control_correct else "N"
        iv = "Y" if result.intervene_correct else ("N" if result.intervene_correct is not None else "-")
        flip = ""
        if result.helped:
            flip = " HELPED"
        elif result.harmed:
            flip = " HARMED"
        typer.echo(f"  [{done}/{total}] {tag} ctrl={ctrl} iv={iv}{flip} {result.case_id}")
        typer.echo(f"    baseline: agentm trace messages --session {result.case_id} --format text")
        if result.forked_session_id:
            typer.echo(f"    fork:     agentm trace messages --session {result.forked_session_id} --format text")
        for f in getattr(result, "audit_firings", []):
            ext = f.extractor_session_id or "-"
            aud = f.auditor_session_id or "-"
            sfx = " ★" if f.surfaced else ""
            typer.echo(f"    turn {f.turn_number:>3}: ext={ext} aud={aud}{sfx}")

    try:
        summary = asyncio.run(
            replay_batch(
                session,
                store=store,
                harness_provider=harness_prov,
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
