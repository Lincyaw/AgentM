"""Replay-fork CLI: offline audit a baseline session, fork at surfaces, judge.

Example::

    uv run python -m rca_eval.replay_fork.cli run \
        --session abc123 \
        --data-dir datasets/ops-lite-clean/cases/batch-XXX \
        --harness-model doubao \
        --agent-model doubao \
        --out runs/replay-fork/results.jsonl
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(
    name="replay-fork",
    help=__doc__,
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.callback(invoke_without_command=True)
def run(
    session: Annotated[
        list[str],
        typer.Option("--session", help="Baseline session id(s) to replay (repeatable)"),
    ],
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", help="Case data directory (parquet + injection.json)"),
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
    max_depth: Annotated[
        int, typer.Option("--max-depth", help="max recursive fork depth"),
    ] = 3,
    out: Annotated[
        Path, typer.Option("--out", help="results JSONL path"),
    ] = Path("runs/replay-fork/results.jsonl"),
    cwd: Annotated[
        Path, typer.Option("--cwd", help="working directory for sessions"),
    ] = Path("."),
) -> None:
    """Run replay-fork over one or more baseline sessions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from .providers import build_profile_provider

    harness_provider = build_profile_provider(harness_model)
    agent_provider = build_profile_provider(agent_model)
    resolved_cwd = str(cwd.resolve())

    typer.echo(
        f"# harness: {harness_provider[1].get('model')}\n"
        f"# agent:   {agent_provider[1].get('model')}\n"
        f"# scenario: {scenario}\n"
        f"# sessions: {len(session)}"
    )

    out.parent.mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(
        _run_all(
            session_ids=session,
            data_dir=str(data_dir.resolve()),
            harness_provider=harness_provider,
            agent_provider=agent_provider,
            scenario=scenario,
            max_depth=max_depth,
            cwd=resolved_cwd,
            out=out,
        )
    )

    typer.echo(f"\n=== replay-fork summary ===\n{summary}\n# results: {out}")


async def _run_all(
    *,
    session_ids: list[str],
    data_dir: str,
    harness_provider: tuple[str, dict[str, Any]],
    agent_provider: tuple[str, dict[str, Any]],
    scenario: str,
    max_depth: int,
    cwd: str,
    out: Path,
) -> str:
    import os

    from agentm.core.abi import AgentSessionConfig
    from agentm.core.abi.loop import LoopConfig
    from agentm.core.runtime import AgentSession, create_agent_session
    from agentm.core.runtime.session_manager import JsonlSessionStore
    from llmharness import offline_audit

    from .judge import RcabenchJudge

    obs_dir = Path(cwd) / ".agentm" / "observability"
    store = JsonlSessionStore(session_dir=obs_dir)
    judge = RcabenchJudge()

    total = 0
    fired = 0
    control_correct = 0
    intervene_correct = 0
    flips_helped = 0
    flips_harmed = 0

    fh = out.open("w", encoding="utf-8")
    try:
        for sid in session_ids:
            total += 1
            source = store.open(sid)
            messages = source.get_raw_messages()

            # Judge control (baseline)
            control_response = _extract_submission(messages)
            os.environ["AGENTM_RCA_DATA_DIR"] = data_dir
            ctrl_outcome = await judge.judge(
                agent_output_json=control_response,
                data_dir=data_dir,
                case_id=sid,
            )
            ctrl_ok = ctrl_outcome.correct
            if ctrl_ok:
                control_correct += 1

            # Offline audit
            surfaces = await offline_audit(
                messages,
                cwd=cwd,
                provider=harness_provider,
            )

            if not surfaces:
                result = {
                    "case_id": sid,
                    "fired": False,
                    "control_correct": ctrl_ok,
                    "intervene_correct": ctrl_ok,
                }
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                fh.flush()
                logging.getLogger(__name__).info(
                    "[%d/%d] %s fired=False control=%s",
                    total, len(session_ids), sid, ctrl_ok,
                )
                continue

            # Fork at first surface
            s = surfaces[0]
            fired += 1
            forked = store.fork(sid, up_to=s.turn_index)
            config = AgentSessionConfig(
                cwd=cwd,
                session_manager=forked,
                scenario=scenario,
                provider=agent_provider,
                loop_config=LoopConfig(max_turns=60),
            )
            session = await create_agent_session(AgentSession, config)
            try:
                fork_messages = await session.prompt(s.reminder_text)
            finally:
                await session.shutdown()

            fork_response = _extract_submission(fork_messages)
            fork_outcome = await judge.judge(
                agent_output_json=fork_response,
                data_dir=data_dir,
                case_id=f"{sid}-fork",
            )
            fork_ok = fork_outcome.correct
            if fork_ok:
                intervene_correct += 1
            if not ctrl_ok and fork_ok:
                flips_helped += 1
            if ctrl_ok and not fork_ok:
                flips_harmed += 1

            result = {
                "case_id": sid,
                "fired": True,
                "surface_turn": s.turn_index,
                "reminder": s.reminder_text[:200],
                "control_correct": ctrl_ok,
                "intervene_correct": fork_ok,
                "forked_session": forked.get_session_id(),
            }
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            logging.getLogger(__name__).info(
                "[%d/%d] %s fired=True control=%s intervene=%s",
                total, len(session_ids), sid, ctrl_ok, fork_ok,
            )
    finally:
        fh.close()

    def pct(n: int) -> str:
        return f"{100.0 * n / total:.1f}%" if total else "n/a"

    return (
        f"cases={total} fired={fired}\n"
        f"control  correct: {control_correct} ({pct(control_correct)})\n"
        f"intervene correct: {intervene_correct} ({pct(intervene_correct)})\n"
        f"flips  W->R(helped): {flips_helped}   R->W(harmed): {flips_harmed}"
    )


def _extract_submission(messages: list[Any]) -> str | None:
    """Find submit_final_report tool call and return its text arg."""
    from agentm.core.abi import AssistantMessage, ToolCallBlock

    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name == "submit_final_report":
                return block.arguments.get("text")
    return None


def main() -> None:
    app()


if __name__ == "__main__":
    main()
