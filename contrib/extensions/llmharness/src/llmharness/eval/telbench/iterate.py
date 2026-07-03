"""TEL prompt evolution loop: eval → reflect → evolve → eval → keep/discard.

Each iteration snapshots prompt files, runs eval+reflect, evolves prompts,
re-evaluates, and keeps or discards based on macro_f1 improvement.  All
artifacts are stored per-iteration; the main repo git history is untouched.

Usage::

    llmharness-iterate \\
        --data datasets/data/TELBench.jsonl \\
        --instance-ids ./failed_cases_first5.json \\
        --model azure-gpt \\
        --max-iter 5 \\
        --runs-dir ./evolution_runs
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

try:
    from agentm.env import autoload_dotenv
    _HAS_AGENTM = True
except ImportError:
    autoload_dotenv = None  # type: ignore[assignment]
    _HAS_AGENTM = False
    logger.debug("telbench.iterate: agentm SDK not available; dotenv autoload disabled")

from .scoring import AggregateScores

_PROMPTS_DIR = Path(__file__).parents[2] / "agents" / "tel" / "prompts"
_PROMPT_FILES = ("notepad.md", "reason.md")

app = typer.Typer(help="Iterative TEL prompt evolution with auto keep/discard.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prompt_sha() -> str:
    """Content hash of the current prompt files (first 8 hex chars)."""
    h = hashlib.sha256()
    for name in _PROMPT_FILES:
        p = _PROMPTS_DIR / name
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:8]


def _snapshot_prompts(dest: Path) -> None:
    """Copy current prompt files into *dest*/prompts/."""
    out = dest / "prompts"
    out.mkdir(parents=True, exist_ok=True)
    for name in _PROMPT_FILES:
        src = _PROMPTS_DIR / name
        if src.is_file():
            shutil.copy2(src, out / name)


def _restore_prompts(src_dir: Path) -> None:
    """Restore prompt files from a snapshot directory."""
    snap = src_dir / "prompts"
    for name in _PROMPT_FILES:
        src = snap / name
        if src.is_file():
            shutil.copy2(src, _PROMPTS_DIR / name)


@dataclass(slots=True)
class IterationRecord:
    iteration: int
    timestamp: str
    prompt_sha: str
    macro_f1: float
    macro_p: float
    macro_r: float
    fea: float
    decision: str
    summary: str

    def to_tsv_row(self) -> str:
        safe_summary = self.summary.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
        return "\t".join([
            str(self.iteration),
            self.timestamp,
            self.prompt_sha,
            f"{self.macro_f1:.4f}",
            f"{self.macro_p:.4f}",
            f"{self.macro_r:.4f}",
            f"{self.fea:.4f}",
            self.decision,
            safe_summary,
        ])


_TSV_HEADER = "iter\ttimestamp\tprompt_sha\tmacro_f1\tmacro_p\tmacro_r\tfea\tdecision\tsummary"


def _append_record(tsv_path: Path, record: IterationRecord) -> None:
    """Append one record to the evolution TSV (create with header if new)."""
    exists = tsv_path.is_file()
    with open(tsv_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(_TSV_HEADER + "\n")
        f.write(record.to_tsv_row() + "\n")


def _parse_eval_output(jsonl_path: Path) -> AggregateScores:
    """Parse per-instance JSONL and compute aggregate scores."""
    from .scoring import SpanScores, aggregate_scores

    scores: list[SpanScores] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            scores.append(SpanScores(
                precision=d["precision"],
                recall=d["recall"],
                f1=d["f1"],
                first_error_accurate=d["first_error_accurate"],
            ))
    return aggregate_scores(scores)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def iterate(
    data: Annotated[Path, typer.Option("--data", help="Path to TELBench JSONL")],
    runs_dir: Annotated[
        Path, typer.Option("--runs-dir", help="Root directory for iteration artifacts")
    ] = Path("./evolution_runs"),
    model: Annotated[str | None, typer.Option("--model", help="config.toml profile name")] = None,
    instance_ids: Annotated[
        Path | None,
        typer.Option("--instance-ids", help="JSON file with instance IDs to evaluate"),
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", "-j")] = 5,
    max_iter: Annotated[int, typer.Option("--max-iter", help="Max evolution iterations")] = 5,
    max_discard: Annotated[
        int, typer.Option("--max-discard", help="Stop after N consecutive discards")
    ] = 2,
) -> None:
    """Run the eval→reflect→evolve loop with auto keep/discard."""
    import os
    import subprocess

    if _HAS_AGENTM:
        autoload_dotenv()

    runs_dir = runs_dir.resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = runs_dir / "evolution.tsv"

    # --- Build common eval args ---
    eval_base: list[str] = [
        "uv", "run", "llmharness-eval",
        "--data", str(data.resolve()),
        "--mode", "tel",
        "--auditor-prompt", "2pass",
        "--concurrency", str(concurrency),
    ]
    if model:
        eval_base.extend(["--model", model])
    if instance_ids:
        eval_base.extend(["--instance-ids", str(instance_ids.resolve())])

    def run_eval(iter_dir: Path, *, with_reflect: bool = False) -> AggregateScores:
        """Run eval in *iter_dir*, optionally with --reflect."""
        iter_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = iter_dir / "out.jsonl"
        cmd = [
            *eval_base,
            "--cwd", str(iter_dir),
            "--output", str(out_jsonl),
        ]
        if with_reflect:
            cmd.append("--reflect")
        typer.echo(f"  $ {' '.join(cmd[-6:])}")
        subprocess.run(cmd, check=True, env=os.environ)
        return _parse_eval_output(out_jsonl)

    def run_evolve(prev_dir: Path, cur_dir: Path) -> str:
        """Run evolve on the reflections in *prev_dir*, write summary to *cur_dir*."""
        reflect_dir = prev_dir / "reflections"
        if not reflect_dir.is_dir() or not list(reflect_dir.glob("*.md")):
            return "no reflections"
        cur_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cur_dir / "evolve_summary.md"
        cmd = [
            "uv", "run", "llmharness-evolve",
            "--reflections", str(reflect_dir),
            "--summary-file", str(summary_path),
        ]
        if model:
            cmd.extend(["--model", model])
        subprocess.run(cmd, check=True, env=os.environ)
        if summary_path.is_file():
            full = summary_path.read_text(encoding="utf-8").strip()
            first_line = full.split("\n", 1)[0].strip()
            return first_line[:200] if first_line else full[:200]
        return "(no summary)"

    # --- Baseline (iter 0) ---
    typer.echo("=== Iteration 0 (baseline) ===")
    iter0_dir = runs_dir / "iter-0"
    _snapshot_prompts(iter0_dir)
    baseline = run_eval(iter0_dir, with_reflect=True)

    _append_record(tsv_path, IterationRecord(
        iteration=0,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        prompt_sha=_prompt_sha(),
        macro_f1=baseline.macro_f1,
        macro_p=baseline.macro_precision,
        macro_r=baseline.macro_recall,
        fea=baseline.first_error_accuracy,
        decision="baseline",
        summary="-",
    ))
    typer.echo(f"  baseline: F1={baseline.macro_f1:.4f}")

    best_f1 = baseline.macro_f1
    best_iter = 0
    consecutive_discards = 0

    # --- Evolution loop ---
    for i in range(1, max_iter + 1):
        typer.echo(f"\n=== Iteration {i} ===")

        # Evolve prompts based on previous reflections
        prev_dir = runs_dir / f"iter-{i - 1}"
        iter_dir = runs_dir / f"iter-{i}"
        typer.echo("  evolving prompts…")
        evolve_summary = run_evolve(prev_dir, iter_dir)

        _snapshot_prompts(iter_dir)

        # Re-evaluate with evolved prompts
        typer.echo("  evaluating…")
        scores = run_eval(iter_dir, with_reflect=True)
        new_f1 = scores.macro_f1

        # Keep or discard
        if new_f1 > best_f1:
            decision = "keep"
            best_f1 = new_f1
            best_iter = i
            consecutive_discards = 0
            typer.echo(f"  F1={new_f1:.4f} > {best_f1 - (new_f1 - best_f1):.4f} → KEEP")
        else:
            decision = "discard"
            consecutive_discards += 1
            typer.echo(f"  F1={new_f1:.4f} <= {best_f1:.4f} → DISCARD (restoring iter-{best_iter})")
            _restore_prompts(runs_dir / f"iter-{best_iter}")

        _append_record(tsv_path, IterationRecord(
            iteration=i,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            prompt_sha=_prompt_sha(),
            macro_f1=new_f1,
            macro_p=scores.macro_precision,
            macro_r=scores.macro_recall,
            fea=scores.first_error_accuracy,
            decision=decision,
            summary=evolve_summary,
        ))

        if consecutive_discards >= max_discard:
            typer.echo(f"\n{max_discard} consecutive discards — converged, stopping.")
            break

        # All cases correct
        reflect_dir = iter_dir / "reflections"
        if not reflect_dir.is_dir() or not list(reflect_dir.glob("*.md")):
            typer.echo("\nAll cases correct — stopping.")
            break

    # --- Summary ---
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Evolution complete. Best F1={best_f1:.4f} at iter-{best_iter}")
    typer.echo(f"Tracking: {tsv_path}")
    typer.echo(f"Artifacts: {runs_dir}")
    typer.echo(f"\nCurrent prompts reflect iter-{best_iter}.")
    typer.echo(f"To inspect:  diff {runs_dir}/iter-0/prompts/notepad.md {runs_dir}/iter-{best_iter}/prompts/notepad.md")
    typer.echo(f"To commit:   git add {_PROMPTS_DIR} && git commit -m 'evolve: iter {best_iter}'")


def main() -> None:
    app()


__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
