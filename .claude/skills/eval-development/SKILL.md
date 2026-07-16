---
name: eval-development
description: >
  Guide for adding new benchmarks to the agentm-eval framework and running
  experiments through the unified `agentm eval` CLI. Covers the adapter
  interface, experiment lifecycle, result schema, ClickHouse linking, and
  output conventions. Use whenever creating a new eval adapter, modifying
  an existing benchmark integration, writing code under
  contrib/evals/src/agentm_eval/, discussing experiment management, or
  when the user mentions "add a benchmark", "new eval", "接入评估",
  "新benchmark", "实验管理", "eval adapter". Also trigger when reviewing
  eval-related code or debugging experiment output structure.
---

# Eval Development Guide

The `agentm-eval` package (`contrib/evals/`) provides a unified experiment
framework. Every benchmark plugs in as an **adapter** — a class that
registers a typer sub-app and uses the `Experiment` lifecycle to get free
exp_id generation, structured output, ClickHouse linking, and a common
result format.

## Architecture at a glance

```
agentm eval <benchmark> <command> [options]
     │
     ├── Framework (experiment lifecycle, CLI shell, result schema)
     │     contrib/evals/src/agentm_eval/
     │       experiment.py   — Experiment class, exp_id, output dir
     │       result.py       — TaskResult (common envelope)
     │       registry.py     — adapter discovery
     │       cli.py          — unified CLI + cross-benchmark commands
     │
     └── Adapters (one per benchmark)
           contrib/evals/src/agentm_eval/adapters/
             sandbox/        — TB1, Harbor, SWE-bench (Docker sandbox)
             aftraj_auditor  — AFTraj-2K safety auditor
             aftraj_grounding — AFTraj-2K grounding pipeline
             tau2            — tau2-bench conversational agent
             telbench        — TELBench span-level error localization
             rescue_window_adapter — fork-at-prefix branching
```

## Adding a new benchmark — step by step

### 1. Create the adapter file

Create `contrib/evals/src/agentm_eval/adapters/<name>.py`. The file must:

1. Define an adapter class with `name` and `description` attributes
2. Implement `create_cli()` returning a `typer.Typer`
3. Call `register()` at module level

Minimal template:

```python
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer

from agentm_eval.experiment import experiment_context
from agentm_eval.registry import register
from agentm_eval.result import TaskResult


class MyBenchAdapter:
    name = "my-bench"
    description = "One-line description of what this benchmark evaluates"

    def create_cli(self) -> typer.Typer:
        cli = typer.Typer(
            name="my-bench",
            help="Longer help text for the sub-CLI.",
            add_completion=False,
        )

        @cli.command()
        def run(
            data: Annotated[Path, typer.Option(help="Dataset path")],
            model: Annotated[Optional[str], typer.Option(help="Model profile")] = None,
            limit: Annotated[Optional[int], typer.Option("-n")] = None,
            exp_id: Annotated[Optional[str], typer.Option(help="Override experiment ID")] = None,
        ) -> None:
            """Run the benchmark."""
            with experiment_context(
                "my-bench", model=model, exp_id=exp_id,
            ) as exp:
                # --- your eval logic here ---
                for task_id, score_dict, session_id in your_results:
                    exp.record_result(TaskResult(
                        task_id=task_id,
                        status="pass" if score_dict["reward"] >= 1.0 else "fail",
                        score=score_dict,
                        session_ids=[session_id] if session_id else [],
                        latency_ms=elapsed,
                    ).to_dict())

                exp.finish(summary={"pass_rate": 0.85, "n_tasks": 100})

        return cli


register("my-bench", MyBenchAdapter.description, MyBenchAdapter)
```

### 2. Register for discovery

Add your module path to `registry.py`'s `discover()` function:

```python
modules = [
    ...
    "agentm_eval.adapters.my_bench",
]
```

### 3. Add optional dependencies (if any)

If your benchmark needs extra packages, add an optional group in
`contrib/evals/pyproject.toml`:

```toml
[project.optional-dependencies]
my-bench = ["some-package>=1.0"]
```

Then run `uv sync --extra eval` to pick it up.

### 4. Verify

```bash
uv run agentm eval list          # should show my-bench
uv run agentm eval my-bench --help  # should show your commands
```

## Key APIs

### Experiment lifecycle

`experiment_context()` is a context manager that handles the full lifecycle:

```python
from agentm_eval.experiment import experiment_context

with experiment_context(
    "benchmark-name",           # used in exp_id and meta.json
    model="model-profile",      # optional, appears in exp_id
    exp_id="custom-id",         # optional override
    output_root=Path("..."),    # optional, default ~/.agentm/eval_runs/
    **params,                   # stored in meta.json for reproducibility
) as exp:
    # exp.exp_id          — auto-generated ID
    # exp.output_dir      — ~/.agentm/eval_runs/{exp_id}/
    # exp.artifacts_dir   — ~/.agentm/eval_runs/{exp_id}/artifacts/
    # exp.record_result() — append to results.jsonl
    # exp.session_config_overrides(task_id) — dict for AgentSessionConfig

    exp.finish(summary={...})   # writes meta.json + report.txt
# on exception: auto-sets status="failed"
```

### TaskResult

The common result envelope. Every adapter records results in this format:

```python
TaskResult(
    task_id="unique-task-name",
    status="pass",              # pass | fail | error | skipped
    score={"reward": 1.0},      # benchmark-specific metrics
    session_ids=["sid-abc123"], # ClickHouse session IDs
    latency_ms=5000,
    error=None,                 # error message if status="error"
    metadata={"attempt": 0},    # benchmark-specific extras
)
```

Call `.to_dict()` before passing to `exp.record_result()`.

### ClickHouse linking

When your benchmark spawns AgentM sessions, pass the experiment overrides
into `AgentSessionConfig` so traces are queryable by exp_id:

```python
from agentm.core.abi.session_config import AgentSessionConfig

overrides = exp.session_config_overrides(task_id)
config = AgentSessionConfig(
    ...
    eval_run_id=overrides["eval_run_id"],     # = exp.exp_id
    eval_task_id=overrides["eval_task_id"],    # = task_id
    task_class=overrides["task_class"],        # = benchmark name
)
```

After the run, `agentm eval export <exp_id>` pulls all matching sessions
from ClickHouse.

## Output structure

Every experiment produces:

```
~/.agentm/eval_runs/{exp_id}/
  meta.json         # benchmark, model, params, status, timestamps, summary
  results.jsonl     # one TaskResult per line
  report.txt        # summary (written by exp.finish)
  artifacts/        # benchmark-specific files (scores, logs, etc.)
```

## Adapter patterns

Different benchmarks have fundamentally different execution models. The
adapter pattern accommodates this — each adapter controls its own
execution strategy. Common patterns:

### Sandbox-based (agent runs in Docker container)

See `adapters/sandbox/` — spawns ARL sessions, runs agent, evaluates
in-sandbox. The adapter manages image resolution, session lifecycle,
and signal handling. This is the most complex pattern.

### Offline LLM evaluation (no sandbox)

See `adapters/aftraj_auditor.py` — runs an LLM pipeline on pre-existing
data. No container, no session creation. The adapter loads data, runs
async evaluation with a semaphore for concurrency, and records results.

### External tool wrapper

See `adapters/telbench.py` — delegates to an external CLI tool
(llmharness-eval), captures its output, and translates results into
TaskResult format. Thinnest adapter pattern.

### External library integration

See `adapters/tau2.py` — imports and calls a third-party library's
runner directly. Manages model profile resolution and result translation.

## Best practices

1. **Always use `experiment_context()`** — it handles meta.json, error
   states, and cleanup. Never manage exp_id or output dirs manually.

2. **Record results incrementally** — call `exp.record_result()` as each
   task completes, not in a batch at the end. This enables resume and
   live progress monitoring.

3. **Pass `--exp-id` option** — every adapter's `run` command should accept
   `--exp-id` so users can override the auto-generated ID (useful for
   reruns or naming conventions).

4. **Set ClickHouse overrides** — if your benchmark spawns AgentM sessions,
   always pass `exp.session_config_overrides()` to link traces.

5. **Store benchmark-specific artifacts under `exp.artifacts_dir`** — scores,
   per-task logs, intermediate files. Keep `exp.output_dir` clean (only
   meta.json, results.jsonl, report.txt).

6. **Map to `status` correctly** — `pass` means the task objective was met;
   `fail` means the agent ran but didn't succeed; `error` means
   infrastructure failure (session crash, timeout, eval error).

7. **Put benchmark-specific scores in `score`** — use the `score` dict for
   metrics that make sense for aggregation (reward, f1, precision). Put
   everything else in `metadata`.

8. **Lazy-import heavy dependencies** — benchmarks often depend on large
   packages (pandas, arl, trajectory_index). Import them inside the
   command function, not at module level, so `agentm eval list` stays fast.

9. **Support resume** — if practical, check for existing results before
   re-running a task. The sandbox adapter does this via `.score.json` files.

10. **Provide `report` and `inspect` commands** — beyond `run`, add commands
    that let users analyze results without re-running. These can read from
    `exp.load_results()` or from the experiment's artifacts.

## Cross-benchmark commands

The framework provides these for free (no adapter code needed):

```bash
agentm eval list                    # list benchmarks
agentm eval runs [--benchmark X]    # list experiments
agentm eval report <exp_id>         # view results
agentm eval export <exp_id>         # export ClickHouse trajectories
```

## Package structure

`contrib/evals/pyproject.toml` defines the `agentm-eval` workspace member.
It's registered in the root `pyproject.toml` workspace and wired into
`agentm eval` via a lazy import in `src/agentm/cli/main.py`.

For the full adapter reference, read the existing adapters — they're
the authoritative examples:

- Simplest: `telbench.py` (~70 lines, external CLI wrapper)
- Medium: `aftraj_auditor.py` (~500 lines, offline LLM evaluation)
- Complex: `sandbox/__init__.py` + `runner.py` (~800 lines, Docker sandbox)
