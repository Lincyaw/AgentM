# injection — adversarial fault-injection workflow

This scenario ports the `aegis-v2` injection skill into AgentM as a
workflow-first campaign. It is the RCA puzzle setter: each round chooses fault
injections that maximize RCA inference difficulty, submits them through
`aegisctl`, verifies that faults landed, and records verifier-compatible case
records for future planning.

## Layout

- `workflow.py` — campaign orchestrator. It owns round scheduling, Aegis state
  preflight, family tally retrieval, verifier-compatible case-context loading,
  child-agent fan-out, heartbeat updates, and appending `case_records` to the
  case library.
- `manifest.yaml` — single-round injection worker scenario composition.
- `injection_context.py` — local atom that injects workflow `atom_config` into
  the worker prompt.
- `prompts/author.md` — worker policy for choosing/submitting/verifying one
  adversarial injection round.
- `knowledge/schema.md` — minimal verifier-compatible case record contract.
- `loop.sh` — compatibility wrapper around `agentm workflow run`; prefer the
  workflow command directly for new automation.

Persistent campaign state follows the original convention plus a case library:

```text
~/.aegisctl/injection-author/<system>/
  metadata.json
  memory.md
  aegisctl_gaps.md
  case_library.jsonl
  rounds/round-<N>-<ts>.json
  loop.log
```

## Workflow Run

```bash
uv run agentm workflow run contrib/scenarios/injection/workflow.py \
  --cwd ~/.aegisctl/injection-author/ts \
  --args '{
    "system": "ts",
    "rounds": 1,
    "state_dir": "~/.aegisctl/injection-author/ts",
    "aegisctl_bin": "/home/nn/.cache/aegisctl-agentm-test",
    "project": "pair_diagnosis",
    "source_dir": null,
    "validation_mode": true,
    "max_parallel": 1
  }'
```

Compatibility wrapper:

```bash
contrib/scenarios/injection/loop.sh ts \
  --rounds 1 \
  --aegisctl-bin /home/nn/.cache/aegisctl-agentm-test \
  --validation-mode \
  --extra-instruction 'Validation only: K_outer=1, K_inner=1.'
```

## Submit Contract

The workflow preflights the current submit contract and passes it to the worker:

- `--pedestal-name` is the target system short code, e.g. `ts`, `sn`, `media`,
  or `sockshop`.
- `--pedestal-tag` must be resolved immediately before submit with
  `aegisctl container versions <system> -o json`; use the newest returned tag.
  Never reuse tags from `memory.md` or old rounds.
- Benchmark is currently fixed as `--benchmark-name clickhouse --benchmark-tag
  1.0.0` unless the caller explicitly changes the campaign policy later.
- Record the resolved contract in each round file under `submit.resolved_contract`.

`--skip-restart-pedestal` is not part of the normal campaign path. Use it only
when the user explicitly asks for a warm-pool validation shortcut, and record
that choice in the round file.

## Diversity

Before each worker round, `workflow.py` pre-computes `family_tally` from the
latest Aegis injection list and injects it via `atom_config`. The worker treats
the Aegis skill's family caps as hard limits in the recent 50-injection window:
network <=25%, pod <=20%, jvm <=25%, http <=20%, dns <=10%, stress <=10%.

## Verifier-Compatible Knowledge

Propagation paths should come from verifier outputs, not from an injection-only
truth store. The injection workflow consumes `case_library.jsonl`, whose records
match the minimal contract in `knowledge/schema.md`:

- `fault`: family, chaos type, target service, and scope.
- `observations`: detector/SLO/injection-landed signals.
- `verifier_evidence`: propagation path, root cause, decoys, confidence, and a
  status of `verified`, `simulated`, `pending`, or `failed`.

If verifier artifacts are not available for a fresh trace, the worker may write
`verifier_evidence.status = "simulated"` as a temporary weak prior or
`"pending"` while the trace is still running. Later verifier imports should
supersede or calibrate those records.

## Long-Running Campaign

`workflow.py` accepts `rounds` and `sleep` in `--args`. The wrapper keeps the old
shape for process managers:

```bash
nohup contrib/scenarios/injection/loop.sh ts \
  --rounds 999 \
  --sleep 900 \
  --aegisctl-bin /tmp/aegisctl \
  --source-dir /path/to/target/system/source \
  > /tmp/agentm-injection-ts.log 2>&1 &
```
