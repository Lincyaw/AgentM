# pattern_miner

Offline failure-pattern miner: the offline half of the policy feedback loop.
It consumes `(task, trajectory, eval result)` triples from Harbor batches and
extracts failure patterns along the feedback dimensions defined in
`docs/policy-feedback-dimensions.md`. One miner session reviews one case
along one dimension; each dimension is a separate angle prompt under
`angles/`. A final attribution session names the root dimension and its
shadows per case.

## Shape

- `agents/` — one YAML per agent role (extension list + system prompt):
  `miner.yaml` reviews one case along one dimension, `distiller.yaml`
  consolidates a batch's findings into checklist items. The root
  `manifest.yaml` / `manifest.distill.yaml` are thin shells the scenario
  loader requires; they just include the agent definitions.
- `prompts/` — prompts bound to their agent: `prompts/miner/` holds the
  eight angle prompts plus `attribution.md`; `prompts/distiller/` holds the
  distillation contract. `prompts/__init__.py` is the host-side builder.
- `extensions/` — shared scenario-local atoms: `case_tools.py`, read-only
  tools over one case bundle (the session `cwd` is the bundle; the miner
  reads nothing else).
- `runner.py`, `export.py`, `schema.py` — host-side orchestration,
  trial-to-bundle exporter, and pydantic output schemas. Structured-output
  schemas are attached per session by the runner (`extra_extensions`)
  because angle reports, attribution, and checklist drafts differ.

## Run

```bash
export OPENAI_API_KEY=...           # key for the miner model endpoint
uv run contrib/scenarios/pattern_miner/runner.py mine \
  jobs/<batch-timestamp> \
  --out .agentm/mining/<batch-name> \
  --dsn postgresql://agentm:agentm@localhost:55432/agentm_test \
  --schema ssb_r1 \
  --model azure-chat --base-url http://100.99.217.100:4000/v1 \
  -n 2
```

After the batch, distill the angle reports into the merged checklist — the
reusable product. One distiller session per dimension consumes all of that
dimension's findings (plus the existing checklist) and submits merged,
generalized items:

```bash
uv run contrib/scenarios/pattern_miner/runner.py distill \
  --out .agentm/mining/<batch-name> \
  --model azure-chat --base-url http://100.99.217.100:4000/v1 \
  --dsn postgresql://agentm:agentm@localhost:55432/agentm_test
```

The checklist (`<out>/checklist.yaml` by default; point `--checklist` at the
durable library to accumulate across batches) is what future critique
consumes: each item carries `check` (a chain-language question a critic can
execute on any case), `when` (a trajectory-computable trigger), `how` (the
cheapest sufficient sensor), and `evidence` (supporting trials). Case
stories stay in `patterns.jsonl`; only mechanisms generalize into items.

Useful flags: `--angle` (repeatable) to run a subset of dimensions,
`--trial <substring>` and `--limit N` for smoke runs, `--include-passes`
to mine passing trials for contrast, `--refresh-bundles` after a store fix.

Angles are mutually blind, so within a case they run in parallel
(`--angle-concurrency`, default 4). With `--with-env` the runner starts one
live ARL environment per case from the task image (`--gateway-url` /
`$ARL_GATEWAY_URL`, `--image-registry`, `--image-tag`) and hands the miner an
`env_bash` tool — repository in pre-fix state, agent and oracle patches at
`/tmp` — so it can apply patches, build, run tests, and probe the scenario as
an active critic. A live environment is stateful, so `--with-env` forces the
case's angles to run serially; the environment is stopped and deleted when
the case finishes.

## Outputs

- `<out>/patterns.jsonl` — one row per finding: task, trial, agent model,
  reward, dimension, root dimension, pattern, instance, turn evidence,
  online signature, gt_dependence, confidence.
- `<out>/cases/<trial>/report.json` — full per-case record (all angle
  reports plus attribution); `bundle/` holds the exported case.
- `<out>/skipped.json` — every case not mined and why (passes without
  `--include-passes`, invalid labels, missing trajectories). No silent drops.

## Discipline

- Labels pass the mechanical G0 gate before mining: empty/unparseable
  `reward.txt` or a harness exception marks the label invalid, and only the
  `label_validity` angle runs on such cases.
- Findings carry `gt_dependence` and `online_signature` so the DSL step can
  tell what transfers online; ground truth itself never does.
- Keep the mining set and the deployment-evaluation set disjoint at task
  level. Trials already analyzed against ground truth belong to the mining
  side permanently.
- The miner journals its own sessions to the same Postgres under
  `--miner-schema` (default `pattern_miner`), never into the batch schema it
  reads — schema-enumerating consumers (policy backfill, later mining
  rounds) treat a batch schema as pure case data. View miner sessions with
  `AGENTM_TRAJECTORY_DSN=<dsn> AGENTM_TRAJECTORY_SCHEMA=pattern_miner
  uv run agentm trace sessions`.
