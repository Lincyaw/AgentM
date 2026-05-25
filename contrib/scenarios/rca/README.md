# agentm-rca

`agentm-rca` is a consumer of the AgentM SDK, not part of the SDK itself.

This package owns RCA-specific code and dependencies, including the DuckDB-backed
observability tools migrated from the earlier monolithic implementation.
The root `agentm` package stays free of RCA, observability, and DuckDB concerns.

## What lives here

- RCA-only tool implementations under `src/agentm_rca/`
- RCA-only dependencies such as `duckdb`, `pyarrow`, and `tiktoken`
- Scenario-local tests and fixtures under `tests/`

## What does not live here

- Changes to `src/agentm/` for RCA-specific behavior
- DuckDB or observability dependencies added to the SDK
- Tight coupling from the SDK back into RCA

## Opting into llmharness

The default `manifest.yaml` and `manifest.baseline.yaml` do **not** mount
the llmharness cognitive-audit adapter. To run rca with the
extractor + auditor pipeline supervising the main agent, pick one of the
`manifest.harness*.yaml` variants. The variant matrix (mode, cadence,
auditor / reminder on or off, intended use) lives in
[`_VARIANTS.md`](_VARIANTS.md) — read that first.

One concrete invocation against a sandbox:

```bash
uv run agentm \
  --cwd <sandbox> \
  --scenario rca:harness.sync \
  "<your RCA prompt>"
```

Adapter behaviour (cadence, sync vs async, reminders on or off) is set
**inside** the manifest's `llmharness.adapters.agentm` config block; the
caller does not override it. Other variants:

* `rca:harness` — async, production cadence.
* `rca:harness.sync` — sync, default for dataset collection.
* `rca:harness.sync.opinions` / `.opinions10` — sync, auditor verdicts
  recorded but reminders **not** injected (clean trajectory for SFT).
* `rca:harness.sync.extractor5` — extractor-only variant (no auditor
  side channel). Kept around as a cheap pipeline-equivalent baseline.
* `rca:harness.live` — `harness.sync` + `contrib.extensions.live_inspector`
  WebSocket server. Use for live-watching a single case from the
  aegis-ui `Live Inspect` sub-app. **Don't** use this for dataset-
  collection runs (the inspector buffers a per-session backlog).

### Live inspection (single case, watch in browser)

Minimum command — run from the AgentM repo root, no sandbox needed
(`GitBackedResourceWriter` is lazy and never touches disk for RCA since
no write tools are mounted). `.env` autoloads from `<cwd>/.env` and the
workspace root, so provider creds don't need a manual `source`:

```bash
AGENTM_RCA_DATA_DIR=/home/ddq/AoyangSpace/dataset/rca/<case-id> \
AGENTM_LIVE_INSPECT_URL_FILE=/tmp/ws.url \
uv run agentm --scenario rca:harness.live "<your RCA prompt>"
```

The inspector prints `LIVE INSPECT: ws://127.0.0.1:<port>/inspect?root=<id>`
on stderr at startup; with `AGENTM_LIVE_INSPECT_URL_FILE` the same URL is
also written to that sidecar. Paste it into the aegis-ui *Live Inspect*
Connection page to watch the main agent + every spawned extractor /
auditor child in real time.

The inspector binds to `127.0.0.1` by default — the browser must be on
the same machine. For remote viewing, override in the manifest config
(`bind: 0.0.0.0`) and add your own auth at the network layer, or open an
SSH tunnel:

```bash
ssh -L <port>:127.0.0.1:<port> <agentm-host>
```

Detection is by manifest composition, not by name — the eval driver's
`_scenario_mounts_harness()` loads the resolved manifest and checks for
`llmharness.adapters.agentm` in the extension list. Renaming a variant
or adding a new one does **not** require any string-pattern update.

### Chained-fork intervention (`chained_fork`)

> **Pair with an `opinions` variant, not the default `harness.sync`.**
> `chained_fork` is an A/B counterfactual experiment: the control segment
> must be a **clean baseline with no live reminder injection** so the
> branches (which seed the reminder via `llmharness.replay.reminder_seed`)
> are the only intervention. The default `rca:harness.sync` has
> `enable_reminders: true` and therefore injects the auditor's first
> surface into the control trajectory live — silently invalidating the
> counterfactual. Use `rca:harness.sync.opinions` (5-turn) or
> `rca:harness.sync.opinions10` (10-turn) instead; both set
> `enable_reminders: false` so the auditor only writes opinions and
> never mutates the control. See [`_VARIANTS.md`](_VARIANTS.md).

When running the eval driver with `--ak chained_fork=true` (optionally
`--ak max_interventions=N`, defaults to 10), the driver:

1. Runs a **control** segment against the configured `scenario` —
   no seeded reminder, no initial messages. (See the warning above:
   the configured `scenario` must have `enable_reminders: false`, else
   the control is no longer a clean baseline.)
2. Replays the cognitive-audit pipeline offline over the control
   trajectory with `stop_on_first_surface=True` to find the first
   auditor firing that surfaces a reminder.
3. If a reminder surfaced, forks a **branch** segment from the parent
   trajectory at that turn with the reminder text seeded into the
   child session via `llmharness.replay.reminder_seed`. The branch
   replays under the same scenario.
4. Repeats step 2-3 for each subsequent branch, threading the prior
   segment's `CumulativeAuditState` into the next via the offline
   driver's `seed_cumulative` / `start_turn` knobs. Stops on the first
   silent auditor or after `max_interventions` branches.
5. Writes a unified replay sidecar
   (`<final_branch_session_log_id>.chained.jsonl`) via
   `write_chained_replay` so the case viewer sees the full N-segment
   chain.

There is no separate "control scenario" anymore — every segment runs
the same scenario; only the seeded prefix + reminder text differ.

The returned `AgentResult.metadata` carries an `intervention_mode:
"chained_fork"` entry plus a `chained_fork.segments` list describing
each segment (`segment_index`, `session_log_id`, `is_control`,
`fork_turn_index`, `reminder_text`, `surfaced_reminder_turn`, and the
segment's full `_SessionRun` metadata).

For the llmharness side of this story (public API, replay sidecar
schema, profile / prompt knobs, prefix-replay flow), see
`contrib/extensions/llmharness/README.md` and the docs it links to.

## Eval-config environment variables

The YAML configs under `contrib/scenarios/rca/eval/` and
`contrib/scenarios/rca/eval/baseline/` use shell-style ``${VAR:-default}``
placeholders for paths and model identifiers so the same config is portable
across machines. The external `rca llm-eval` runner does **not** expand them
itself — pre-process the file with `envsubst` (or `gettext-base`) before
piping it in:

```bash
RCA_DATASET_ROOT=/abs/path/to/rca \
MODEL_NAME=Doubao-Seed-2.0-pro \
JUDGE_MODEL=Doubao-Seed-2.0-pro \
envsubst < contrib/scenarios/rca/eval/config.yaml \
  | uv run rca llm-eval run - -a agentm -l 1
```

Recognized variables:

| Variable            | Used in                          | Default                |
|---------------------|----------------------------------|------------------------|
| `RCA_DATASET_ROOT`  | `source_path`                    | (required, no default) |
| `MODEL_NAME`        | top-level `model_name`           | `Doubao-Seed-2.0-pro`  |
| `JUDGE_MODEL`       | `judge_model.model_provider.model` | `Doubao-Seed-2.0-pro` |

`RCA_DATASET_ROOT` has no default — set it explicitly or the resulting YAML
will contain an empty string and the runner will reject the config.

> Note: `envsubst` does **not** honour the shell-style `${VAR:-default}`
> form — it leaves the literal placeholder in place when the variable is
> unset. Either export every variable referenced by the config before
> piping (the snippet above), or use the no-default companion configs
> (`config.ops-lite.yaml`).

## Reproducing the harness.sync run on `ops-lite` (anon-ops/ops-lite)

This is the end-to-end recipe used to validate `rca:harness.sync` on the
Hugging Face `anon-ops/ops-lite` datapack with 5-way concurrent rollout
and Doubao as both rollout-model and per-evidence judge.

### Prerequisites

1. **Latest `rcabench-platform` installed editable.** The lockfile
   pins `>=0.4.43` from PyPI; if you also keep the upstream worktree
   locally (e.g. at `~/AoyangSpace/aegis/rcabench-platform`), point the
   ambient Python at it so registered processers / agents stay in sync
   with HEAD:

   ```bash
   pip install -e ~/AoyangSpace/aegis/rcabench-platform
   ```

   Then refresh the AgentM workspace so its `.venv` picks up
   `rcabench-platform[sdk,llm-eval]` (sqlmodel etc. live behind the
   `llm-eval` extra — `uv sync` alone is not enough):

   ```bash
   uv sync --all-packages
   ```

2. **`ops-lite` cases on disk** under
   `datasets/ops-lite/cases/<case_name>/` (parquet + `causal_graph.json`
   + `injection.json`).

3. **`eval.db` ingested** with `dataset='ops-lite'`. The repo ships an
   `eval.db` already populated for both `RCABench` (openrca2-lite slice)
   and `ops-lite`; verify with `sqlite3 eval.db "select dataset, count(*)
   from data group by dataset"`.

4. **`.env`** with `AGENTM_PROVIDER=openai`, `OPENAI_BASE_URL`,
   `WARPGATE_TICKET`, and `OPENAI_VERIFY_SSL=false` — same wiring the
   agent uses. `agentm_rca.eval` monkey-patches the rcabench-platform
   judge client at import time so the judge call also flows through the
   Warpgate ticket header; nothing extra to do for the judge.

### Run

```bash
set -a && source .env && set +a
export RCA_DATASET_ROOT=$PWD/datasets/ops-lite/cases \
       MODEL_NAME=Doubao-Seed-2.0-pro \
       JUDGE_MODEL=Doubao-Seed-2.0-pro

envsubst < contrib/scenarios/rca/eval/config.ops-lite.yaml \
  > /tmp/opslite-eval.yaml

uv run --no-sync rca llm-eval run /tmp/opslite-eval.yaml \
  -a agentm --ak scenario=rca:harness.sync \
  -n 5 -l 50 \
  --exp-id agentm-rca-opslite-harness-sync
```

`-n 5` is the 5-thread concurrent rollout; drop `-l 50` to run the full
500-sample slice (expect many hours under `harness.sync` — the auditor /
extractor children fire synchronously every turn).

### What we patch / extend, and why

| Where                                                                  | Change                                                                                                                    | Reason                                                                                                                                                                            |
|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `manifest.{baseline,harness,harness.sync}.yaml`                        | Prepend `agentm.extensions.builtin.operations_local`                                                                      | Post `harness-collapse` (commit `e062913`) the session factory fail-stops if no atom registered Operations. ThinkDepthAI tools never use it but the registration is mandatory.    |
| `eval/config.ops-lite.yaml`                                            | New config for the `ops-lite` slice with `dataset: ops-lite`, `tags: [ops-lite]`, plain `${VAR}` placeholders (no `:-`)   | The shipped `config.yaml` targets RCABench/`openrca2-lite` and uses `:-` defaults that `envsubst` leaves un-expanded.                                                             |
| `agentm_rca/eval/__init__.py` — `_register_processer_aliases`          | `PROCESSER_FACTORY.register("ops-lite", RCABenchProcesser)` (and `opslite`)                                               | rcabench-platform routes preprocess/judge by `sample.dataset.lower()`. `ops-lite` has no native processer; its on-disk layout matches RCABench, so it can reuse RCABenchProcesser. |
| `agentm_rca/eval/__init__.py` — `_patch_judge_client_for_warpgate`     | Replace `BaseLLMJudgeProcesser.judge_client` with a property that injects `default_query={"warpgate-ticket": ...}` + skips TLS verify | `ModelProviderConfig` has no `extra_query` / `verify_ssl` field. Without this patch the per-evidence judge gets 401/403 against the LiteLLM-behind-Warpgate gateway.              |

All changes live in the AgentM tree — the `rcabench-platform` checkout
is untouched.

## Observability: OTel-native identity

As of the 2026-05-12 unification, AgentM's identity model maps 1:1 onto
the OTel data model so any OTel-compatible store (Jaeger, Grafana
Tempo, OTLP collectors) can ingest `.agentm/observability/*.jsonl`
without translation:

| AgentM field              | OTel field        | Shape       | Notes                                                                                                                                              |
|---------------------------|-------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `api.root_session_id`     | `trace_id`        | 32 hex chars | Generated once per *root* session; every spawned child session inherits it verbatim → one OTel trace per agent tree.                              |
| `api.session_id`          | `span_id`         | 16 hex chars | This session's root span; equal to the JSONL filename (`<session_id>.jsonl`).                                                                     |
| `api.parent_session_id`   | `parent_span_id`  | 16 hex chars | Parent's `session_id` for spawned children; `None` for top-level sessions.                                                                        |
| `AgentSessionConfig.session_id`       | (caller-supplied span_id)   | 16 hex      | If your embedder already maintains an OTel span id, pass it through and AgentM uses it verbatim — no internal renaming.                            |
| `AgentSessionConfig.root_session_id`  | (caller-supplied trace_id)  | 32 hex      | Same idea for the trace_id; combined with `parent_session_id` this turns AgentM into a drop-in child of an upstream OTel span.                    |

`session.start` and `session.end` are the canonical *session-root* span
of each session: their `span_id` equals `api.session_id` and they share
the same `trace_id` across the whole agent tree. Recovering the full
parent ↔ extractor ↔ auditor structure of a rollout is a single filter:

```bash
trace_id=$(sqlite3 eval.db \
  "select trace_id from evaluation_data where source = '<case>' \
   and exp_id like 'agentm-rca-opslite-fixed-50-%' \
   order by created_at desc limit 1")

grep -l "\"trace_id\": \"$trace_id\"" .agentm/observability/*.jsonl
```

A helper that materialises the mapping for an entire eval run lives at
`contrib/scenarios/rca/eval/scripts/build_trace_map.py`:

```bash
uv run python contrib/scenarios/rca/eval/scripts/build_trace_map.py \
    --exp-id agentm-rca-opslite-fixed-50-baseline \
    --out eval-data/agentm-rca-opslite-fixed-50-baseline/trace_map.json
```

The output groups every JSONL file by `trace_id`, separates parent and
spawned children by `purpose`, and surfaces per-sample fan-out
(extractor / auditor session counts).

## Frozen 50-sample regression set (`ops-lite-fixed-50`)

After every strategy change (atom rewrite, prompt tweak, harness mode
swap) the agent should be re-measured against the same 50 ops-lite
cases that produced the 2026-05-11 baseline. The case set lives in:

```
contrib/scenarios/rca/eval/fixtures/ops-lite-fixed-50.txt
```

The DB-side contract is a tag, not a list: the helper script
`scripts/retag_fixed_set.py` keeps the `data.tags` column in sync with
the fixture file (adds the `ops-lite-fixed-50` tag to listed rows,
strips it from any non-listed row). `config.ops-lite-fixed-50.yaml`
selects samples via `tags: ["ops-lite-fixed-50"]` — there is no need to
extend rcabench-platform with a list-of-sources filter.

### Re-run

```bash
bash contrib/scenarios/rca/eval/scripts/rerun_opslite_fixed_50.sh \
    <exp_id_suffix> [<scenario>]
```

Examples:

```bash
# Baseline reproduction
bash contrib/scenarios/rca/eval/scripts/rerun_opslite_fixed_50.sh baseline

# A/B a new system prompt
bash contrib/scenarios/rca/eval/scripts/rerun_opslite_fixed_50.sh after-prompt-v2

# Compare async vs sync harness on the same 50 cases
bash contrib/scenarios/rca/eval/scripts/rerun_opslite_fixed_50.sh async-2026-06 rca:harness
```

Each invocation writes a distinct `exp_id`
(`agentm-rca-opslite-fixed-50-<suffix>`) into `eval.db`, so before/after
diffs are a plain SQL query:

```sql
select exp_id,
       avg(case when correct then 1.0 else 0.0 end) as accuracy,
       avg(json_extract(eval_metrics, '$.evidence_support_rate'))
         as evidence_support_rate,
       count(*) as n
  from evaluation_data
 where exp_id like 'agentm-rca-opslite-fixed-50-%'
   and stage = 'judged'
 group by exp_id;
```

### Updating the case set

The fixture is intentionally append-only — replacing it invalidates
historical comparisons. If a new regression set is needed, copy the
file to a new name (e.g. `ops-lite-fixed-100.txt`), make a matching
`config.ops-lite-fixed-100.yaml` with a fresh tag, and the existing
`baseline` numbers stay reproducible against the original 50.
