# Harness scenario variants

The `manifest.harness*.yaml` files share the same baseline atom stack
(`operations_local`, `thinkdepth_sql`, `finalize`, `observability`,
`rcabench_contract`, `prompt_loader`, `runtime_context`) plus
`llmharness.adapters.agentm`. They differ only in the adapter's config
dict and (for a couple of them) which atoms are mounted on top.

The AgentM scenario loader has **no `extends:` / inheritance support**
(see `src/agentm/extensions/loader.py`), so each manifest is repeated in
full. Document drift first; collapse later if a loader feature lands.

## Matrix

| Manifest                                 | mode  | extractor_interval | audit_interval | enable_auditor | enable_reminders | Used for                                                                              |
|------------------------------------------|-------|--------------------|----------------|----------------|------------------|---------------------------------------------------------------------------------------|
| `manifest.harness.yaml`                  | async | 5                  | 5              | (default true) | (default true)   | Production async harness — main agent runs at provider-floor latency.                 |
| `manifest.harness.sync.yaml`             | sync  | 5                  | 5              | (default true) | (default true)   | Dataset collection — main agent stops every k turns waiting for extractor + auditor.  |
| `manifest.harness.sync.opinions.yaml`    | sync  | 5                  | 5              | (default true) | false            | Opinions-only: auditor verdicts are persisted but reminders are NOT injected.          |
| `manifest.harness.sync.opinions10.yaml`  | sync  | 10                 | 10             | true           | false            | Opinions-only at 10-turn cadence; chained-fork intervention experiments.               |
| `manifest.harness.sync.extractor5.yaml`  | sync  | 5                  | 5              | false          | false            | Extractor-only variant (no auditor side channel).                                      |
| `manifest.harness.live.yaml`             | sync  | 5                  | 5              | (default true) | (default true)   | `harness.sync` + `live_inspector` WebSocket — single-case live viewing in aegis-ui.   |
| `manifest.baseline.briefed.yaml`         | —     | —                  | —              | —              | —                | `baseline` with system-prompt pre-briefing on auditor reminders; for replay-fork A/B. |

When adding a new variant, update this table. Chained-fork mode (the
current intervention pipeline — see `README.md`'s
"Chained-fork intervention" section) runs every segment under the same
scenario, so no per-variant control mapping is needed anymore. **But the
scenario you pick must have `enable_reminders: false`** — otherwise the
control segment gets the first auditor surface injected live and the A/B
counterfactual is destroyed before the first branch ever runs. Pair
`chained_fork=true` with `harness.sync.opinions` or
`harness.sync.opinions10`, never with the default `harness.sync`.

## Chained-fork mode (2026-05-24)

Strict-A/B fork mode (the old `baseline_fork` intervention with a
separate `control_scenario` / `branch_scenario` and a `.strict_ab.jsonl`
sidecar) was replaced by chained-fork mode in the harness-runner
refactor's P5. The chained driver
(`llmharness.run_chained_fork_experiment`) repeatedly seeds the agent
with each surfaced reminder under the same scenario and writes a single
`<final_branch_sid>.chained.jsonl` sidecar covering every segment. See
`README.md` for the operator-facing invocation
(`--ak chained_fork=true --ak max_interventions=N`).
