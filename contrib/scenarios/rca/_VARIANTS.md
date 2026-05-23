# Harness scenario variants

The `manifest.harness*.yaml` files share the same baseline atom stack
(`operations_local`, `thinkdepth_sql`, `finalize`, `observability`,
`otel_tracing`, `rcabench_contract`, `prompt_loader`, `runtime_context`)
plus `llmharness.adapters.agentm`. They differ only in the adapter's
config dict and (for a couple of them) which atoms are mounted on top.

The AgentM scenario loader has **no `extends:` / inheritance support**
(see `src/agentm/extensions/loader.py`), so each manifest is repeated in
full. Document drift first; collapse later if a loader feature lands.

## Matrix

| Manifest                                 | mode  | extractor_interval | audit_interval | enable_auditor | enable_reminders | Used for                                                                              |
|------------------------------------------|-------|--------------------|----------------|----------------|------------------|---------------------------------------------------------------------------------------|
| `manifest.harness.yaml`                  | async | 5                  | 5              | (default true) | (default true)   | Production async harness — main agent runs at provider-floor latency.                 |
| `manifest.harness.sync.yaml`             | sync  | 5                  | 5              | (default true) | (default true)   | Dataset collection — main agent stops every k turns waiting for extractor + auditor.  |
| `manifest.harness.sync.opinions.yaml`    | sync  | 5                  | 5              | (default true) | false            | Opinions-only: auditor verdicts are persisted but reminders are NOT injected.          |
| `manifest.harness.sync.opinions10.yaml`  | sync  | 10                 | 10             | true           | false            | Opinions-only at 10-turn cadence; baseline-fork intervention experiments.              |
| `manifest.harness.sync.extractor5.yaml`  | sync  | 5                  | 5              | false          | false            | Extractor-only control for strict A/B fork mode (no auditor side channel).             |
| `manifest.harness.live.yaml`             | sync  | 5                  | 5              | (default true) | (default true)   | `harness.sync` + `live_inspector` WebSocket — single-case live viewing in aegis-ui.   |

When adding a new variant, update this table **and** add the
``rca:<new>`` → control mapping in
``_HARNESS_VARIANT_TO_CONTROL`` at the top of
``src/agentm_rca/eval/agent.py`` if it should be eligible for strict
A/B fork mode.

## `rca:baseline` as a strict-A/B control (2026-05-23)

With the HarnessRunner refactor (`.claude/designs/harness-runner.md` P4),
``baseline_fork`` now also accepts ``rca:baseline`` as control —
``_HARNESS_VARIANT_TO_CONTROL`` maps ``rca:baseline`` → ``rca:baseline``.
The control session mounts no llmharness adapter and therefore produces
no audit-replay sidecar on its own; instead the offline runner
(``llmharness.replay.offline_driver.replay_pipeline_over_trajectory``)
synthesises extractor + auditor records retroactively from the
captured ``final_messages`` and persists them to
``<sid>.baseline_offline.jsonl`` for the strict-A/B stitch. Use this
variant when you want a strict A/B that excludes the cost of in-line
extractor during the control rollout — the control is then the cheapest
possible baseline trajectory.
