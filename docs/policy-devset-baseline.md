# Dev-set baseline: 12 Senior SWE-Bench tasks, k=3

The 12 `train` tasks are the development set for checklist and acceptance work;
the remaining tasks are held out and must stay unread until the dev set is
stable. This is the no-intervention baseline they are measured against.

Run on 2026-07-26. Job `jobs/2026-07-26__11-31-03`, trajectory schema
`harbor_base_0726`.

```bash
AGENTM_TRAJECTORY_SCHEMA=harbor_base_0726 \
ARK_BASE_URL=http://101.126.39.61:8088/v1 ARK_API_KEY=$K \
SSB_JUDGE_MODEL=openai/DeepSeek-V4-pro \
uv run --no-sync python .agentm/harbor-gpt55/run_better_auth_eval.py \
  azure-gpt train -k 3 -n 12
```

`AGENTM_CHECKLIST_WATCH_ENABLED` is unset, so `policy_engine` logs
`symbol sync only (interventions disabled)`: no checks, no `submit`, no
acceptance review. Confirm that line appears before trusting a baseline.

## Results

Each cell is the three attempts in order. `-` means the metric does not exist
for that task, not that it failed. Tasks without a functional suite have no
`verifier_score`, tasks without user stories have no `validation_score`.

| Task | n | reward | verifier_score | validation_score |
|---|---|---|---|---|
| teleport-add-traits-matching-logic | 3 | **1/1/1** | 1/1/1 | 1/1/1 |
| better-auth-fix-api-return-response | 3 | 1/0/1 | 1/0.25/1 | – |
| electric-fix-elixir-client-cache | 2 | 1/0 | 1/0.5 | – |
| immich-fix-server-live-photo | 3 | 0/0/1 | 0.5/0.5/1 | – |
| harbor-add-agent-file-retention | 3 | 0/1/0 | 0/1/1 | 0.667/1/0.667 |
| turborepo-perf-reuse-input-hashes | 3 | 0/0/0 | 0.75/0.75/0.75 | – |
| prefect-fix-resolve-race-condition | 3 | 0/0/0 | 0.667/0.667/0.667 | – |
| plausible-feat-shared-dashboard-deeplink | 3 | 0/0/0 | – | 0.667/0.667/0.667 |
| posthog-feat-approval-gating | 2 | 0/0 | 1/1 | 0.333/0.333 |
| gitea-fix-diff-highlight-overlap | 2 | 0/0 | 0.286/0.286 | – |
| paperless-ngx-perf-document-counts | 3 | 0/0/0 | 0/0/0 | – |
| firezone-feat-portal-add-recent | 1 | 0 | 1 | 0.4 |

**1 of 12 passes on every attempt. 4 more pass at least once. 7 never pass.**

`reward` requires the functional tests to pass *and* every validation story to
pass, so `verifier_score: 1.0` with `reward: 0.0` is normal and common here.
Posthog and firezone both look perfect on functional tests and fail on stories.

## k=3 is load-bearing

Four tasks swing across the full range between attempts of an identical
configuration:

| Task | attempts |
|---|---|
| better-auth | 1.0 / **0.25** / 1.0 |
| immich | 0.5 / 0.5 / **1.0** |
| harbor | 0.0 / **1.0** / 1.0 |
| electric | **1.0** / 0.5 |

At k=1 each of these lands in "pass" or "fail" by luck, and any checklist item
fitted to that reading is fitted to noise. Every single-attempt batch before
this one is subject to that.

## What actually fails

Assertion-level, across all attempts. "never" means it failed in every scored
attempt; "flaky" means it passed in some and failed in others.

| Task | Assertions that never pass |
|---|---|
| gitea | `TestVerifyDiffTagWrapping/{keyword_class, name_class_different_payload, operator_class_alphanumeric_payload}`, `TestVerifyDiffTagWrapping`, `TestVerifyMultiTokenPrefixAdditionPreservesStructure` |
| posthog | stories `threshold_gating`, `change_amount_gating`, `multi_location_detection`, `multi_policy_conflict_response` |
| plausible | story `query_param_return_to_ignored` |
| prefect | `test_compound_trigger_fires_exactly_once_under_concurrency` |
| turborepo | `test_turborepo_scm_lib_tests_pass` |
| paperless | `shell::verify` |
| harbor | `python::verify` |

| Task | Flaky assertions |
|---|---|
| better-auth | the three `HTTP request contexts return Response` branches |
| electric | `bounded_retry_with_clear_error_on_persistent_stale_cdn`, `cache_buster_param_on_stale_retry` |
| immich | both `live photo album / date migration` cases |
| firezone | `{client,policy,resource}_panel_lists_authorization` |
| harbor | `trajectories_only_pruning`, `trajectories_only_on_remote_download_path` |

## Cause analysis

### The deterministic failures share one shape

Every task that never passes states a set of specific behavioural claims and
demonstrates almost none of them. Not a coverage gap, not a capability gap: the
agent addresses the requirement, believes it works, and never runs the thing
that would show it does. This is dimension D6, belief-evidence consistency.

| Task | What it claimed | What it ran | What broke |
|---|---|---|---|
| posthog | Its closing summary lists threshold gating, delta/magnitude gating, and a 400 on multiple matching policies | One self-written test, `test_rollout_threshold_policy_gates_only_matching`, covering one of the three | All four gating stories fail, including the one it tested |
| plausible | "destination-like query params such as `redirect=` or `next=` are preserved as filters, not used as redirect targets" | `mix test …stats_controller_test.exs:1659`, once, then only `mix format` and `mix compile` | `query_param_return_to_ignored` |
| turborepo | Input-hash reuse | `cargo test -p turborepo-scm package_deps…` in all three attempts, never the unfiltered `-p turborepo-scm --lib` | The unfiltered suite exits 101: the fix broke an invariant a sibling test already pinned |
| paperless | Listings are fast again | 18 bash commands: targeted pytest, ruff, grep, git diff. **Zero timing measurements** | `shell::verify`, a latency budget |
| prefect | Fixed the race condition | The existing composite-trigger tests, which are not concurrent | `test_compound_trigger_fires_exactly_once_under_concurrency` dies with `DeadlockDetectedError` |
| gitea | Diff tag nesting fixed | A probe over the `nx` token class | `keyword`, `operator`, and different-payload cases are still inverted |

Three of these are reachable by a check about process, because the missing step
is nameable in advance from the task itself: paperless asks for a latency
budget and nothing was timed, prefect asks about concurrency and nothing ran
concurrently, turborepo's change is package-wide and only one module was run.
The other three need something that reads the claims and demands a
demonstration per claim, which is the acceptance reviewer's job.

The reviewer is not yet up to it. It failed this same gitea case twice on
2026-07-26: once testing a single token class, once testing ten inputs that
were all the same token class.

### The flaky failures are not explained by anything measurable in the run

Verification volume does not discriminate pass from fail:

| Task | test commands, per attempt |
|---|---|
| better-auth | 13 (pass) / 8 (fail) / 9 (pass) |
| immich | 18 (pass) / 12 (fail) / 14 (fail) |
| electric | 10 (pass) / 7 (fail) |
| harbor | 2 (pass) / 2 (fail) / 4 (fail) |

Nor does where the run ends: better-auth's failing attempt stopped mid
exploration while its passing attempts ended on a verification, but electric's
passing attempt ended on an edit 45 calls after its last test and passed
anyway. The graded assertions here are design details the model either arrives
at or does not (better-auth: whether a hook's return value is wrapped in a
`Response`). Treat this group as sampling variance until some feature of the
run is shown to predict it.

### firezone

One scored attempt of five. Its three failing stories all assert on an
`overflow_*` element missing from the rendered HTML, which is the same
pagination behaviour the discarded validation scripts also probed. Not enough
data to classify.

## Infrastructure notes

Three failure modes cost 8 of 36 trials. All are environment, not agent.

**ARL exec timeout, 1800 s.** `agentm_harbor/arl_environment.py` sets
`_DEFAULT_EXEC_TIMEOUT_SECONDS = 1800`, and Harbor's verifier issues its test
command without `timeout_sec`. Long verifiers die at 30.5–30.9 min with
`RewardFileNotFoundError` or `ValidationError` depending on where the output was
cut. Harbor's own `--verifier-timeout-multiplier` does not govern this. Raising
it to 12 changed nothing. Distinct from `VerifierTimeoutError` (electric, 25.0
min), which *is* Harbor's timeout.

**Warm-pool pod loss.** Every session for an image shares one pool and one pod.
When any sibling session hits its 2 h idle timeout the pool scales down, the pod
goes, and any *busy* forked session on that pod dies with
`Dropping session <id>: runtime lost (pool=…, pod=…)`. Harbor then reports
`DownloadVerifierDirError`. Observed on `pool=gitea-…-4b240eec5a48` (pods
`-8qcvn`, `-49lt6`) and `pool=firezone-…-c4989ecac5e0` (pod `-469d2`). A forked
session's own activity does not keep its pod alive, so fork-based re-judging is
a coin flip whenever siblings are near expiry, which `--no-delete` guarantees,
since it leaves 36 sessions ticking toward the same deadline.

**Validation-agent script truncation.** One firezone run had its generated
scripts cut mid-procedure; the pre-retry review discarded all stories for
`expected_fidelity`, leaving `total_stories: 0` and an empty reward file. The
other firezone run produced 5/5. Intermittent, not model-specific.

Judge reliability itself is fixed: the Volcengine weekly quota that emptied
four tasks' rewards across three earlier batches is gone with
DeepSeek-V4-pro at `101.126.39.61:8088`, and teleport, never scorable before,
now scores 1.0 three times.

## Data completeness

28 of 34 attempts scored. The six missing cells are all on tasks already
classified as deterministic failures whose scored attempts are identical
(gitea 0.286 twice with the same five assertions, posthog 0.333 twice with the
same four stories), so filling them would not move any conclusion. firezone is
the one real gap: 1 of 5 attempts survived.

See also `contrib/extensions/policy/README.md` for the architecture this feeds,
and `docs/policy-feedback-dimensions.md` for the dimension model.
