# Two-Stage Anomaly Detection for Agent Trajectories

Status: design note, pending data validation

Date: 2026-07-22

This note consolidates the design discussion that followed
`docs/policy-trajectory-failure-observability.md` and supersedes parts of it
(see "Supersessions" at the end). It defines the positioning, the detection
model, the feature set, and the validation protocol to run once the trajectory
databases for the labeled evaluation batches are available locally.

## Positioning

The policy layer is a mediating proxy between the model and the environment.
Its job is to maintain the feedback loop: surface the model's own process
problems to the model, and surface environment problems (broken harness,
infrastructure faults) so the model does not misattribute them. It does not
make decisions for the model.

The architecture is a two-stage detection system:

```text
stage 1: lightweight structural anomaly triggers
  deterministic, cheap, recall-oriented, runs on every session
    -> fires a candidate anomaly with structural evidence
stage 2: heavyweight critic review
  an investigating model, triggered only by stage 1
    -> reads the actual trajectory, using stage-1 evidence as leads
    -> verdict: confirm or suppress
confirm -> feedback injected to the working model (notify),
           optionally a recovery compaction
suppress -> alarm suppressed, verdict logged as a stage-1 label
```

Consequences of this split:

- Stage 1 optimizes recall under a trigger budget, not standalone precision.
  The budget is trigger rate times critic cost per invocation. Weak signals
  (for example narrow-only validation, which also occurs in successful
  sessions) are acceptable as trigger conditions.
- The working model never sees raw structural signals. It only receives
  critic-confirmed feedback with semantic grounding and a concrete missing
  action. This removes the alert-fatigue failure mode observed when an agent
  dismissed raw policy stats as irrelevant.
- Every critic verdict is a free online label for stage 1. Signals whose
  suppress rate grows are automatically raised in threshold or dropped.
  Calibration is continuous, not a one-time offline exercise.
- The indistinguishability boundary (Gitea-class failures where the
  implementation and self-authored tests encode the same wrong invariant)
  softens at the system level: structure cannot decide such cases, but a cheap
  cue such as "zero independent oracle" is exactly the state where semantic
  review has the highest expected value. The boundary remains inside stage 1;
  the system as a whole can reach past it.

Permitted interventions remain unchanged: inject feedback (notify) and request
recovery compaction. The critic is read-only. Enforcement gates are out of
scope.

## Coarse-grained action model

Enumerating shell commands in a schema is not a practical path. The model is
simplified with one assumption:

> Read, edit, and write go through dedicated tools. Bash performs only
> search and execution.

The action alphabet is therefore:

| Action | Source | Fidelity |
| --- | --- | --- |
| read | read tool | path, content hash |
| mutate | edit/write tools | path, before/after hash, hunks |
| search | bash segments with path/pattern operands and no side effects | paths, patterns |
| exec | remaining bash segments | argv, exit code, duration, fingerprint |
| stop | loop decision event | default action |

No attempt is made to decide what an exec is doing (test, build, benchmark).
The finer semantics are recovered from three syntactic relations and one
declaration, none of which require command knowledge:

1. **Template identity.** `ifg_actions.template` already normalizes commands.
   "A template that failed was never later run with exit 0" needs no
   understanding of the command.
2. **Argv subset.** Narrowing a run almost always adds tokens
   (`pytest a.py::t` versus `pytest a.py` versus `pytest`). A later exit-0
   exec whose argv is a subset of a failed exec's argv approximates
   "superseded by an equal-or-broader run", across languages, with zero
   enumeration.
3. **Argv-path intersection with the session mutation set.**
   `ifg_path_candidates` already extracts path references from commands. An
   exec whose referenced paths all fall inside the set of files created or
   modified by this session's write tools is validating against self-authored
   material. This is the mechanical form of oracle independence.
4. **Purpose declaration.** The `tool_purpose` atom injects a required
   `purpose` field into every tool call, and the value survives in
   `policy_tool_events.args_json` and in the trajectory turn payloads. The
   model's own declaration provides the intent layer (verify, reproduce,
   explore) that the syntactic layer refuses to infer. Declared intent versus
   observed structure is checked by diff, and a mismatch is itself a signal.

The coarse assumption is monitored, not trusted: bash write-redirect detection
(`_has_write_redirect`) yields a per-session **write-leak rate**. Sessions
where bash frequently mutates files degrade the model's fidelity; the leak
rate is carried as a data-quality feature and high-leak sessions are
down-weighted.

## Episode model

A trajectory is segmented mechanically into repair episodes: a maximal run of
mutations, followed by the search/exec events up to the next mutation or the
end of the session. Failure discrimination is expected to live in **how
episodes terminate**, not in surface tidiness. Successful trajectories may be
messy (repeated edits, long exploration) while still closing every episode;
the recurring failure shape is a final episode that ends red, ends unexecuted,
or never opens.

## Stage-1 feature set

All features are computable from existing persistence
(`policy_tool_events`, `ifg_actions`, `ifg_path_candidates`,
`ifg_action_file_edges`, `policy_file_state`) with no new collection:

| # | Feature | Definition | Evaluated at |
| --- | --- | --- | --- |
| F1 | terminal unresolved template | a failed exec template has no later exit-0 run whose argv is a subset of the failed argv | pre-stop |
| F2 | no execution after final mutation | last mutation turn is later than the last exec turn, or no exit-0 exec follows it | pre-stop |
| F3 | self-authored oracle only | post-mutation execs reference only paths created or modified in this session | pre-stop |
| F4 | zero independent oracle | no post-mutation exec references any pre-existing path | pre-stop |
| F5 | declared-verify closure | no exec declared as verification (purpose lexicon) after the final mutation, or all such execs failed | pre-stop |
| F6 | episode non-closure | fraction of episodes ending without an exit-0 exec; whether the final episode is unclosed | per episode / pre-stop |
| F7 | purpose-structure mismatch | declared purpose says verification but the call is not an exec, or declared verification never appears | per call, aggregated |
| F8 | state cycling | `reverts_to_prior_hash`, repeated rewriting of the same unit hash | per mutation |
| Q1 | write-leak rate | bash segments with write redirects / all bash segments | data quality, not a trigger |

The purpose lexicon in F5/F7 is a small word list over free text and is
treated as a monitored heuristic, not a ground truth.

Known non-coverage, accepted deliberately:

- **Scope gap** (Turborepo: mutated several crates, validated one filter).
  Package-selector tokens are not paths, so relation 3 misses them. Recovery
  path, if ever needed, is a workspace-manifest layer, not command
  enumeration.
- **Task-contract modality** (Paperless: task requires performance evidence).
  This belongs to a one-shot literal contract extraction per task, a separate
  layer.
- **Mutually consistent wrong invariants** (Gitea). Inside stage 1 this stays
  undetectable; F4 is the cheap cue that routes such sessions to the critic.

## Expected hits on the labeled batch

Validation targets against `jobs/2026-07-21__22-27-02` (post-replay labels).
The table records predictions to check once trajectory data is available;
features failing these expectations are recalibrated or dropped.

| Trial | Label | Expected stage-1 fires |
| --- | --- | --- |
| BetterAuth | pass | none (narrow tests alone must not fire) |
| Immich | pass | none |
| Prefect | pass | none: failures at turns 70/73/76 are superseded by exit-0 subset runs at 80-82 |
| Teleport | pass | none |
| Electric | fail | F3 (mostly session-modified tests) |
| Firezone | fail | F1 (mix test failure never superseded), F2 partially |
| Harbor | fail | F3 (session-created retention test as sole oracle) |
| Paperless | fail | weak or none (accepted: task-contract layer) |
| Plausible | fail | F1 (two unresolved targeted failures) |
| Turborepo | fail | weak or none via F1-F8 (accepted: scope gap) |
| Gitea | fail | F4 only (routes to critic; structure cannot decide) |
| PostHog | invalid | environment-fault class, excluded from calibration |

Aggregate acceptance for the trigger set: recall of at least 4 of 7 genuine
failures with zero fires on the 4 passes, at a trigger rate whose critic cost
is acceptable. Weak-signal features may fire on passes only if their role is
critic routing and the measured budget allows it.

## Trigger points

- **Pre-stop is the first-class trigger point.** The mechanism already
  exists: `DecideEvent` (core ABI) fires every turn with a `default_action`,
  which becomes `Stop` when the model ends its turn. A policy subscription to
  this channel can observe and, after critic confirmation, `Inject` feedback
  before the stop commits. No core change is required. Five of the seven
  genuine failures have their strongest predicate at this point.
- Post-episode (a mutation cluster closed) for F6/F8.
- Turn-committed remains for cheap bookkeeping only.

## Critic contract

- Input: the fired signals with their structural evidence (unresolved
  templates, self-authored oracle list, episode timeline) plus pointers into
  the trajectory. The evidence is a set of leads, not conclusions; the critic
  is an investigating agent that reads the actual turns.
- Output: a verdict (`confirm` or `suppress`). On confirm, a feedback message
  for the working model in plain language: what happened, what the world
  state is, what the next concrete action is. No policy jargon. Optionally a
  recovery-compaction recommendation per
  `docs/policy-guided-context-recovery.md`.
- Every verdict is persisted and joined back to the firing signal as a label.

## Calibration protocol

1. Join per-trial trajectory databases with trial labels
   (159 labeled trials exist across local job directories; reward,
   verifier-per-test, rubric-per-criterion, validation stories).
2. Compute the F1-F8/Q1 vector per session; produce per-feature fail-hit rate
   versus pass-fire rate, and fire timing relative to the stop decision.
3. Prefer within-task contrast pairs (same task, pass and fail rollouts)
   wherever multi-rollout data exists; this controls the task-difficulty
   confounder that made earlier progress-style rules (7/7 fail hits, 4/4 pass
   hits) useless as discriminators.
4. Pre-register each feature's expected direction before measuring, to avoid
   fishing in a small sample.
5. After deployment, fold critic verdicts in as continuous labels.

Data prerequisite: the labeled trials' policy databases and trajectory schema
currently live on the evaluation machine; labels are local. Validation starts
when both sides of the join are in one place.

## Supersessions

Relative to `docs/policy-trajectory-failure-observability.md`:

- The recommendation to fix validation command classification by extending
  `bash_command_schema.yaml` is **dropped**. The coarse-grained action model
  plus the three syntactic relations replace command enumeration. The schema
  stays as-is for search/read classification of bash segments.
- The claim that no terminal policy decision point exists is **corrected**:
  `DecideEvent` provides the mechanism in core; the policy engine simply does
  not subscribe to it yet.
- Signal evaluation criteria shift from standalone precision to recall under
  a trigger budget, because stage 2 owns precision.
- The "proposed normalized observations" (runner, selectors, scopes, test
  provenance) are **reduced**: provenance and supersession are covered by the
  syntactic relations over existing tables; runner/selector/scope
  normalization is deferred along with the scope-gap feature.

Relative to `docs/policy-guided-context-recovery.md`: the recovery ladder is
unchanged, but Stage 1 injection is now gated by critic confirmation rather
than sent directly on a structural transition. The fork experiment described
there remains valid as an experiment on raw injection.
