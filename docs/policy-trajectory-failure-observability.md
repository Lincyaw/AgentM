# Structural Observability for Agent Failure Detection

Status: analysis and design note; superseded in part by
`docs/policy-anomaly-detection.md` (two-stage positioning, coarse-grained
action model replacing command enumeration, recall-under-budget calibration,
and the corrected `DecideEvent` finding)

Date: 2026-07-22

This note records the discussion prompted by the 12-task GPT evaluation in
`jobs/2026-07-21__22-27-02`. Its purpose is to determine which failure risks can
be detected by the policy engine without asking an LLM to understand the task or
judge whether the implementation is semantically correct.

The immediate scope is observability and deterministic detection. Critic or
subagent activation is deliberately out of scope until the policy layer can
produce useful, well-calibrated structural signals.

## Executive conclusion

The policy engine should not try to infer correctness from generic trajectory
counts. Without semantic judgment, its useful target is narrower:

> Detect whether the trajectory established a sufficiently recent, resolved,
> broad, independent, and diverse feedback loop in which an incorrect solution
> had a realistic opportunity to fail.

This is a test of falsifiability, not a test of correctness.

The distinction matters because a wrong implementation and a correct
implementation can produce the same observable sequence of edits and green
tests. No deterministic policy can distinguish those cases unless it receives
additional evidence such as branch coverage, a static invariant, a visual
oracle, an independent test suite, or a verifier result.

The current policy data is partially sufficient to reconstruct structural
feedback closure offline. The live DSL is not yet sufficient. Its largest
immediate defect is even more basic: many real validation commands are not
classified as validation at all.

## Evaluation context

The 12 tasks were:

| Task | Current interpretation |
| --- | --- |
| BetterAuth | Confirmed pass |
| Electric | Failed: implemented the immediate stale-handle case but not the complete bounded retry behavior |
| Gitea | Failed: implemented the DOM wrapping invariant in the opposite direction |
| Immich | Confirmed pass |
| Firezone | Failed: fixed-size recent-result query with no continuation beyond the first page; one rubric expectation may be ambiguous |
| Harbor | Failed: read the exact filename patterns but later used a broader `trajectory*.json` glob |
| Paperless | Failed performance thresholds despite passing functional count tests |
| Plausible | Failed to treat `return_to` as an untrusted destination |
| PostHog | Replay invalid because the verifier environment lacked `SECRET_KEY`; do not classify as a model failure yet |
| Prefect | Confirmed pass after repeatedly using failing tests as feedback and closing them with broader successful runs |
| Teleport | Confirmed pass |
| Turborepo | Failed existing regression tests outside the selected test filters |

Within this batch, all 12 system prompts had the same content hash:
`eb666309ac4072aca08c611f7582b6ab392698eca1efcdf475c919dc468eebf6`.
There is therefore no within-batch evidence that system-prompt variation caused
the outcome differences. Determining whether the prompt changed from an older
batch requires an older prompt as a baseline.

## Failure taxonomy discussed

The initial three-way classification was:

1. incomplete understanding;
2. misunderstanding or semantic drift;
3. correct understanding followed by a wrong implementation.

The trajectories suggest a broader causal taxonomy:

1. **Requirement undercoverage**: only part of the requested behavior enters the
   working task model.
2. **Semantic drift**: the agent solves an adjacent or inverted problem.
3. **Incorrect diagnosis**: the observed symptom is attributed to the wrong
   mechanism or component.
4. **Evidence absorption failure**: relevant evidence is read but does not
   constrain the later implementation.
5. **Incomplete or regressive implementation**: the intended change is only
   partially realized or breaks behavior outside the local target.
6. **Validation closure failure**: validation is absent, stale, too narrow,
   self-confirming, or remains red at termination.
7. **Oracle, harness, or environment failure**: the recorded reward does not
   validly measure the implementation, as in the current PostHog replay.

At a first-principles level, the common model-side failure is not simply
"attention." The full chain is:

```text
task
  -> complete operational constraints
  -> causal diagnosis
  -> faithful implementation
  -> discriminating validation
  -> evidence-updated stopping decision
```

The recurring control failure is that the loop does not reliably force negative
evidence back into the current hypothesis and implementation. Feedback may
exist, but it is frequently weak, self-authored, narrow, stale, or ignored.

## Correction to the first observability proposal

Several initially proposed metrics were still semantic judgments in disguise:

- `requirement_coverage` requires deciding what the requirements mean;
- `evidence_to_mutation_fidelity` requires deciding whether code faithfully
  implements evidence;
- general `validation_modality_alignment` requires knowing which modality the
  task needs;
- `claim_evidence_consistency` generally requires interpreting the final claim;
- deciding whether a counterexample is relevant requires a model of the
  intended behavior.

These may be useful later for a critic, a task-specific verifier, or a static
domain policy, but they should not be presented as generic non-semantic policy
metrics.

## Three levels of non-LLM detection

### Level 0: event and temporal structure

This level uses only tool type, arguments, exit status, paths, hashes, turns,
and ordering. It does not parse source code or task language.

Examples:

- a validation failed and was never followed by a successful validation of the
  same or broader scope;
- the last code mutation occurred after the last successful validation;
- all validation commands used exact test selectors;
- mutation spanned several repository scopes but validation covered only one;
- every executed test was created or modified in the current session;
- an unchanged failed command was repeated without a code or input change;
- the session terminated immediately after unresolved negative evidence;
- source state oscillated between previously seen hashes.

### Level 1: syntax and program structure

This level may parse commands, manifests, diffs, ASTs, control-flow graphs, or
data-flow graphs, but it does not interpret the business requirement.

Examples:

- a newly introduced configuration field has no read site;
- a new exception type is never raised;
- a retry counter is introduced but no control-flow back edge exists;
- a fixed `LIMIT N` is introduced without iteration or cursor state in the
  calling control flow;
- an exact selector is replaced by a higher-entropy wildcard;
- a new branch is never executed by the validations run after the mutation;
- an untrusted input reaches a redirect sink without a recognized sanitizer.

These are deterministic analyzers. They encode structural invariants, not a
general natural-language understanding of the task.

### Level 2: literal task contract

This level extracts only explicit, mechanically recognizable task artifacts:

- quoted identifiers and paths;
- numeric limits;
- command names;
- exact error strings;
- explicit modality words such as `benchmark`, `browser`, `EXPLAIN`, or
  `full suite`.

It can check whether those literals appeared in reads, mutations, or
validation. This is still not semantic requirement coverage. It should not
infer unstated obligations or synonyms.

## Core structural dimensions

The most useful generic dimensions are:

| Dimension | Structural question |
| --- | --- |
| Recency | Did relevant validation occur after the final mutation? |
| Resolution | Was every observed validation failure superseded by a successful run of the same or broader scope? |
| Breadth | Did validation cover the repository scopes affected by the mutations, and was at least one run unfiltered? |
| Independence | Did validation include pre-existing tests or another oracle not authored together with the implementation? |
| Diversity | Did the trajectory use more than one narrow form of evidence, such as test plus compile, integration, coverage, benchmark, or external verifier? |

These dimensions measure whether the solution was exposed to meaningful
falsification pressure. None of them proves that a green result is correct.

## Candidate deterministic signals

### Terminal unresolved failure

```text
validation failed
AND no later validation of equal or broader scope passed
AND the session is stopping
```

This is the strongest current Level 0 signal.

- Firezone: the targeted `mix test` failed at turn 118; later actions included
  formatting and compilation, but no passing test.
- Plausible: targeted tests failed at turns 48 and 58; no later test passed.
- Prefect is the useful contrast: failures at turns 70, 73, and 76 were followed
  by successful targeted and broader runs at turns 80 through 82.

### No validation after the final mutation

```text
last_mutation_turn > last_relevant_validation_turn
```

The current `calls_since_last_mutation` metric is not an adequate substitute.
Formatting, reading, searching, or compiling can increase that count without
validating behavior.

### Mutation-to-validation scope gap

Repository scopes can be derived mechanically from paths, working directory,
and the nearest package or workspace manifest:

```text
scope_gap = mutated_scopes - validated_scopes
```

Turborepo is the clearest motivating example: the agent mutated seven files
across multiple crates, then ran only named test filters. It never ran the full
`cargo test -p turborepo-scm --lib` suite that contained the regressions.

### Narrow-only validation

Normalize common selector forms:

```text
pytest path::test_name
pytest -k expression
cargo test exact_test_name
go test -run regex
mix test file.exs:line
vitest specific-file
```

Useful derived fields include:

```text
narrow_validation_ratio
maximum_validation_scope
has_unfiltered_suite_after_last_mutation
```

Electric, Paperless, Plausible, and Turborepo all show narrow-only or heavily
filtered validation. This feature is a risk indicator, not a correctness
classifier: successful tasks such as BetterAuth also used narrow tests.

### Self-authored validation only

```text
executed_test_files subset_of files_created_or_modified_in_session
```

An even stronger form is:

```text
no_pre_existing_test_executed_after_last_mutation
```

This detects a lack of oracle independence. Harbor validated the retention
behavior primarily through the newly added
`tests/unit/test_agent_file_retention.py`; Electric mainly ran modified target
tests and a specific line. The signal must be combined with scope and breadth
because successful implementations also add and run focused tests.

### Ad hoc oracle only

Classify validation provenance without judging its content:

- pre-existing repository test;
- session-created or session-modified test;
- temporary test under `/tmp`;
- heredoc or inline script;
- compile, typecheck, or lint;
- CLI smoke check;
- external verifier.

Gitea used several temporary `tmp_debug_test.go` programs, while Harbor used
inline CLI-help smoke checks. This is valuable context for a compound risk
rule, but should not fire as an error by itself.

### Unabsorbed negative evidence

Normalize failures by runner, repository scope, selector, and error
fingerprint, then detect:

- failure followed by no rerun;
- failure followed only by format, lint, or compile;
- a broader failure replaced by a narrower passing command;
- exact repeated failure with no intervening mutation or input change;
- shell structures such as `|| true` or pipelines that hide the meaningful
  exit status.

### Structural mutation instability

Use content and hunk hashes to detect:

- repeated rewriting of the same hunk;
- `A -> B -> A` state cycles;
- repeated addition and deletion of the same symbol;
- substantial mutations after a passing validation;
- expanding mutation scope without expanding validation scope.

This is more closely tied to feedback closure than raw repeated-read counts.

## What the seven genuine failures expose structurally

| Case | Non-semantic signal available | What remains undetectable without an additional oracle |
| --- | --- | --- |
| Electric | Narrow validation, mostly session-modified tests, no broad suite; a CFG or coverage analyzer could check repeated retry structure | The exact intended stale-retry semantics from generic event counts alone |
| Gitea | Heavy use of ad hoc tests; no independent visual/browser oracle if that modality is mechanically required | The inverted DOM wrapping invariant when implementation and tests are mutually consistent |
| Firezone | Terminal unresolved test failure; large mutation surface; no passing behavioral test after the failure | Whether pagination is the exact missing behavior without a task or static contract |
| Harbor | Validation concentrated in a newly authored test; syntax analysis can flag widening from an exact selector to a broad glob | Which deceptive filenames must be excluded unless supplied by a contract or verifier |
| Paperless | Only targeted functional tests; no benchmark or query-plan evidence if the task literally requests performance verification | Whether the SQL plan meets the intended performance threshold without measuring it |
| Plausible | Two unresolved targeted test failures at termination | The exact `return_to` trust invariant without a security/data-flow policy |
| Turborepo | Mutation-to-validation scope gap and filtered-only validation | The exact regression behavior without running the omitted tests |

The table intentionally does not claim that every failed task is detectable by
a generic policy. Firezone and Plausible have high-confidence Level 0 signals.
Turborepo has a strong scope signal. Electric, Harbor, and Paperless have useful
compound risk signals. Gitea demonstrates the hard observability boundary.

## The indistinguishability boundary

Suppose two trajectories have identical observable facts:

```text
same files read
same files edited
same command shapes
same exit codes
same green tests
same stopping point
```

but one implementation satisfies the requirement and the other implements an
inverted DOM invariant. A policy over those observations must return the same
answer for both trajectories. It cannot infer correctness by adding more
aggregate counters.

Gitea is close to this case. The agent ran the relevant Go package tests and
they passed, but the tests encoded the same wrong invariant as the
implementation. Reliable detection therefore requires at least one new source
of information: a pre-existing adversarial test, a DOM invariant checker, a
browser/visual oracle, or a semantic review.

This boundary should be explicit in the policy design. Structural policy is a
risk detector and feedback gate, not a universal verifier.

## Findings about the current policy implementation

### Configured policy surface

The current Harbor scenario explicitly configures only
`package:ifg_evidence.yaml`. The loader reads only explicitly configured policy
layers. The skill documentation that describes an automatically loaded
18-rule base policy is stale relative to the inspected implementation.

The configured file currently contains these observe-mode rules:

- `exploration-not-converging`;
- `mutation-target-drift`;
- `unvalidated-intervention`;
- `unchanged-anchor-reentry`;
- `investigation-anchor-churn`;
- `created-artifact-replacement`;
- `unchanged-investigation-state-cycle`;
- `repeated-region-reading`.

### Current rules mostly measure complexity

Across the seven genuine failures and four confirmed passes:

| Rule | Failed sessions hit | Passing sessions hit | Interpretation |
| --- | ---: | ---: | --- |
| `repeated-region-reading` | 7/7 | 4/4 | No discrimination |
| `unchanged-anchor-reentry` | 7/7 | 4/4 | No discrimination |
| `mutation-target-drift` | 5/7 | 3/4 | Mostly task complexity |
| `unvalidated-intervention` | 4/7 | 2/4 | Weak and noisy |

These signals may still describe investigation state, but the present data does
not support treating them as proxies for correctness or completeness.

### Validation command classification is incomplete

The command schema recognizes direct pytest/vitest commands and some package
manager forms, but misses common real commands:

- `uv run pytest`;
- `python -m pytest`;
- `python manage.py test`;
- `mix test`;
- `go test`;
- `cargo test`.

As a result, the IFG classified real test executions as `reference` or `exec`:

- Electric and Firezone `mix test`;
- Gitea and Teleport `go test`;
- Harbor, Paperless, and most Prefect `uv run pytest`;
- Turborepo `cargo test`.

This explains why several intervention summaries report zero validation
attempts even for sessions that visibly ran tests. Until this grammar is fixed,
validation-related policy statistics are not trustworthy.

### Raw persistence is richer than the live DSL

The per-session SQLite databases persist useful raw facts:

- `policy_tool_events` stores complete `args_json`, `result_json`, exit code,
  timing, hashes, and processed metadata;
- `ifg_actions` stores normalized command, action kind, family, template,
  source, and raw evidence;
- IFG tables persist action, symbol, and edge information.

However, live `PolicyState` retains mostly rolling aggregates and drops the
historical result and edit content needed for relational comparisons. The DSL
also cannot enumerate and relate evidence propositions, mutation targets, and
validation observations. Offline evidence may therefore exist without being
expressible as a live rule.

### No terminal policy decision point

Correction (2026-07-22, later the same day): the mechanism exists in core.
`DecideEvent` fires every turn with a `default_action` that becomes `Stop`
when the model ends its turn, and handlers may return `Inject`
(`goal.py` already uses this pattern). The policy engine simply does not
subscribe to that channel yet. Session shutdown remains too late; the
pre-stop gate is one subscription away, with no core change required.

## Proposed normalized observations

The next data-model step should add normalized structural observations rather
than more aggregate counters.

### Mutation observation

```text
turn
repository_generation
paths
repository_scopes
pre_hashes
post_hashes
changed_hunks
introduced_test_files
```

### Validation observation

```text
turn
runner
kind
cwd
repository_scopes
selectors
is_narrow
outcome
error_fingerprint
executed_test_paths
test_provenance
repository_generation
```

### Validation closure state

```text
last_mutation_turn
last_passing_validation_turn
mutations_since_last_pass
unresolved_failure_fingerprints
mutated_scopes
validated_scopes
has_unfiltered_validation
has_pre_existing_test_validation
```

The policy DSL should eventually be able to express queries such as:

```text
validation.unresolved_failure()
validation.none_after_last_mutation()
validation.only_narrow_after_last_mutation()
validation.only_session_authored_tests()
validation.missing_mutated_scopes()
validation.broader_failure_replaced_by_narrower_pass()
```

The exact API names remain a design choice. The important point is that they
represent relations between mutation, validation, and termination rather than
standalone counts.

## Recommended implementation order

Superseded by `docs/policy-anomaly-detection.md`. Steps 1 and 2 (validation
command normalization and runner/selector/scope persistence) are dropped in
favor of the coarse-grained action model: bash is assumed to only search and
execute, and supersession, breadth, and provenance are approximated by three
syntactic relations (template identity, argv subset, argv-path intersection
with the session mutation set) plus the declared `purpose` field. Step 5 is
resolved by subscribing to the existing `DecideEvent` channel. Steps 6 to 8
are replaced by the two-stage design in which structural signals are cheap
critic triggers rather than standalone rules.

## Evaluation criteria for new signals

Each signal should be assessed as a detector, not justified by a plausible
story. At minimum record:

- hit rate on genuine failed sessions;
- hit rate on confirmed passes;
- whether it fires before the final mutation or stop decision;
- whether the diagnostic points to a concrete missing feedback action;
- whether the signal clears after the agent performs that action;
- incremental token and wall-time cost if later connected to intervention.

A useful structural diagnostic should be actionable. For example, "the latest
behavioral test remains unresolved" is actionable; "you read the same region
again" often is not.

## Decision recorded

The immediate policy goal is not semantic completeness detection. It is to
build a deterministic account of validation closure and falsification pressure.
The first implementation work should therefore focus on command
classification, normalized mutation-validation relations, and terminal closure
state. Semantic critics, counterexample generation, and verifier dispatch are
later consumers of those signals, not substitutes for missing observability.
