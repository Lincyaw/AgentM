# Refactor v2 code review findings

Last reviewed: 2026-07-22

## Review scope

This ledger records code smells and isolation regressions found while reviewing:

- `origin/main..cbe9f8f2c392`, including historical commits on the branch.
- The uncommitted worktree as observed during the review.

The worktree was changing concurrently, so line numbers may drift. Paths, behaviors, and close criteria are the durable references.

## How to use this ledger

Work through the findings one at a time. Keep each finding's status and decision notes current.

Allowed statuses:

- `open`: confirmed and not yet addressed.
- `in_progress`: actively being addressed.
- `resolved`: the exit criteria are satisfied.
- `accepted`: the behavior is intentionally retained, with rationale recorded.
- `superseded`: replaced by another tracked change or design.

A finding should not be marked `resolved` only because code changed. Its behavioral exit criteria and relevant verification must also pass.

## Summary

| ID | Priority | Status | Finding |
| --- | --- | --- | --- |
| RV2-001 | P1 | resolved | Compaction publication bypasses the atomic trajectory commit |
| RV2-002 | P1 | accepted | Compaction can knowingly return an over-budget context |
| RV2-003 | P1 | resolved | Fork-inherited history is invisible to compactors |
| RV2-004 | P2 | open | Live policy turn summaries use the next turn number |
| RV2-005 | P2 | open | Bash writes can fail to advance repository generation |
| RV2-006 | P2 | open | A contrib atom bypasses Operations, and AM004 does not detect it |
| RV2-007 | P2 | open | The presenter owns concrete compaction policy details |
| RV2-008 | P2 | open | The test-edit hook produces false positives for read-only commands |

## RV2-001: Compaction publication bypasses the atomic trajectory commit

- Priority: P1
- Status: `resolved`
- Area: trajectory consistency and auditability

### Evidence

- `src/agentm/core/abi/store.py:225` exposes `commit_compaction()` as the atomic boundary for state, head, and compaction-boundary publication.
- Before the repair, `src/agentm/presenter/compaction.py` instead read the old
  state, wrote the summary resource, read the head, and called
  `save_content_replacement_state()` without a boundary or head advance.

### Problem and impact

The presenter implements a check-then-write sequence outside the store's atomic compaction contract. This can:

- omit the `compact_boundary` node from the source trajectory;
- leave the audit chain detached from the replacement state;
- overwrite newer state when concurrent work advances the session between reads and writes;
- leave an orphan summary resource if a later publication step fails.

### Recommended direction

Publish through `commit_compaction()`, or introduce an equivalent compare-and-swap/versioned store operation that atomically validates the expected head and commits the resource, boundary, and replacement state.

### Exit criteria

- [x] Compaction publication has one atomic store boundary.
- [x] The committed source trajectory contains an auditable compaction boundary.
- [x] A stale publisher cannot overwrite a newer head or replacement state.
- [x] Partial failure cannot expose a replacement state that references an incomplete publication.
- [x] Concurrency and failure-path behavior has focused regression coverage or a deterministic verification procedure.

### Decision notes

Resolved on 2026-07-22.

- `CompactionResult` now carries a `CompactionSourceAnchor` containing the exact
  source head and final committed turn. Publication rejects the artifact if
  either has changed.
- `TrajectoryCompactionPublisher` writes the immutable summary resource and
  adopts it with one `commit_compaction()` call that atomically appends the
  `compact_boundary`, advances the selected head, and saves replacement state.
- The store validates inside that atomic commit that the boundary still anchors
  the latest committed turn. This also closes the race where a committed turn
  without provider-visible nodes advances history without advancing the head.
- The summary resource can exist without being adopted if the compare-and-swap
  fails, but no incomplete or stale replacement state becomes visible.
- A deterministic JSONL-store probe confirmed that the boundary, head, and
  replacement state identify the same node and that publication of an artifact
  after another turn is rejected. Re-adopting identical summary content from a
  newer source head also produces a distinct, correctly chained boundary.

## RV2-002: Compaction can knowingly return an over-budget context

- Priority: P1
- Status: `accepted`
- Area: provider-context safety

### Evidence

- `src/agentm/extensions/builtin/llm_compaction.py:589` preserves all non-synthetic user messages.
- Around `src/agentm/extensions/builtin/llm_compaction.py:376`, an active summary that covers the last turn causes the projected context to be returned even when `_within_budget()` is false.
- The newly generated summary path around line 443 also lacks a final budget postcondition check.

### Problem and impact

The compactor can return a context that it already knows violates `max_messages` or the effective token budget. A long user history can therefore overflow the downstream model provider despite compaction being enabled. The configured reserve and message limits are not reliable contracts.

### Recommended direction

Make the postcondition explicit: every successful compaction result must fit the effective budget. If the protected-message policy makes that impossible, return a typed unsatisfiable-budget result, or support an explicit policy for compacting older user messages.

### Exit criteria

- [ ] Every successful compaction result satisfies both message and token limits.
- [ ] An impossible protected-message budget produces an explicit, observable outcome.
- [ ] Active-summary reuse and newly generated summaries share the same postcondition check.
- [ ] Long user-only and mixed user/tool histories have focused regression coverage.

### Decision notes

Accepted on 2026-07-22. The deployment contract for this work assumes the
context budget is sufficient. Preserving every real user message is intentional,
and an over-budget projection is not considered a correctness failure under
that contract. No budget-policy behavior was changed as part of the compaction
consistency repair.

## RV2-003: Fork-inherited history is invisible to compactors

- Priority: P1
- Status: `resolved`
- Area: fork isolation and logical trajectory semantics

### Evidence

- `src/agentm/core/runtime/session_factory.py:302` persists a fork's initial turn with `nodes=[]` when an `initial_head` exists and relies on the logical parent for inherited history.
- Before the repair, both `src/agentm/presenter/compaction.py` and the Harbor
  compactor queried only nodes physically owned by the child session.

### Problem and impact

The persistence model represents inherited history through the logical-parent chain, but compactors read only nodes physically owned by the child session. As a result:

- generic summarization may report that there are no persisted messages for the inherited prefix;
- a Harbor fork may retain an intervention message while losing the original user task;
- incremental summaries can omit inherited assistant and tool evidence.

This is a semantic mismatch between session-local storage ownership and the logical trajectory visible to the agent.

### Recommended direction

Build compaction input from the logical chain, for example with `load_chain(..., include_logical_parent=True)` anchored at the current head, or render it from persisted turn snapshots whose contract already includes inherited context.

### Exit criteria

- [x] A fork compacts the same logical prefix that it can observe at runtime.
- [x] Parent-owned nodes remain isolated in storage and are not copied or mutated by the child.
- [x] The original task and inherited assistant/tool evidence survive child-session compaction.
- [x] Nested forks and matching logical-leaf cases have deterministic verification.

### Decision notes

Resolved on 2026-07-22.

- The generic and Harbor compactors capture a stable source head and read its
  active logical chain with `load_chain(..., include_logical_parent=True)`.
- They retry if the source turns or head change while the snapshot is being
  captured, so one artifact cannot combine two source versions.
- Deterministic probes covered a direct fork, a nested fork, successful boundary
  publication, and a fork from the logical leaf covered by an existing compact
  state. Inherited user/assistant history survived, parent-owned nodes remained
  physically isolated, and sibling-only parent history was excluded.
- No test files were added or edited for this repair, following the repository's
  instruction to add tests only when explicitly requested.

## RV2-004: Live policy turn summaries use the next turn number

- Priority: P2
- Status: `open`
- Area: live/replay consistency

### Evidence

- `contrib/extensions/policy/src/policy_engine/state.py:245` records tool evidence using the current `_turn_count`.
- `contrib/extensions/policy/src/policy_engine/__init__.py:773` advances the turn before querying entries for the committed turn summary.
- The offline projector summarizes before advancing, so it does not share the live ordering.

### Problem and impact

The live path queries the next turn after incrementing the counter. `policy_turn_summary` can therefore be empty or summarize the wrong turn, while replay/offline projection produces a different result from the same events.

### Recommended direction

Use one turn-finalization reducer for both live and replay processing. Capture the committed turn identifier, summarize that identifier, then advance state in a single defined order.

### Exit criteria

- [ ] Live and replay paths use the same turn-finalization ordering.
- [ ] A committed turn summary contains the tool evidence recorded in that turn.
- [ ] Multi-turn traces produce identical live and replay summaries.

### Decision notes

The operation-detection false positives remain open. The separate approval
delivery failure observed during this review was fixed on 2026-07-22:

- `UserPromptSubmit` remains the automatic approval path for clients that emit
  it.
- The deny response now includes an operation-scoped fallback token. After the
  user explicitly agrees, the guard's `approve` command marks only that exact
  pending operation as approved, once, without relying on `UserPromptSubmit`.
- Tokens are bound to both the hashed session state and the complete tool-input
  hash; mismatched, expired, or reused approvals are rejected.

## RV2-005: Bash writes can fail to advance repository generation

- Priority: P2
- Status: `open`
- Area: repository mutation tracking

### Evidence

- `contrib/extensions/policy/src/policy_engine/ifg_investigation.py:259` increments support and returns early when a path is already present in repository artifacts, without first handling `relation == "write"`.
- The `_touch_repository(mutated=True)` path around line 301 advances the revision/generation but is skipped by that early return.

### Problem and impact

When a Bash write targets an already known repository artifact, the mutation may not advance repository generation. Later evidence can then classify the changed artifact as unchanged or churn, weakening freshness and causality decisions.

### Recommended direction

Keep the rule that Bash observations do not create repository anchors, but when an anchor already exists and the relation is `write`, always mark the repository as mutated before returning.

### Exit criteria

- [ ] A write to an existing repository artifact advances repository revision/generation exactly once.
- [ ] A read-only observation does not advance generation.
- [ ] A Bash-only observation still cannot create a repository anchor.
- [ ] Subsequent evidence observes the updated generation.

### Decision notes

Pending.

## RV2-006: A contrib atom bypasses Operations, and AM004 does not detect it

- Priority: P2
- Status: `open`
- Area: execution isolation and architectural enforcement

### Evidence

- `contrib/extensions/policy/src/policy_engine/source_parser.py:418` performs direct temporary-file/path I/O and invokes `subprocess.run`; another direct subprocess invocation appears around line 540.
- The real-time IFG path reaches this code from `contrib/extensions/policy/src/policy_engine/ifg/project.py:225`.
- `src/agentm/code_health.py:364` limits AM004 atom-path recognition to paths containing `extensions/builtin`, so contrib package atoms are not covered.

### Problem and impact

The atom bypasses the runtime's Operations abstraction and directly executes host subprocesses and filesystem access. This weakens environment isolation, permission control, auditability, and compatibility with controllers such as Harbor. The project linter currently gives false confidence because its path heuristic misses contrib atoms.

### Recommended direction

Move parsing/execution behind a host or toolbox service exposed through `AtomAPI`, or use the appropriate Operations port. Update AM004 to identify atoms by manifest/reachability or package configuration rather than one directory substring.

### Exit criteria

- [ ] Policy atoms do not directly call subprocess or raw filesystem APIs for this path.
- [ ] The execution is routed through an auditable runtime service or Operations port.
- [ ] AM004 detects equivalent violations in both built-in and contrib atoms.
- [ ] A negative linter fixture demonstrates that ordinary non-atom host code remains allowed.

### Decision notes

Pending.

## RV2-007: The presenter owns concrete compaction policy details

- Priority: P2
- Status: `open`
- Area: mechanism/policy boundary

### Evidence

- `src/agentm/presenter/compaction.py:33` imports the concrete `LlmCompactionConfig` plus private helpers `_serialize_message_for_summary` and `_summary_ref` from a built-in atom.
- The presenter also constructs strategy-specific prompts and references.

### Problem and impact

The presenter is coupled to one built-in atom's private implementation and policy choices. Alternative compaction strategies cannot be plugged in without changing presenter code, and changes to private atom helpers can break the presenter. This reverses the repository's intended dependency direction: the SDK should provide mechanism while atoms provide policy.

The worktree Harbor scenario's atom-local compaction policy is a better ownership direction, although it still shares the inherited-history issue in RV2-003.

### Recommended direction

Keep the presenter responsible for orchestration and lifecycle only. Move prompt construction, serialization policy, summary-reference strategy, and strategy configuration behind an atom/service protocol with stable DTOs.

### Exit criteria

- [ ] The presenter imports no concrete built-in compaction config or private atom helper.
- [ ] A compaction strategy can be replaced without modifying presenter code.
- [ ] Stable ABI DTOs describe requests, results, and publication metadata.
- [ ] The lifecycle still supports cancellation, resume, and atomic publication.

### Decision notes

Pending.

## RV2-008: The test-edit hook produces false positives for read-only commands

- Priority: P2
- Status: `open`
- Area: developer tooling correctness

### Evidence

- `.codex/hooks/pre_tool_use_tests_guard.py:15` uses regular expressions over the raw shell command.
- `_paths_from_bash()` around line 302 can match write-like text anywhere in the command, including quoted search patterns.
- During this review, a read-only `rg` command was blocked because its search text contained write-like syntax and mentioned a test path.

### Problem and impact

The guard conflates command arguments with executed write operations. This interrupts safe inspection, encourages workarounds, and makes the hook less trustworthy without reliably improving test protection.

### Recommended direction

Parse shell command/token boundaries conservatively, recognize known read-only commands, and ignore write-like strings inside quoted search arguments. When parsing is ambiguous, report the exact suspected write target.

### Exit criteria

- [ ] Read-only `rg`, `grep`, and inspection commands are not classified as test edits because of their pattern text.
- [ ] Actual redirection, in-place editing, file removal, and patch commands targeting tests remain blocked as intended.
- [ ] Hook diagnostics identify the parsed operation and target.
- [ ] Positive and negative command examples are covered by focused hook verification.

### Decision notes

Pending.

## Verification baseline

This baseline records repository health at review time; it is not proof that every failure is caused by one finding.

### Tests

After resolving RV2-001 and RV2-003 and removing the four obsolete compact
contract tests, `uv run pytest --tb=short` completed with:

- 147 passed
- 10 skipped

The removed cases asserted the superseded direct provider-message shape or
expected compacted user messages to disappear. The logical-chain, boundary,
fork-isolation, and stale-publication behaviors were verified with the
deterministic probes recorded under RV2-001 and RV2-003.

### Static checks

- `uv run mypy src/` passed.
- Targeted mypy for the generic and Harbor compaction modules passed.
- `uv run ruff check src/ tests/ contrib/` passed.
- `uv run agentm lint src/ contrib/` reported no errors and the structural
  warnings listed below.
- `git diff --check` passed.

## Structural warnings to monitor

These are not promoted to behavioral findings yet, but they indicate growing concentration of responsibility:

| Path or symbol | Warning |
| --- | --- |
| `_PolicyEngineRuntime` | 44 methods; threshold is 25 |
| `PolicyPersistence` | 28 methods; threshold is 25 |
| `contrib/extensions/policy/src/policy_engine/trace_view.py` | 2,629 lines; threshold is 1,500 |
| `src/agentm/code_health.py` | 1,729 lines; threshold is 1,500 |
| `src/agentm/core/runtime/reaction.py` | 1,577 lines; threshold is 1,500 |

If one of these areas must change to resolve a finding, prefer extracting a cohesive responsibility rather than adding another conditional path to the existing large unit.
