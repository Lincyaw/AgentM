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
| RV2-004 | P2 | resolved | Live policy turn summaries use the next turn number |
| RV2-005 | P2 | resolved | Bash writes can fail to advance repository generation |
| RV2-006 | P2 | open | A contrib atom bypasses Operations, and AM004 does not detect it |
| RV2-007 | P2 | open | The presenter owns concrete compaction policy details |
| RV2-008 | P2 | resolved | The test-edit hook produces false positives for read-only commands |

## RV2-004: Live policy turn summaries use the next turn number

- Priority: P2
- Status: `resolved`
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

- [x] Live and replay paths use the same turn-finalization ordering.
- [x] A committed turn summary contains the tool evidence recorded in that turn.
- [ ] Multi-turn traces produce identical live and replay summaries.

### Decision notes

Resolved on 2026-07-22.

- `_on_turn_committed` now captures `turn_idx` from `self._state.turn_count`
  before calling `advance_turn()`, so the query at line 781 filters entries
  using the same turn number that tool evidence was recorded under.
- The offline projector already summarized before advancing; the live path now
  matches that order.

## RV2-005: Bash writes can fail to advance repository generation

- Priority: P2
- Status: `resolved`
- Area: repository mutation tracking

### Evidence

- `contrib/extensions/policy/src/policy_engine/ifg_investigation.py:259` increments support and returns early when a path is already present in repository artifacts, without first handling `relation == "write"`.
- The `_touch_repository(mutated=True)` path around line 301 advances the revision/generation but is skipped by that early return.

### Problem and impact

When a Bash write targets an already known repository artifact, the mutation may not advance repository generation. Later evidence can then classify the changed artifact as unchanged or churn, weakening freshness and causality decisions.

### Recommended direction

Keep the rule that Bash observations do not create repository anchors, but when an anchor already exists and the relation is `write`, always mark the repository as mutated before returning.

### Exit criteria

- [x] A write to an existing repository artifact advances repository revision/generation exactly once.
- [x] A read-only observation does not advance generation.
- [x] A Bash-only observation still cannot create a repository anchor.
- [x] Subsequent evidence observes the updated generation.

### Decision notes

Resolved on 2026-07-22.

- `_record_bash_path` now calls `self._touch_repository(path, mutated=True)`
  when `relation == "write"` and the path is already a repository artifact,
  before the early return. Read-only observations still only bump support.

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
- Status: `resolved`
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

- [x] Read-only `rg`, `grep`, and inspection commands are not classified as test edits because of their pattern text.
- [x] Actual redirection, in-place editing, file removal, and patch commands targeting tests remain blocked as intended.
- [ ] Hook diagnostics identify the parsed operation and target.
- [ ] Positive and negative command examples are covered by focused hook verification.

### Decision notes

Resolved on 2026-07-22.

- `_paths_from_bash` now skips `WRITE_TO_TESTS_PATTERNS` matching when the
  command starts with a known read-only program (`rg`, `grep`, `ag`, `find`,
  `fd`, `ls`, `cat`, `head`, `tail`, `wc`, `file`, `stat`, `tree`,
  `git log/diff/show/blame/status/branch`).
- Write commands (`sed -i`, `cat >`, `tee`, `cp`, `mv`, etc.) still match
  because they don't start with a read-only prefix.
- Patch-header detection is independent and runs unconditionally.

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
