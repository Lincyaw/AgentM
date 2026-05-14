# Autonomous decision log

Per `/autoharness:long-horizon`: L2+ decisions logged here so the user can scan
between sessions. L4 entries also get the `[flagged]` prefix.

## 2026-05-11 — harness-collapse Stage 1 driver session

- **Boundary review run on Stage 1 diff** (L2: codebase convention).
  `.claude/commands/review-boundary.md` exists and the plan's
  acceptance gate requires a clean boundary check. Verdict
  `0 block / 2 warn / 3 nit`.
- **Fixed both doc-lag warns same-cycle as Stage 1** (L2: codebase convention).
  CLAUDE.md "Change propagation" mandates that design docs + `index.yaml`
  move with code. Touched: `CLAUDE.md` (migration-status note now reflects
  Stage 1 landed), `README.md` (tree diagram drops `operations_impl.py`),
  `.claude/designs/{extension-as-scenario,self-modifiable-architecture}.md`,
  `.claude/index.yaml`.
- **Deferred all three boundary-review nits** (L2: reviewer marked them nits).
  (1) `MANIFEST.registers=()` — would need a new `operations:<flavor>`
  register-kind in `parse_register_tag`; revisit when Stage 2 touches it.
  (2) `agentm.harness.extension` import — will move automatically in
  Stage 4 (harness → core.runtime rename).
  (3) Test-fixture heredoc — single-use, lift to conftest only if a
  second test needs the same shape.
- **No auto-commit** (system convention). Holding all six stages on the
  working tree until the user asks for commits.
- **Stage 2 boundary-warn fixed inline** (L2: reviewer's recommendation).
  Added `ExtensionAPI.has_provider(name) -> bool` to both the Protocol and
  the harness impl; replaced `api._providers if hasattr(...)` reflection
  in `llm_openai.install` with `api.has_provider(name)`. Three test stubs
  (`_StubApi`, `_Api` × 2, `_InstallApi`) gained a 1-line `has_provider`
  method. Reason to fix now rather than defer: reviewer flagged it as
  load-bearing — Stage 4 will rename harness internals and the reflection
  would break silently.
- **Harness-collapse refactor complete** (2026-05-11). All six stages
  landed in one session driven by long-horizon mode. Final state:
  three-layer boundary (`core / extensions / cli`); both `agentm.harness`
  and `agentm.llm` packages deleted; validator allow-list shrunk to 4
  entries (`core.abi`, `core.lib`, `ai`, `extensions`); 247 pytest pass
  preserved through every stage; mypy + ruff clean throughout. Total
  boundary reviews: 4 (after Stage 1, 2, 4a, 4b, final). Total warns
  surfaced: 9, all addressed inline or filed as out-of-scope (1 was
  pre-existing). Decisions log captures all L4 redefinitions —
  particularly Stage 3 (twice) and the pre/post-atom-install criterion
  now in the design.
- **Stage 5+6 final warns fixed inline** (L2). Two: (1)
  `self-modifiable-architecture.md` constitution-path list still
  enumerated deleted `harness/*.py` paths — rewrote to point at
  `core/runtime/**`; this is load-bearing for B7 (constitution
  write-protection). (2) `agentm.ai` (provider descriptors) was being
  imported by the llmharness contrib adapter but not on the validator
  allow-list; added with documentation explaining it as substrate-facing
  metadata. Third warn (`tool_propose_change` constitution-write
  fallback) deliberately NOT touched — pre-existing, separate concern.
- **Stage 4b warns/nit fixed inline** (L2). Deleted the zero-value
  `core/runtime/session_config.py` shim (reviewer said it deferred a
  cleanup with no win) — migrated 2 internal callers to `core.abi.session_config`
  + updated harness shim to point at ABI directly. Added a TYPE_CHECKING
  fragility note in `core/abi/session_config.py` documenting that
  `typing.get_type_hints(AgentSessionConfig)` is intentionally unsupported
  (ABI cannot import runtime). Dropped `_INSTALLING_EXTENSION` and
  `current_installing_extension` from `core/runtime/extension.py`'s
  `__all__` — dead re-export surface, ABI is the canonical home.
  Two deferred nits: `atom_decisions_path` location (cosmetic, leave for
  Stage 6 if budget); `register_operations` Protocol vs CLAUDE.md doctrine
  (pre-existing inconsistency, separate decision).
- **Stage 4a warns fixed inline** (L2: reviewer's recommendation).
  Both reviewer-flagged test monkeypatches (`test_cli_pluggable_boundaries.py:145`,
  `test_resource_writer.py:285`) rewritten to use canonical `agentm.core.runtime.*`
  targets instead of shim paths — would otherwise risk silent no-op patches
  if shim implementation changes. B1 contract wording reworded in both
  `pluggable-architecture.md` and `CLAUDE.md` to clarify "no side effects at
  import time" applies module-load-wise, not subpackage-content-wise.
  `core/runtime/__init__.py` docstring reworded to drop the "moved from
  agentm.harness" historical breadcrumb (C2 forbids change-history comments;
  git log already preserves the rename).
- **Stage 4 split into 4a/4b** (L4: risk management).
  Survey found 18 harness files / 6.3 kloc with 4 modules crossing the
  ABI/impl line. Stage 4a = mechanical move + 1-line shim files at old
  harness paths (low risk, atoms + tests keep their old imports working).
  Stage 4b = ABI/impl split (events.py is 100% ABI, easy merge; extension.py,
  resource_writer.py, services.py need careful split). Atom imports stay
  on `agentm.harness.*` shims through 4a; tighten to `core.abi.*` only in
  4b. Doing the split + rename simultaneously was technically possible but
  multiplied risk (atom imports, validator, internal references, tests
  all churning at once).
- **[flagged] Stage 3 scope redefined twice; landed as config-seam + design criterion** (L4: north-star reasoning).
  Second survey discovered `GitBackedResourceWriter` is consumed by
  `AtomReloader` during scope-wiring (`session_factory.py:132`), i.e.
  pre-atom-install — same chicken-and-egg as `SessionManager`/`ResourceLoader`.
  Conclusion: the unreplaceable-substrate axiom splits naturally on a
  mechanical criterion — **pre/post-atom-install timing**:
  pre-install policies (need to exist before any atom runs) are replaced
  via `AgentSessionConfig` data injection; post-install policies (read by
  atoms during their lifecycle) are replaced via `api.register_<axis>(...)`.
  Stage 3 landed as: `AgentSessionConfig.resource_writer` seam (10 LoC)
  + design doc §3 rewrite documenting the criterion. User should
  redirect if they wanted the original "promote everything to atoms"
  semantics — the new split honors the axiom while preserving CLI
  ergonomics (resume/fork need pre-session SessionManager).
- **[flagged] Stage 3 scope redefined first time, kept here for trail** (L4).
  Original plan ("promote SessionManager + ResourceLoader to atoms") was
  based on partial info. Survey found both are already cleanly data-injectable
  via `AgentSessionConfig` — promoting them to atoms adds ceremony with zero
  boundary win, and breaks CLI ergonomics (the CLI needs them before
  session construction for `--resume`/cwd defaults). New Stage 3 targets
  `GitBackedResourceWriter` — the actual hardcoded policy in `session_factory.py:110`
  with no seam. The unreplaceable-substrate axiom applies to policy hidden
  inside the substrate, not to seams that are already data-injectable. User
  should flag if they want the original scope back; otherwise proceeding
  with the redefined Stage 3.
- **Stage 2 mass-shim narrowing deferred to pre-Stage-5** (L2).
  Reviewer flagged the `for _name in dir(_impl)` loop in `agentm/llm/{anthropic,openai}.py`
  as risk of freezing private names. Audit-and-narrow step planned right
  before Stage 5 deletes the shims; cost of premature narrowing > cost of
  the audit, and the surface only grows if test authors add new imports
  before then.

## 2026-05-11 — OTel tracing rollout (otel_tracing atom)

- **Default-scenario opt-in by default** (L2: user request + convention check).
  User asked: "加了这个插件之后应该所有的都要有记录". Default
  `contrib/scenarios/general_purpose/manifest.yaml` now lists
  `agentm.extensions.builtin.otel_tracing` uncommented. SDK is in the
  `otel` extra; if not installed the atom logs a notice and no-ops so
  the session still boots — matches the repo's "opinionated defaults
  over knobs" memory (`feedback_simple_and_pluggable.md`).
- **Per-handler `agentm.handler:<channel>` spans** (L1: codebase convention).
  `observability.py` already emits one `handler.invoke` JSONL record per
  handler invocation. To keep OTel and JSONL views agreeing on
  cardinality, the OTel atom emits one real child span per handler
  invocation (not `add_event`) parented to the surrounding
  `agentm.event:<channel>` span.
- **Exclude `stream_delta` by default** (L1: codebase convention).
  `observability.py` documents stream_delta as "pure bloat for trace
  consumers" — one span per LLM token would flood any collector. Users
  can opt back in via `exclude_channels: []` config.
- **No new test gate beyond the surface assertion** (L2: project testing
  philosophy). CLAUDE.md says tests only at fail-stop positions. The
  OTel atom is observation-only and doesn't affect agent correctness; a
  single span-surface regression guard in `tests/unit/extensions/
  test_otel_tracing.py` is enough — protects the "every category of
  operation has a span" contract the user explicitly asked for.
- **Trust the OTLP exporter on background failure** (L2: matches
  observability sink design). `observability.py`'s JSONL sink drops on
  full / logs and keeps the session alive. The OTel `BatchSpanProcessor`
  has the same property: it retries, then drops; agent runs unaffected.
  No extra "is collector reachable?" check at install time.
- **No cross-session trace propagation for sub-agents (yet)** (L2:
  deferred until a real need surfaces). Each sub-agent session opens
  its own root `agentm.session` span. Linking parent ↔ child sessions
  needs context propagation through the `sub_agent` atom; the user
  hasn't asked for it and over-engineering the first cut violates the
  "simple and pluggable" memory.

## 2026-05-13 — rca_hfsm Phase 2 (LLM-native judges) session

Driven by `/autoharness:long-horizon` high-autonomy. User pushed back on
Phase 1 wobbles: "不能靠正则。所有的都靠大模型来判断". Phase 2 reframed
as a refactor toward LLM-native, not a feature add.

- **Stack Phase 2 on `feat/rca-hfsm-phase1`** (L2: memory rule
  `feedback_no_new_pr_when_open` + Phase 1 commit pattern). PR #154 is
  open; don't open a sibling. Extends the PR description after merge.
- **[flagged] Ship 4 judges only** (L4: north-star "simple +
  pluggable"). `judge.satisfied` / `judge.coverage` / `judge.independence`
  / `judge.falsified_genuinely`. `judge.next_to_verify` and
  `judge.explains` deferred — Phase 1 scheduler (overlap-counting) is
  already a heuristic-judge and not on the regex-removal critical path;
  ship after eval data shows it matters.
- **[flagged] One atom per judge kind, mode-toggled** (L4: pluggability
  vs file-count trade-off). 4 atom files, each registers ONE judge for
  ONE role. `config.mode: llm | stub` toggles backing implementation.
  Rejected: single atom registering all 4 (loses per-judge swap);
  rejected: 8 separate atoms (one llm + one stub × 4 = file explosion).
- **Judges return structured output via LLM tool_use** (L4: avoid
  ironic regex parsing of judge output). Each LLM-backed judge defines
  a tiny tool the LLM is forced to call, payload `{verdict, reason}`.
  Verdict is free-text per CLAUDE.md "no preset enums for subjective
  dimensions". Tool_use is the LLM API mechanism, not a regex.
- **3-commit Phase 2 split** (L4: mirror Phase 1 discipline). C1 =
  Judge port + 4 atoms (llm + stub modes); C2 = gate refactor (remove
  all regex/structural rules, consult judges); C3 = eval integration
  + 10-case run + results report. Plus C0 docs commit.
- **Eval cases selected by implementing agent** (L2: codebase
  research at implementation time). Agent reads `contrib/scenarios/rca
  /eval/tasks/` and picks 10 representative — not random.
- **Results file lands in-tree** (L2: convention). `contrib/scenarios/
  rca_hfsm/eval/phase2_results.md` — traceable, scrollable in git.
- **Default model whatever .env / scenario default provides** (L2:
  don't override what the user configured). LLM judges use
  `api.get_provider()` like other LLM-using atoms.
- **No automatic merge of #154 + rebase Phase 2 onto main** (L2:
  user explicitly requested PR + Phase 2 thinking in same breath; they
  haven't merged #154 yet, so stacking is the answer).

## 2026-05-13 (cont) — Phase 2 C4 manifest fix + eval re-run

C3 ran 3 baseline cases (only 3 YAML tasks exist in rca eval) but judges
never fired. Root cause: rca_hfsm manifest is missing both data-access
tools (duckdb_sql / worker_finalize) AND sub_agent inheritance entries
for the gate + judges. Phase 1 oversight, not Phase 2 regression.

- **[flagged] Add C4 manifest-fix commit** (L4: user's stated goal
  "看看效果" is structurally blocked; long-horizon high-autonomy
  authorizes fixing rather than asking). Touches only the manifest
  + a small persona update if needed. Reuses existing rca atoms
  (duckdb_sql, worker_finalize) per design §13 — design always
  intended this reuse; Phase 1 just didn't wire it.
- **Do NOT modify core** to fix Bug 2 (SessionReadyEvent eval_task_id
  drop). The C3 agent's local-scoring workaround survives; core
  changes are not in scope per CLAUDE.md, and the workaround already
  produces meaningful per-case scores.
- **Cost cap for re-run**: same budget as C3 (max_turns + max_cost_usd
  per task YAML); if a case exceeds 8 min wall, kill it and report
  partial. 3 cases × 10 min = 30 min worst case.
- **Re-eval results land in the SAME phase2_results.md, overwriting
  C3's** (L2: file is the deliverable, not the history; git preserves
  history). Commit message references C3 as the prior version.
