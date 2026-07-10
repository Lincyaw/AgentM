# Reliability Substrate

**Status**: accepted (design discussions 2026-07-10); phase 1 implementation in progress
**Source**: `knowledge/reliable_multi_agent_system_modeling.md` (the EAC reference
model) deconstructed against AgentM's actual architecture. This doc records
which parts of that model become SDK mechanism, which become skills, and how
the mechanism parts land on existing atoms.

## 1. Problem and position

In a naive multi-agent system, **every message pass is simultaneously an
unaudited trust promotion and an unaudited scope promotion**: a child
session's output gets spliced into the parent's context (it becomes a premise
for downstream reasoning) and thereby gains the power to influence every
subsequent parent action — at zero cost, with no record. Summarization is
worse: it strips provenance before re-injecting. Error cascade is therefore
the *default* path in MAS plumbing, not an exception path.

The EAC model's central retreat is the honest one: a substrate cannot
guarantee "answers are true" (a semantic property requiring oracles that
often don't exist), but it *can* guarantee "errors do not silently gain
power" — a safety property achievable with pure mechanism. Concretely, the
substrate promises that errors are **locatable** (complete traces),
**boundable** (blast radius is computable), and **recoverable** (invalidate,
recompute the affected subgraph only, or compensate). Making errors *rarer*
is the job of skills and scenario design, not of the substrate.

### Adoption criterion

One rule decides what goes where, and it coincides with AgentM's existing
SDK-vs-scenario boundary:

> **Deterministic and free of self-reference → mechanism (core/atoms).
> Requires semantic judgment → skill (best practices for the judge).**

Idempotency keys, budgets, append-only logs, content-addressed journal keys,
invalidation flags: deterministic, mechanism. Task decomposition, claim
schemas, verification-strategy selection, deciding *which* node introduced
an error: LLM or human judgment, skill. An "admission gate" whose judge is
an LLM is just another agent whose errors correlate with the producer's —
enforcement built on it manufactures pseudo-safety. So semantic gates
**record**, they do not **block** (see §7 Rejected).

## 2. Scope

Two mechanism planes plus the skills layer:

1. **Session index (traceability)** — a near-real-time index over every
   agent session's event stream: lineage edges, taint states, groundedness.
   Split into a deterministic layer (code-derived, always sound) and a
   refined layer (LLM-local judgments, best-effort). §4.2.
2. **Fault tolerance (control plane)** — bounded execution, attempt
   identity, a small set of deterministic recovery primitives (invalidate /
   forced-miss resume / fencing / action gateway), and a recovery loop
   driven by agents on top of those primitives. §4.1, §4.3.
3. **Skills** — everything requiring semantic judgment: claim contracts,
   verification ladders, topology, culprit identification during recovery.
   §6.

**Layering discipline (load-bearing):** fault tolerance may consume only
the session index's *deterministic* layer. The refined layer detects and
advises; it is never on the recovery-correctness path. A recovery mechanism
that depends on a best-effort LLM component would itself need recovering —
the self-reference this substrate exists to avoid.

## 3. Current state inventory

| Capability | Where it lives today | Gap |
|---|---|---|
| Bounded execution | `loop_budget`, `cost_budget`, `agent_loop` TerminationCause (final causes), workflow 1000-agent backstop | none — done |
| Trace completeness | `single_event_log`: one OTLP JSONL per session, every signal through the bus, `dispatch_id` join keys, fingerprint binding | none structural; detectors underexploit it |
| Artifact lineage | `artifact_store`: append-only, `ProvenanceRef`, `parent_artifact_ids` (revision chain); `artifact_read` calls are traced | consumption edges exist latently in the trace but are not derived or queryable; no invalidation operation |
| Workflow resume | journal keyed by `sha256(prompt, opts)` — since upstream results are interpolated into dependent prompts, the keys *already* content-address the dependency graph | journal stores result but not prompt (backward edges need a trace join); no invalidation operation, so the latent partial-recompute semantics cannot be triggered |
| Sub-agent delivery | `sub_agent` posts `<subagent_result>` to parent SessionInbox | no attempt identity: a superseded/re-dispatched child's late result is indistinguishable from a current one |
| Side effects | `permission` (approval), `tool_bash_guard` | no idempotency, no confirm-after-execute, no compensation record; a retried turn can re-send a message/re-run a mutation |
| Detectors | `trajectory_index` (contrib): grounding taint analysis, LLM-local / code-global | exists as a one-off; not yet positioned as the session-index embryo |
| Terminal vocabulary | TerminationCause covers *loop* end | no task-level outcome vocabulary; a worker cannot honestly return "unresolved" — everything collapses into completed-with-text |

## 4. Design

### 4.1 Control plane

**C1 — Attempt fencing (implemented).** Whether two dispatches are "the
same logical work retried" is semantic judgment, so the mechanism does not
guess: the dispatcher declares it — `dispatch_agent(..., supersedes=
<old_task_id>)`. A validated declaration marks the old attempt
`superseded_by=<new_task_id>` (only after the replacing dispatch is
definitely live, so a failed spawn never falsely fences); when a superseded
attempt finalizes later, its `<subagent_result>` is delivered flagged
`stale="true" superseded_by=...` with an explicit warning — never as
current. Scope note: the workflow `agent()` path needs no fencing — it
awaits its child inline (timeout cancels the coroutine), so there is no
late-delivery channel into a shared inbox; the fencing surface is
`sub_agent`'s async inbox path only. Full lease/epoch machinery for
distributed workers remains rejected (§7). Fencing matters most *during
recovery*: without it, retry itself manufactures new contamination.

**C2 — Typed task outcomes.** Child results carry
`outcome ∈ {completed, unresolved, rejected, failed, aborted}` plus free-text
reason. `unresolved` is first-class and honest: mechanism transports it
without coercing it into an answer; whether a worker *should* return it is
prompt/skill policy. If the transport layer has no slot for "unresolved", no
prompt policy can preserve it. Lands on the existing `<subagent_result>`
block and the workflow `agent()` return path.

**C3 — Classified retry.** Retry decisions key on outcome class:
transport/timeout failures may retry identically (bounded, jittered);
semantic failures must vary something — model, prompt, decomposition — or
escalate. Retrying an identical semantic failure with an identical context
is re-sampling the same failure mode. C3 couples with invalidation (§4.3):
an invalidation's `feedback` is precisely the "vary something" for the
forced re-run — without it, re-running an invalidated node may reproduce
the same wrong text verbatim, downstream keys never shift, and the
invalidation is a no-op.

**C4 — Action gateway.** A builtin atom `action_gateway` through which
tools *declared irreversible* (per-tool opt-in) must route: it records an
ActionIntent (operation, target, params, idempotency_key), composes with
the existing `permission` approval flow, executes, then records
confirmation (read-back where the tool supports it) and an optional
compensation reference. Guarantee: the same idempotency_key never executes
twice; every external side effect has an intent → outcome pair in the
trace. In the recovery loop, intent records answer "which committed side
effects consumed an invalidated input" — the one class of damage that
invalidate-and-recompute cannot fix.

### 4.2 Session index (traceability)

One atom family owns all derived views over session event streams — the
generalization of `trajectory_index` from an offline analysis into a
near-real-time session index. Near-real-time is what upgrades fault
tolerance from post-mortem to in-flight containment: a taint flagged while
the workflow is still running can stop dependents *before* they consume it
(the EAC Node Risk formula multiplies by detection delay; the index
compresses that factor).

Two layers, following the split already present in `trajectory_index`'s
SCHEMA ("a model failure leaves the deterministic layer intact; the derived
layer is rebuilt wholesale"):

**Deterministic layer** — code-only, incremental, always sound. Holds the
three lineage channels:

- *Workflow channel: already content-addressed, by construction.* The
  journal keys every `agent()` call by `sha256(prompt, opts)`
  (`_workflow/sdk.py`), and downstream calls interpolate upstream results
  into their prompts — an upstream result is *literally part of* every
  dependent call's key. Backward edges are derived without LLM involvement:
  entry X's result body appearing verbatim in entry Y's prompt ⇒ X → Y.
  (Requires the journal to store the prompt alongside the result, or a
  trace join; storing it is cheaper.) When the script transforms a result
  before interpolating (e.g. extracts one JSON field), verbatim matching
  misses the edge and the channel degrades to conservative program-order
  edges — precision lost, completeness kept.
- *Artifact channel: observed, not declared.* `artifact_read` is a traced
  tool call. Conservative consumption edge: session S read artifact A and
  subsequently wrote artifact B (or produced its journaled result) ⇒ A → B.
  An optional `input_refs` declaration on `artifact_write` may *narrow* the
  derived set; mechanism checks declared ⊆ traced-read-set. Self-report is
  never load-bearing.
- *Context channel: conservative order taint.* Once text enters a session's
  context, every later output of that session potentially depends on it;
  the closure is "everything after turn t" (message order and identity are
  already in the event log — nothing new recorded).

**Refined layer** — LLM-local judgments propagated by code: entity
extraction, alias/coreference resolution, grounding and value-fidelity
(`trajectory_index`'s five risk states: grounded / premature / ungrounded /
contradicted / stale — the EAC epistemic states discovered post-hoc).
Asynchronous, best-effort, rebuildable wholesale. Its two jobs: **detect**
(contradicted/ungrounded findings are invalidation candidates) and
**advise** (narrow the deterministic layer's conservative closures for a
human or recovery agent). Never consumed by recovery primitives (§2
layering discipline).

Query surface: the index answers `lineage(node)` (backward/forward edges)
and `contaminated(target)` (deterministic conservative closure). If the
index is unavailable, recovery degrades to the maximal conservative closure
— more expensive, never wrong.

### 4.3 Fault tolerance: primitives and the recovery loop

Four deterministic primitives (mechanism), one agent-driven loop (policy).
No new subsystem: the loop's driver is an agent — the parent session, a
recovery scenario, or a human — with `monitor` providing periodic triggers
where wanted.

**Primitives:**

1. **`invalidate(target, reason, feedback?, carry_previous?)`** — append a
   flag record against a journal entry or artifact (append-only; never
   mutates the original). Carries optional `feedback` (why it was wrong /
   what to do differently) and `carry_previous` (whether the forced re-run
   should see a compacted record of the previous attempt — decoupled
   option, see below).
2. **Forced-miss resume.** Journal lookup treats a key as a miss when its
   newest invalidation record is newer than its newest result record. The
   re-run executes with `feedback` (and, if `carry_previous`, a compact
   summary of the previous attempt) injected into the child's prompt, but
   **records its result under the original key** — the key is the node's
   identity as authored by the script; injection is execution detail.
   Propagation needs no computation: the new result changes dependent
   prompts, their keys shift, they re-run; where the re-run output (or the
   fragment the script actually extracted) is unchanged, dependents still
   hit — correctly, because no contamination flowed (build-system early
   cutoff, at dataflow precision, no script-API change).
3. **Attempt fencing** (C1) — protects the loop itself from late results of
   superseded attempts.
4. **Action-gateway intents** (C4) — locate side effects that consumed
   invalidated inputs; drive compensation.

**The recovery loop:**

```
detect      → human / refined-layer detector / downstream failure /
              action-confirm failure
scope       → deterministic closure from the session index
              (refined layer advises narrowing; judgment decides)
locate      → walk the lineage path upstream, at each hop asking
              "inherited or introduced?" (semantic judgment; the
              `contradicted` detector automates the tool-grounded part)
invalidate  → primitive 1, with feedback
re-drive    → workflow: resume (primitive 2 cascades exactly the
              affected subgraph)
              live session: invalidation notice into its inbox
              finished sub-agent: re-dispatch (fencing protects)
compensate  → walk action-gateway intents; compensate or escalate
```

**Two execution models, two strengths of guarantee.** Workflow nodes are
functional (prompt in, result out), so recovery is mechanical replay with
exact subgraph precision. Session context is not replayable — a live
context cannot surgically un-know a sentence; mechanism delivers the
invalidation notice and fences stale attempts, and repair runs through the
agent's own reasoning (or a fork from a pre-contamination point). This
asymmetry is a fact about the execution models, not a design gap — and it
is why the `mas-composition` skill teaches: flows that may need recomputing
belong in workflow orchestration, not scattered across conversation.

**Previous-attempt compaction (decoupled option).** When a forced re-run
injects feedback, the caller chooses what the new attempt sees of the old
one: nothing (fresh solve, anchoring-free), the previous result verbatim,
or a compacted summary of the previous attempt's transcript. This is a
knob on the invalidation record, independent of the invalidation mechanism
itself; the compaction (when chosen) reuses the existing compaction engine
rather than growing a new summarizer.

### 4.4 What stays out of mechanism

No claims/evidence/assumptions schema in core or in `artifact_store`'s
required fields. Claim structure is defined **at contract time** by the task
issuer (scenario prompt / workflow script / persona), because deciding what
the checkable units of a task are is semantic judgment. The substrate only
guarantees that whatever structure the contract demanded is stored,
versioned, linked, and traceable.

## 5. Invariants

Fail-stop positions. The first three are **user-authorized for tests**
(2026-07-10); the rest need per-item confirmation before a test is written:

- [authorized] A flagged journal entry is never served as a resume hit; a
  re-run result newer than the flag supersedes it.
- [authorized] An invalidation's feedback reaches the re-run child's prompt,
  and the result is recorded under the original key (addressability
  preserved).
- [authorized] Backward-edge derivation agrees with the script's real
  dataflow under standard interpolation patterns.
- [authorized] A superseded task's late result is never delivered as
  current (C1 fencing).
- The same action idempotency_key never executes twice (C4).
- Invalidation closure is complete: every session that read a flagged
  artifact has its subsequent writes and journaled results flagged.
- A declared `input_refs` set that is not a subset of the session's traced
  read-set is rejected.
- The artifact store remains append-only under invalidation.
- Every declared-irreversible tool call has an intent record preceding it
  and an outcome record following it (C4).

## 6. Skills layer

The semantic best practices — everything the EAC document says that is *not*
deterministic — become one skill (working name `mas-composition`), pointing
at the knowledge doc for depth:

- Contract-time claim schemas: the task contract states which checkable
  assertions the worker must return; never extract claims post-hoc.
- Verification-strategy ladder: classify verifiability first (deterministic
  oracle → checkable witness → external evidence → judgment → unverifiable),
  then choose validation; an LLM judge is the *last* resort, not the default.
- Correlated errors: agreement among same-model agents is not evidence;
  independence requires different failure domains (model/tool/source).
- Context isolation only at high fan-out aggregation nodes; elsewhere let
  narrative context flow (over-isolation trades contamination for
  compositional failure).
- `unresolved` discipline: when evidence is insufficient, return it — the
  substrate transports it honestly (C2).
- Blast-radius design: decompose along verifiable interface boundaries;
  spend verification budget on aggregators; put recompute-worthy flows in
  workflow orchestration (§4.3).
- Recovery practice: inherited-vs-introduced bisection along lineage paths;
  always attach feedback to invalidations (C3 coupling).

## 7. Rejected / deferred

- **Blocking admission gates with LLM judges** — correlated with producers;
  record admission rationale in trace instead.
- **Recovery consuming the index's refined layer** — layering discipline
  (§2): best-effort components never on the recovery-correctness path.
- **Producer-declared `input_refs` as the load-bearing edge source** — its
  only possible author is the producing LLM; self-reported dependencies are
  the unreliable extraction this substrate refuses to build on. Kept only
  as an optional narrowing declaration checked against traced reads.
- **Lease/epoch for distributed workers** — single-process gateway; attempt
  fencing (C1) covers the real failure. Revisit if execution crosses
  process boundaries.
- **Consensus/BFT** — protects control-plane agreement, which a single
  process gets for free; cannot make committed content true.
- **EAC three-graph subsystems** — graphs are index-computed views over the
  event log, never maintained stores.
- **Trust calibration/amortization** (pricing trust from historical
  per-atom/per-task error rates) — promising; belongs to
  `evolution_substrate`'s observation data; deferred until detector metrics
  exist to calibrate against.

## 8. Measurement and phasing

Observe first, enforce later. Core metrics, computable from the event log
plus the new invalidation/intent records:

- **False Commit Rate** — child results adopted by a parent and later
  invalidated / total adopted. North-star; go/no-go for later phases.
- **Invalidation completeness** — flagged descendants / reachable descendants.
- **Ungrounded-use rate and taint depth** — from the refined layer.
- **Duplicate side-effect rate** — from action-gateway intent records.

Phases (atom-level changes; no core ABI additions expected):

1. **Workflow-channel fault tolerance** *(shipped)*: journal stores
   prompt; `invalidate` primitive with feedback + carry_previous;
   forced-miss resume with original-key recording; backward-edge derivation
   query. Three authorized fail-stop tests.
2. **Control-plane hardening**: attempt fencing via explicit `supersedes`
   declaration *(shipped)*; typed outcomes on `<subagent_result>` and
   workflow `agent()` — outcome authority is partitioned: the substrate
   stamps `failed`/`aborted` (from TerminationCause), the consumer stamps
   `rejected`, and only `completed` vs `unresolved` is producer-declared
   (the one distinction with no mechanical source; the slot is
   asymmetric-safe — a false `unresolved` wastes a retry, a false
   `completed` is today's status quo, measured by detectors not prevented).
   `submit_result` gains an `outcome` enum; `outcome=unresolved` may skip
   payload schema validation so the schema cannot coerce fabrication.
3. **Artifact channel + side effects**: derived consumption edges exposed
   via `agentm trace`; artifact invalidation + closure; `action_gateway`.
4. **Index maturation + skills**: near-real-time refined layer;
   `mas-composition` skill; metrics through `agentm trace`.

## 9. Open questions

- **Filesystem side channel.** A workflow child writes a file via bash;
  another child reads it — invisible to both the journal keys and the
  artifact channel. Current stance: accept and document; the
  `mas-composition` skill steers cross-child data through return values or
  artifacts. Deriving coarse edges from traced bash/file tool calls
  (same-path read-after-write) is a possible phase-4 refinement.
- **Conservative closure cost.** "S read A ⇒ everything S later wrote is
  contaminated" may over-flag badly in practice. If phase-3 observation
  shows high false-flag rates, the optional `input_refs` narrowing gets
  promoted from optional to expected.
- **Journal growth.** Invalidation records and superseding results
  accumulate per key; needs a retention story eventually (time-based, like
  artifact retention).
