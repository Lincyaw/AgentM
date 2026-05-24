# HarnessRunner

> Single driver for the llmharness cognitive-audit pipeline. Collapses the
> three duplicated "fold the op log, window the trajectory, compose the
> payload, invoke the child, persist the result" implementations
> (`_drain_extractor`, `replay_extractor_record`,
> `run_offline_auditor_over_control`) into one runner parametrised by two
> seams: how to invoke the child session, and where to persist the ops.

Related: [llmharness-cognitive-audit](llmharness-cognitive-audit.md),
[audit-check-registry](llmharness-cognitive-audit.md#audit-check-registry),
[pluggable-architecture](pluggable-architecture.md),
[sub-agent-lifecycle](sub-agent-lifecycle.md).

## 1. Problem

The 2026-05-22 event-source refactor turned the cumulative audit graph
into a log of `GraphOp` entries (`AUDIT_GRAPH_OP` SessionEntries) folded
on demand by `_scan_branch`. That decoupling made offline replay of
**individual** firings trivial, but left three separate implementations
of "drive the audit pipeline over a trajectory":

| # | Where | Reads ops from | Invokes child via | Persists ops to |
|---|---|---|---|---|
| 1 | `adapters/agentm.py::_drain_extractor` + `_drain_auditor` | `api.session.get_branch()` (live session log) | `api.spawn_child_session` | `api.session.append_entry` |
| 2 | `replay/runner.py::replay_extractor_record` / `replay_auditor_record` | a single recorded `ReplayRecord` | `tools.engine.run_phase_standalone` (top-level session) | nothing |
| 3 | `replay/chained_fork.py::run_chained_fork_experiment` (was `replay/strict_ab.py::run_offline_auditor_over_control` pre-2026-05-24) | re-runs the offline pipeline over each segment's trajectory | `StandaloneChildRunner` (via `replay_pipeline_over_trajectory`) | `InMemorySink` |

The duplication has three concrete costs:

1. **Windowing logic** (compute `[cursor+1, len(messages)-1]`, collect
   turn_texts, enrich `recent_graph` with `source_turn_texts`, compute
   `next_event_id`, build the `payload`) is ~80 lines inside
   `_drain_extractor` and **does not exist anywhere else**. Single-firing
   replay sidesteps it by accepting a pre-built payload from the
   recorded `ReplayRecord`; offline-auditor sidesteps it by reading
   already-recorded extractor outputs. No path can ingest a bare
   `list[AgentMessage]` and produce extractor firings.
2. **Cumulative-state threading** is implicit in path #1 (re-read the
   session log each firing) and **explicitly disclaimed** in path #2
   (`chain_replay`'s docstring: "Chain does **not** re-thread fresh
   extractor outputs back into the auditor's input graph"). The
   disclaimer is a function of the missing abstraction, not a design
   intent — event-sourcing makes threading natural.
3. **Sidecar role is conflated**. A `ReplayRecord` carries
   `compose_kwargs` (system prompt, cards config) AND `payload`
   (`recent_graph`, `next_event_id`) AND `output`. The first two are
   reconstructable from the trajectory; only `output` is genuinely a
   diagnostic snapshot. Path #2 treats the record as a *seed* (read
   `compose_kwargs` + `payload` to rebuild state); path #3 treats it as
   *output to fold over*. The same struct plays two roles, and the
   "circular" failure mode of "to offline-replay you need a sidecar; to
   produce a sidecar you need a live run" falls out directly.

Downstream pain: the (now-removed) `baseline_fork` intervention in
`agentm_rca/eval/agent.py` was forced to use
`rca:harness.sync.extractor5` (extractor online, auditor off) as its
control variant — there was no way to start from a pure `rca:baseline`
trajectory and add extractor + auditor retroactively, because no code
path turned bare messages into extractor firings. The current
`chained_fork` intervention solves this directly: it runs every
segment under the same scenario and synthesises extractor + auditor
records offline against each segment's captured `final_messages`.

## 2. Decision

Introduce a single driver, `HarnessRunner`, that owns:

- **Cadence**: when extractor fires, when auditor fires
- **Windowing**: which trajectory slice each extractor firing sees
- **Cumulative state**: the fold of all ops emitted so far
- **Payload composition**: `recent_graph`, `next_event_id`, `turn_texts`,
  enriched `source_turn_texts`, trajectory snapshot
- **Sidecar emission**: `ReplayRecord` writes

Parametrise it by two Protocols and one in-memory state holder:

```python
# agentm/contrib/extensions/llmharness/src/llmharness/audit/_runner.py

@dataclass
class CumulativeAuditState:
    """Event-sourced graph state + auditor side-channel state.

    Single source of truth for the cumulative view; replaces both the
    live `_scan_branch` re-read-each-firing pattern and the offline
    fold inside the (deleted) `run_offline_auditor_over_control`. The
    `chained_fork` driver threads the same struct across segment
    boundaries via `seed_cumulative=`.
    """
    ops: list[GraphOp]
    cursor_last_turn_index: int
    recent_verdicts: deque[dict[str, Any]]
    continuation_notes: list[str]
    firing_id_counter: int

    def graph_view(self) -> tuple[
        tuple[Event, ...], tuple[Edge, ...], tuple[Phase, ...]
    ]:
        """fold_graph(ops) + merge_to_phases. Cached behind ops length."""

    def next_event_id(self) -> int:
        """max(graph.id) + 1, derived from graph_view()."""

    def absorb_extractor_firing(
        self,
        *,
        firing_ops: list[GraphOp],
        firing_cursor: int,
        firing_id: int,
    ) -> None: ...

    def absorb_auditor_verdict(
        self,
        verdict: dict[str, Any],
        *,
        is_silent: bool,
    ) -> None: ...

    @classmethod
    def hydrate_from_session_log(
        cls, branch: list[SessionEntry]
    ) -> "CumulativeAuditState":
        """Live-mode seed: walk session entries, populate ops + verdicts.

        Used by the live adapter on (re)start to recover state across
        process restarts; thereafter the in-memory state is authoritative.
        Legacy ``AUDIT_EVENT`` / ``AUDIT_EDGE`` entries are converted
        through the `op_from_event` / `op_from_edge` helpers in
        :mod:`llmharness.audit.graph_ops` so a mixed pre-/post-
        event-sourcing session folds correctly.
        """

    @classmethod
    def fresh(cls) -> "CumulativeAuditState":
        """Offline-mode seed: empty state."""


class ChildRunner(Protocol):
    """How a single child phase is invoked."""

    async def run_extractor(
        self,
        *,
        extensions: list[tuple[str, dict[str, Any]]],
        provider: tuple[str, dict[str, Any]] | None,
        payload: dict[str, Any],
        turn_window: list[int],
    ) -> ChildPhaseResult: ...

    async def run_auditor(
        self,
        *,
        extensions: list[tuple[str, dict[str, Any]]],
        provider: tuple[str, dict[str, Any]] | None,
        payload: dict[str, Any],
    ) -> ChildPhaseResult: ...


class OpSink(Protocol):
    """Where ops + cursor + verdict entries are persisted.

    Live path appends to the AgentM session log via api.session.append_entry,
    offline path is a no-op (state lives only in CumulativeAuditState +
    optional sidecar).
    """

    def append_op(
        self,
        op: GraphOp,
        *,
        firing_id: int,
        op_index: int,
        turn_window: list[int],
    ) -> None: ...

    def append_cursor(self, *, last_turn_index: int) -> None: ...
    def append_verdict(self, verdict: dict[str, Any]) -> None: ...
    def append_failure(self, entry_type: str, payload: dict[str, Any]) -> None: ...


class HarnessRunner:
    def __init__(
        self,
        *,
        cumulative: CumulativeAuditState,
        child: ChildRunner,
        sink: OpSink,
        sidecar: SidecarWriter | None,
        extractor_settings: ExtractorSettings,
        auditor_settings: AuditorSettings,
        extractor_interval: int,
        audit_interval: int,
        enable_auditor: bool,
        root_session_id: str,
        provider_extractor: tuple[str, dict[str, Any]] | None,
        provider_auditor: tuple[str, dict[str, Any]] | None,
        audit_registry: AuditCheckRegistry | None = None,
        # Only consulted by the single-firing replay seams below;
        # trajectory-driven and live paths leave this at None.
        cwd: str | None = None,
    ): ...

    # Single public entry point — both live and offline call this.
    async def on_trajectory_progress(
        self,
        messages: list[AgentMessage],
        *,
        turn_count: int,
    ) -> StepResult: ...

    # Lower-level trajectory entry points kept public for tests.
    async def fire_extractor_once(
        self,
        messages: list[AgentMessage],
    ) -> _ExtractorFiringResult: ...

    async def fire_auditor_once(
        self,
        messages: list[AgentMessage],
    ) -> _AuditorFiringResult: ...

    # Single-firing replay seams (invariant #2 — degenerate case).
    # Both raise ValueError if cwd was not set at construction.
    async def fire_extractor_from_record(
        self, record: ReplayRecord
    ) -> PhaseResult: ...

    async def fire_auditor_from_record(
        self, record: ReplayRecord
    ) -> PhaseResult: ...
```

`StepResult` carries `{fired_extractor: bool, fired_auditor: bool,
surfaced_reminder: Reminder | None, extractor_record: ReplayRecord |
None, auditor_record: ReplayRecord | None}`. Both record fields are
populated on every firing (even when `sidecar=None`) so offline
drivers — notably `llmharness.replay.chained_fork.write_chained_replay`
— can assemble multi-segment replay sidecars without a disk round-trip
between segments.

`fire_extractor_from_record` / `fire_auditor_from_record` reconstruct
the per-firing `ExtractionState` inline from `record.payload` +
`record.extras.turn_texts` (in lieu of the earlier sketch's
`CumulativeAuditState.hydrate_from_replay_record` helper — that
helper was never landed; the inline reconstruction in the runner
serves the same role and stays scoped to the seam that needs it).

`ExtractorSettings.from_compose_kwargs(compose_kwargs, *,
prompt_override=None)`, `AuditorSettings.from_compose_kwargs(...)`,
and `AuditorSettings.empty()` are the public constructors callers use
to rebuild per-firing settings from a recorded `ReplayRecord` (the
single-firing replay wrappers in `replay/runner.py` and the
`chained_fork` driver both consume them).

### 2.1 Three concrete pluggings

| Path | `cumulative` seed | `child` impl | `sink` impl |
|---|---|---|---|
| **live adapter** | `hydrate_from_session_log(api.session.get_branch())` on `install`; mutated in-place per firing | `LiveChildRunner(api)` → `api.spawn_child_session` | `LiveOpSink(api)` → `api.session.append_entry(AUDIT_GRAPH_OP, ...)` (legacy AUDIT_EVENT / AUDIT_EDGE / AUDIT_PHASE writes stay, for the auditor + viewer) |
| **offline-from-trajectory** | `fresh()` | `StandaloneChildRunner(cwd)` → `run_phase_standalone` | `InMemorySink` (no AgentM session log; the sidecar is the artefact) |
| **offline-from-sidecar single firing** | `fresh()` — `fire_extractor_from_record` / `fire_auditor_from_record` reconstruct `ExtractionState` inline from `record.payload` + `record.extras.turn_texts`; cumulative state is not consulted by these seams | `StandaloneChildRunner(cwd)` | `NoopSink` (defined in `audit/_offline_seams.py`) |

### 2.2 Live adapter shrinkage

`_drain_extractor` and `_drain_auditor` become thin wrappers:

```python
# inside install(api, config) — only one runner per session install
runner = HarnessRunner(
    cumulative=CumulativeAuditState.hydrate_from_session_log(
        api.session.get_branch()
    ),
    child=LiveChildRunner(api),
    sink=LiveOpSink(api),
    sidecar=SidecarWriter(_replay_log_path_for(api)) if _enabled else None,
    extractor_settings=...,
    auditor_settings=...,
    extractor_interval=k_ext,
    audit_interval=k_aud,
    enable_auditor=enable_auditor,
    cwd=api.cwd,
    root_session_id=_audit_session_id(api),
    provider_extractor=...,
    provider_auditor=...,
)

def _on_turn_end(event: TurnEndEvent) -> None:
    nonlocal turn_count
    turn_count += 1
    _ensure_worker()
    queue.put_nowait(_RunnerStepJob(messages=list(event.messages), turn_count=turn_count))

# worker:
async def _drain_step(job):
    await runner.on_trajectory_progress(job.messages, turn_count=job.turn_count)
```

The async queue stays — it serialises mutations to `cumulative` and to
the session log. The cadence decision (`turn_count % k`) moves inside
`HarnessRunner.on_trajectory_progress` so it isn't duplicated between
live and offline.

### 2.3 Offline driver shape

```python
# agentm/contrib/extensions/llmharness/src/llmharness/replay/offline_driver.py

async def replay_pipeline_over_trajectory(
    *,
    messages: list[AgentMessage],
    cwd: str,
    root_session_id: str,
    provider: tuple[str, dict[str, Any]] | None,
    extractor_settings: ExtractorSettings,
    auditor_settings: AuditorSettings,
    extractor_interval: int = 5,
    audit_interval: int = 5,
    enable_auditor: bool = True,
    stop_on_first_surface: bool = True,
    sidecar_path: Path | None = None,
) -> OfflineRunResult:
    runner = HarnessRunner(
        cumulative=CumulativeAuditState.fresh(),
        child=StandaloneChildRunner(cwd),
        sink=InMemorySink(),
        sidecar=SidecarWriter(sidecar_path) if sidecar_path else None,
        ...,
    )
    for turn_count in range(1, len(messages) + 1):
        step = await runner.on_trajectory_progress(
            messages[:turn_count], turn_count=turn_count
        )
        if stop_on_first_surface and step.surfaced_reminder is not None:
            return OfflineRunResult(
                reminder=step.surfaced_reminder,
                state=runner.cumulative,
                sidecar_path=sidecar_path,
            )
    return OfflineRunResult(reminder=None, state=runner.cumulative,
                            sidecar_path=sidecar_path)
```

The (deleted) `run_offline_auditor_over_control` and the
no-cumulative-threading behaviour of `chain_replay` both collapse into
this single function; the `chained_fork` driver builds N-segment
experiments on top by calling it once per segment with the prior
segment's `state` threaded back in via `seed_cumulative`.

## 3. End-to-end pipeline (the user-facing goal)

The motivation for the refactor is being able to run, for each case, a
chained-fork pipeline of the form **control segment → offline
extract+audit → seed reminder, fork → offline extract+audit → ...** for
N segments, all under the same scenario:

```
┌── one-shot per-case chained-fork ─────────────────────────────────────┐
│                                                                       │
│ SEGMENT 0 (control)                                                   │
│   session_factory(initial_messages=None, seed_reminder_text=None)     │
│     → control.final_messages, control.session_log_id                  │
│                                                                       │
│   offline_run = await replay_pipeline_over_trajectory(                │
│       messages = control.final_messages,                              │
│       cumulative = CumulativeAuditState.fresh(),                      │
│       stop_on_first_surface = True,                                   │
│   )                                                                   │
│   # offline_run carries: reminder, state, all_step_results            │
│   # (per-firing extractor_record + auditor_record captured even with  │
│   # sidecar_path=None, so write_chained_replay can stitch later)      │
│                                                                       │
│ for k in 1..max_interventions while a reminder surfaces:              │
│   SEGMENT k (branch)                                                  │
│     prefix = _fork_prefix(parent.final_messages, surface_turn_{k-1})  │
│     # ^ inclusive cut + paired-tool_result bump                       │
│     branch = await session_factory(                                   │
│         initial_messages = prefix,                                    │
│         seed_reminder_text = reminder_{k-1}.text,                     │
│     )                                                                 │
│     offline_run = await replay_pipeline_over_trajectory(              │
│         messages = branch.final_messages,                             │
│         seed_cumulative = previous.state,                             │
│         start_turn      = surface_turn_{k-1} + 1,                     │
│         stop_on_first_surface = True,                                 │
│     )                                                                 │
│                                                                       │
│ FINALISE                                                              │
│   write_chained_replay(                                               │
│       segments, out_path = <final_sid>.chained.jsonl                  │
│   )                                                                   │
│   # Each segment k contributes records in [turn_lo_k, turn_hi_k];     │
│   # all entries rebound to segments[-1].payload.session_log_id.       │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

A 100-case batch is the same pipeline in a per-case loop, parallelisable
within provider rate limits. The `chained_fork` intervention in
`agentm_rca/eval/agent.py` is a thin caller: it supplies the
`SessionFactory` closure that turns `(initial_messages,
seed_reminder_text)` into an `_SessionRun`; everything else (cadence,
windowing, cumulative-state threading, sidecar emission) lives inside
`replay_pipeline_over_trajectory` + `run_chained_fork_experiment`.

## 4. Acceptance invariants

A correct refactor satisfies three diff-testable invariants:

1. **Live ≡ Offline equivalence on a fixed trajectory.** Capture a live
   `rca:harness.sync.extractor5` run's `final_messages` and sidecar.
   Re-run `replay_pipeline_over_trajectory` with the same provider on
   those messages. The resulting sidecar's extractor `output.events` /
   `output.edges` / `output.dropped_edges` and the per-firing
   `compose_kwargs` / `payload` schemas match byte-for-byte (modulo
   non-deterministic fields: `ts_ns`, `extraction_run_id` uuid,
   `provider`'s `config` if non-canonical, `raw_assistant_messages`
   under LLM nondeterminism). This is the only test that proves
   windowing + payload composition exists in exactly one place.

2. **Single-firing replay is a degenerate case.** As shipped in P3,
   `replay_extractor_record` / `replay_auditor_record` are thin
   wrappers in `replay/runner.py`:

   ```python
   async def replay_extractor_record(record, *, cwd, provider_override, prompt_override):
       runner = HarnessRunner(
           cumulative=CumulativeAuditState.fresh(),
           child=StandaloneChildRunner(cwd),
           sink=NoopSink(),
           sidecar=None,
           extractor_settings=ExtractorSettings.from_compose_kwargs(
               record.compose_kwargs, prompt_override=prompt_override,
           ),
           auditor_settings=AuditorSettings.empty(),
           extractor_interval=1, audit_interval=1, enable_auditor=False,
           root_session_id=record.root_session_id,
           provider_extractor=provider_override or _coerce_provider(record),
           provider_auditor=None,
           cwd=cwd,
       )
       return await runner.fire_extractor_from_record(record)
   ```

   The seam is `fire_extractor_from_record` (resp.
   `fire_auditor_from_record`) on the runner itself; it reconstructs
   per-firing state inline from `record.payload` + `record.extras`
   rather than going through a `CumulativeAuditState`
   hydrate-helper. No separate code path for "single firing"; the
   wrapper is sugar.

3. **Sidecar is an artefact, not a seed.** A `HarnessRunner`'s only
   non-message input is `CumulativeAuditState` (which the runner can
   construct fresh) and the settings (prompt, cadence). The runner
   never reads a `ReplayRecord` to decide what to do. Sidecar writing
   is a `SidecarWriter` sink; the same runner with `sidecar=None`
   produces identical state in `cumulative`.

## 5. Migration plan

Phased, no shims, hard-cut per phase:

| Phase | Scope | Deliverable |
|---|---|---|
| P1 | Land `CumulativeAuditState` + `ChildRunner` / `OpSink` Protocols + `HarnessRunner` (live path only) | live adapter routed through runner; live `_drain_extractor` / `_drain_auditor` shrink to wrappers; existing tests (live sidecar shape) still pass byte-identical |
| P2 | Land `StandaloneChildRunner` + `InMemorySink` + `replay_pipeline_over_trajectory` | invariant #1 (live ≡ offline) verifiable; `chain_replay` reimplemented over runner with cumulative threading enabled |
| P3 | Collapse `run_offline_auditor_over_control` + rewrite `replay_extractor_record` / `replay_auditor_record` as thin wrappers | **Shipped:** `HarnessRunner.fire_extractor_from_record` / `fire_auditor_from_record` are the new seams; `replay/runner.py` wrappers route through them. `NoopSink` (in `audit/_offline_seams.py`) is the sink for single-firing replay. Invariant #2 holds. |
| P4 | Wire `agentm_rca` `baseline_fork` to support `rca:baseline → rca:baseline` via the offline runner | **Superseded by P5.** Initially shipped as `_HARNESS_VARIANT_TO_CONTROL[rca:baseline] = rca:baseline` + a baseline-branch inside `_run_baseline_fork`; replaced 2026-05-24 (see P5). |
| P5 | Replace strict-A/B `baseline_fork` with N-segment `chained_fork` | **Shipped 2026-05-24:** `replay_pipeline_over_trajectory` gains `seed_cumulative` + `start_turn` so segment k+1's auditor sees segment k's verdicts; `StepResult` gains `extractor_record` (mirroring `auditor_record`) so multi-segment sidecars assemble in-process. New driver `llmharness.replay.chained_fork.run_chained_fork_experiment` runs the loop; `write_chained_replay` materialises a single `<final_sid>.chained.jsonl`. `strict_ab.py` deleted; `_HARNESS_VARIANT_TO_CONTROL` deleted (every segment uses the same scenario); `agentm_rca` constructor narrowed to `chained_fork: bool` + `max_interventions: int`. Previously: strict_ab — see git history (commit `7b259e2..c966e24`). |

Each phase is independently revert-able: P1's wrapper preserves the
existing live behaviour; P2-P4 add capability without changing P1.

## 6. Non-goals

- **Cross-case state sharing**: each case still owns its own
  `CumulativeAuditState`. Cross-case aggregation (the existing
  `aggregate.py` pipeline) is unchanged.
- **Parallelism within a case**: the in-case loop stays serial. The
  async queue inside the live adapter is for serialising mutations to
  one session log, not parallelism.
- **Replay-record format changes**: `ReplayRecord` schema is unchanged.
  The conflation noted in §1.3 is resolved by *how* the record is read,
  not by changing its shape.
- **Auditor verdict semantics**: surfacing rules, reminder format, and
  the `audit_check_registry` plug-in surface are unchanged.
