# Running modes

Three concerns — **extractor**, **auditor**, **reminder injection** —
are decoupled at the function/composer layer and re-wired only by
`adapters/agentm.py`. This doc spells out which knobs flip which
piece, how to run them separately, how to combine them, and where to
plug an SFT-trained model.

If you only want a quick mental model: read §1 and skim §5.

---

## 1. Decoupling matrix

| Piece | Composer | Where it runs | Inputs | Outputs |
|---|---|---|---|---|
| Extractor | `audit/extractor/extensions.py: compose_extractor_extensions` | child session (live: `spawn_child_session`; offline: `replay.engine.run_phase_standalone`) | trajectory window `{new_turns, recent_graph}` | `events[] + edges[]` written into the parent session entry tree + replay sidecar |
| Auditor | `audit/auditor/extensions.py: compose_auditor_extensions` | child session, same two execution paths | `{events, edges, phases, findings, continuation_notes}` — **graph only**, trajectory used only by the optional `get_turn` drill-down tool | `Verdict` (incl. `surface_reminder?`) into entry tree + sidecar |
| Reminder injection | `adapters/agentm.py:_make_reminder_injector` (live) and `replay/reminder_seed.py` (offline / prefix-replay) | on the **main** agent's bus via `DecideTurnActionEvent` → `Inject([reminder_msg])` | the `surface_reminder` text from a Verdict | one synthetic user message + `REMINDER_DELIVERED` entry |

The single source of truth for the reminder message shape is
`audit/_reminder_format.py: build_reminder_message`; both call sites
import from there so train-time and inference-time messages stay
byte-identical.

### Live adapter knobs (`adapters/agentm.py`)

| Config key | Default | Effect |
|---|---|---|
| `enable_auditor` | `true` | Run Phase 2 every `audit_interval_turns` turns. `false` → extractor still records the graph, no auditor calls. |
| `enable_reminders` | `true` | Drain queued reminders into the main agent on the next `DecideTurnAction`. `false` → verdicts are persisted (and the `surface_reminder` field still written to the sidecar) but the main agent never sees them. Ignored when `enable_auditor=false`. |
| `enable_replay_log` | `true` | Append every phase invocation to `<cwd>/.agentm/audit_replay/<sid>.jsonl`. Required for distill / replay / prefix-replay workflows. |
| `audit_interval_turns` | `3` | k for "every k turns". |

There is **no** `enable_extractor` flag — extractor is always-on under
the adapter because auditor depends on the graph it builds.

---

## 2. Run extractor only

### Live (build a graph from a real agent run)

```bash
agentm --cwd "$RUN_DIR" \
       --extension 'llmharness.adapters.agentm:{"enable_auditor":false,"enable_reminders":false,"enable_replay_log":true}' \
       "<your prompt>"
```

(`agentm`'s `--extension` flag is unified: `MODULE` or
`MODULE:{json_config}`. There is no separate `--extension-config`.)

Produces `<RUN_DIR>/.agentm/audit_replay/<sid>.jsonl` containing only
`phase: "extractor"` records, plus `audit_event` / `audit_edge` /
`extractor_cursor` entries on the session entry tree.

### Offline from a legacy eval.db row

`adapters/eval_db.py` is the host-side driver — folds a langgraph
event stream into the same `new_turns` shape and runs the extractor as
a top-level session. Same composer, same prompt, same witness pipeline.

```bash
python -m llmharness.adapters.eval_db extract \
    --db    path/to/eval.db \
    --out-dir runs/eval_db/<exp-id> \
    --window 0                          # 0 = whole trajectory single-shot
```

Both formats end up emitting `ReplayRecord(phase="extractor", ...)` to
the same sidecar shape, so downstream (auditor replay, distill,
aggregate) doesn't care where the graph came from.

### Bisect a specific extractor firing

```bash
llmharness-replay extractor \
    --record runs/.../audit_replay/<sid>.jsonl:7    # turn_index=7
    --prompt audit/extractor/prompts/<variant>.md
    --provider 'openai:{"model":"gpt-4o"}'
```

---

## 3. Run auditor only

The auditor is graph-in / verdict-out, so anything that produced an
extractor sidecar can feed it — there is no separate "auditor input
adapter" for eval.db.

```bash
llmharness-replay auditor \
    --record runs/.../audit_replay/<sid>.jsonl:11
    --prompt audit/auditor/prompts/<variant>.md
    --provider 'openai:{"model":"gpt-4o"}'
```

To iterate on auditor prompts / tool profiles offline, use the chain
form:

```bash
llmharness-replay chain --record-dir runs/.../audit_replay/ ...
```

See [`docs/05-profiles-and-prompts.md`](05-profiles-and-prompts.md)
for profile / prompt variant management.

---

## 4. Run reminder injection only (prefix replay)

When you've already recorded an auditor verdict that fires a reminder
at turn t, you can branch the original main-agent session at turn t
and replay just the tail — no need to re-run turns 0..t-1.

Full recipe: [`docs/07-prefix-replay.md`](07-prefix-replay.md).

```bash
llmharness-replay agent-from-reminder \
    --audit-replay runs/.../audit_replay/<sid>.jsonl \
    --turn 7
# emits an `agentm --resume <new-sid> --extension llmharness.replay.reminder_seed ...` command
```

The one-shot `reminder_seed` atom drains its configured reminder on the
**first** `DecideTurnActionEvent` via the same `Inject` channel the
live adapter uses, then unsubscribes — so the resumed agent's first
visible message is byte-identical to what it would have been live.

Note: the emitted `agentm --resume <sid>` command carries no positional
prompt argument. Since agentm v0.1.x the CLI accepts resume-only
invocations and routes them through `AgentSession.tick`, giving
extensions on `decide_turn_action` (the seed atom here) the first
chance to supply input via `Inject`. If no extension injects, the
session exits cleanly with `NoPendingInput` and the trajectory is
unchanged.

---

## 5. Run them together

Default mode — every knob defaults to `true`:

```bash
agentm --extension llmharness.adapters.agentm "<prompt>"
```

For data collection (live full pipeline, no reminder side-effects on
the main agent), the distill recipe pattern:

```bash
LLMHARNESS_DISTILL_SAMPLE_ID="$SAMPLE_ID" \
LLMHARNESS_DISTILL_DATASET="$DATASET_PATH" \
  agentm --cwd "$RUN_DIR" \
         --scenario rca \
         --extension 'llmharness.adapters.agentm:{"enable_reminders":false,"enable_auditor":true,"enable_replay_log":true,"audit_interval_turns":3}' \
         --extension llmharness.distill.binding \
         "$PROMPT"
```

`enable_reminders=false` is what keeps the trajectory clean for SFT:
the auditor still fires and writes verdicts to the sidecar, but the
main agent never sees them, so the recorded trajectory is the same
trajectory the teacher main agent would have produced without
supervision.

See [`docs/03-distill-recipe.md`](03-distill-recipe.md) §2 for the
full per-sample loop and `LLMHARNESS_DISTILL_*` env contract.

---

## 6. Plug an SFT-trained model in

The harness has **three** independently-replaceable LLM endpoints. An
SFT'd 4B can drop into any one of them.

| Endpoint | Where it runs | How to swap | Read |
|---|---|---|---|
| **Main agent** | the AgentM session itself | `agentm --provider <id> --model <id>` (or `AGENTM_PROVIDER` / `AGENTM_MODEL` env) | `<AgentM-root>/CLAUDE.md` |
| **Extractor child** | spawned per turn by the adapter | adapter config key `extractor_provider: {"module":"openai","config":{"base_url":"http://vllm:8000","model":"sft-extractor-4b"}}` (defaults to inheriting the parent provider) | adapter `config_schema` |
| **Auditor child** | spawned every k turns | same shape under key `auditor_provider` | adapter `config_schema` |

Wire those into the unified `--extension` flag:

```bash
agentm --extension 'llmharness.adapters.agentm:{
  "extractor_provider": {"module":"openai","config":{"base_url":"http://vllm:8000","model":"sft-extractor-4b"}},
  "auditor_provider":   {"module":"openai","config":{"base_url":"http://vllm:8000","model":"sft-auditor-4b"}}
}' "<prompt>"
```

(The JSON above is shown formatted for readability; pass it on one line
or quote-escape the newlines for the shell.)

For **offline** replay (the cheaper iteration loop while the model is
still training):

```bash
# extractor-only A/B on the recorded firing
llmharness-replay extractor --record <sidecar>:<turn> \
    --provider 'openai:{"base_url":"http://vllm:8000","model":"sft-extractor-4b"}'

# auditor-only A/B on the recorded graph
llmharness-replay auditor   --record <sidecar>:<turn> \
    --provider 'openai:{"base_url":"http://vllm:8000","model":"sft-auditor-4b"}'

# whole agent's response to a reminder seeded at turn t
llmharness-replay agent-from-reminder --audit-replay <sidecar> --turn <t>
# ...then run the emitted command with --provider pointing at the SFT model
```

The provider tuple shape is the same one AgentM uses everywhere:
`(provider_id, config_dict)`. The config dict is forwarded verbatim to
the provider atom, so vLLM / SGLang / TGI / OpenRouter etc. all work
through the OpenAI-compatible provider with `base_url` + `model`
overrides.

### Sanity check before/after a model swap

Like `agentm`, the replay CLI bridges `AGENTM_PROVIDER` / `AGENTM_MODEL`
and provider-specific env vars (`OPENAI_BASE_URL`, `OPENAI_VERIFY_SSL`,
`WARPGATE_TICKET`, ...) into the provider config automatically. So if
your shell already targets a Warpgate-fronted or self-signed endpoint
for `agentm`, the same shell can run replay without re-stuffing each knob:

```bash
# Pulls AGENTM_PROVIDER/AGENTM_MODEL + OPENAI_* from env, same as `agentm`.
llmharness-replay extractor --record <sidecar> --turn <t>

# Explicit override (config dict wins; no env bridging).
llmharness-replay extractor --record <sidecar> --turn <t> \
    --provider 'agentm.extensions.builtin.llm_openai:{"model":"sft-4b","base_url":"http://vllm:8000"}'
```

The extractor SFT contract is `{system, user} → submit_events_batch(events[...])`.
The auditor SFT contract is `{system, user} → submit_verdict(...)`. If
the SFT'd model's tool-call name or schema drifts, the replay record's
`status` flips to `no_call`. Run a small sample with the existing
sidecar before launching a full eval:

```bash
llmharness-replay chain --record-dir runs/.../audit_replay/ \
    --provider 'openai:{"base_url":"http://vllm:8000","model":"sft-..."}' \
    --limit 20
```

The chain runner reports per-record status counts and writes the
re-run output next to the original sidecar for diff inspection.

---

## 7. Cheat sheet

| Goal | Adapter knobs | Where to look |
|---|---|---|
| Live agent with supervision (default) | all defaults | this doc §5 |
| Collect SFT data live | `enable_reminders=false`, distill binding mounted | `03-distill-recipe.md` |
| Build extractor graphs from legacy traces | n/a — use `adapters.eval_db extract` | this doc §2 |
| A/B an extractor prompt or model | n/a — `llmharness-replay extractor` | `05-profiles-and-prompts.md` |
| A/B an auditor prompt, profile, or model | n/a — `llmharness-replay auditor` | `05-profiles-and-prompts.md` |
| See what an agent does if you seed a reminder at turn t | n/a — `llmharness-replay agent-from-reminder` | `07-prefix-replay.md` |
| Run the trained 4B in production | `extractor_provider` / `auditor_provider` config keys; main agent via `--provider`/`--model` | this doc §6 |
