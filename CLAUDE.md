# CLAUDE.md

Index for Claude Code working in this repo. Pointers, not exhaustive docs —
follow links for detail.

## Project

AgentM — pluggable agent framework. Python 3.12+, `uv` only. The SDK is a
mechanism; every policy is a replaceable atom. Boundary contract in
`.claude/designs/pluggable-architecture.md`.

## CLI

- `agentm "<prompt>"` — one-shot prompt (default scenario `general_purpose`).
- `agentm trace …` — query the OTLP/JSON session log
  (`messages` · `turns` · `tools` · `chats` · `info` · `index`); preferred
  over hand-parsing `.agentm/observability/*.jsonl`.
- Channel CLIs: `agentm-gateway`, `agentm-worker`, `agentm-terminal`,
  `agentm-feishu`.
- Shared `AGENTM_*` env namespace; `.env` autoloaded. Long-lived model
  settings can live in `~/.agentm/config.toml` instead of env vars
  (`$AGENTM_HOME/config.toml` overrides the directory). Precedence:
  CLI flag > shell env / `.env` > `config.toml` profile/default_model >
  provider default. Per-CLI prefixes (`AGENTM_GATEWAY_*`) are **not**
  supported. Minimal config:
  ```toml
  default_model = "doubao"

  [models.doubao]
  provider = "openai"
  model = "doubao-seed-2-0-pro-260215"
  base_url = "https://ark.cn-beijing.volces.com/api/v3"
  api_key = "..."
  context_window = 131072
  ```
  Select a profile with `agentm --model doubao`; if `default_model` is set,
  no env vars are needed for the default provider/model/key. Run
  `<cli> --help` for flags.
- Optional extra: `uv sync --extra agent-env` installs `arl-env` for the
  `operations_agent_env` atom (ARL-sandboxed Operations).

### Trace debugging combos

A logical trace spans many JSONL files — one root session + N spawned
children (`purpose` distinguishes `root` / `cognitive_audit_extractor` /
`cognitive_audit_auditor`). Composition pattern:

- `agentm trace index` is the **selection layer**: it scans the
  observability dir and emits one identity row per session file
  (`{path, trace_id, session_id, parent_session_id, purpose, scenario,
  records}`). It maps a `trace_id` — the id `eval.db.evaluation_data.trace_id`
  stores — to its session files. It's the **only** directory-granular verb;
  every other verb stays single-file by design (`--file`/`--session`/`--latest`).
- Idiom: `index --format ndjson | jq 'select(.trace_id==…)' | <loop per .path>
  agentm trace {tools,messages,…} --file "$f" --format ndjson | jq …`.
- Canonical example — extract auditor verdicts across one trace:

  ```bash
  TID=dfc09e403bd64ca59c01bfa805962526
  agentm trace index --format ndjson \
   | jq -r --arg t "$TID" 'select(.trace_id==$t and .purpose=="cognitive_audit_auditor")|.path' \
   | while read f; do agentm trace tools --file "$f" --tool submit_verdict --format ndjson; done \
   | jq -c '.args.verdict|{fired:.surface_reminder, reminder:.reminder_text}'
  ```

  Filtering is the consumer's job (jq) — `index` has no `--trace` flag.

## Repo exploration

- `agentm list-extensions [--source builtin|contrib|user|all] [--filter X]`
- `ls contrib/scenarios/` — names usable as `--scenario <name>`
- `ls src/agentm/extensions/builtin/` — builtin atoms (one file per atom)
- `ls contrib/extensions/` — third-party atoms (flat files auto-discover;
  nested packages need `--extension <dotted.path>`)
- `.claude/index.yaml` — design-concept graph; `.claude/designs/` — concept docs
- `CONTEXT.md` — project glossary; check here when an unfamiliar term appears.
- `core-manifest.yaml` — constitution layer (kernel-singleton declarations); read-only.

## Architecture

```
presenters: agentm.cli  /  embedded SDK
atoms:      src/agentm/extensions/builtin/  +  contrib/extensions/
substrate:  agentm.core/  (abi · runtime · lib — write-protected)
```

- Atoms reach stateful subsystems only through `ExtensionAPI` services;
  `extensions.validate` rejects direct `core.runtime.*` imports.
- Five pluggability axes are `Protocol`s in `core.abi`, registered by atoms
  via `api.register_*`. See `.claude/designs/pluggable-architecture.md`.

## Extensions & scenarios

- **Builtin atom**: `src/agentm/extensions/builtin/<name>.py` — one file,
  exports `MANIFEST` + `install(api, config)`. §11 contract: no
  atom-to-atom imports, no `core.runtime.*`, no `core._internal`.
- **Scenario**: YAML at `contrib/scenarios/<name>/manifest.yaml`, selected
  via `--scenario <name>`. Default is `general_purpose`.
- **contrib/extensions/**: flat `<name>.py` auto-discovers; nested packages
  mount via `--extension <dotted.path>` and are **not** scenarios.

## Design docs (`.claude/`)

- `index.yaml` — concept graph; keep in sync on every concept change.
- `designs/<concept>.md` — continuously maintained.
- `plans/YYYY-MM-DD-*.md`, `tasks/YYYY-MM-DD-*.md` — append-only.

Concept-change flow: update the design doc → check `index.yaml`
`related_concepts` → propagate → update `index.yaml` → append plan/task
if implementing.

## Testing

Quality over quantity. A test exists only to protect a **fail-stop
position**:

| Position | Why load-bearing |
|---|---|
| Constitution boundary (`is_constitution_path`, manifest reload) | Agent self-modifies kernel |
| Atom hash determinism (`compute_atom_hash`) | Evidence attribution corrupt |
| Active-set fingerprint pairing | Observation can't link to atom version |
| Catalog freeze idempotence | Catalog state untrustworthy |
| Indexer rebuild idempotence | Evolution evidence drifts |
| Transactional reload atomicity | Live agent in inconsistent state |
| §11 extension contract validator | Bad atoms slip into catalog |

`pytest` markers: `ui` (Textual TUI) and `slow` (real-LLM E2E,
minutes-long) — both opt-in.

**E2E** = drive `agentm` with NL prompts on a sandbox repo and verify
through `agentm trace`. Never substitute SDK-internal assertions
(`session._tools`, `session._apis`). Lock real bugs down with a
stub-provider integration test.

## Dev loop

After every change:

```bash
uv run ruff check <files>
uv run mypy <files>
uv run pytest --tb=short
```

Prefer targeted `# type: ignore[attr-defined]` over broad suppression.
For identity-affecting changes (atoms, kernel, catalog): also run an E2E
prompt against a sandbox repo and inspect the trace.

CI lints/types a broader scope — `src/`, `contrib/channels/src/`,
`contrib/extensions/llmharness/src`, `contrib/scenarios/rca/src` — and
runs mypy per workspace from each member's root (per-package overrides).
For sweeping changes, mirror that scope locally.

## Iteration tracking

- `progress.tsv` — dev-loop keep/discard decisions + metric values.
- `decisions.md` — long-horizon autonomous decisions (L2+).
- `.claude/{plans,tasks}/` — append-only design history.

## Conventions

- **Language**: code, comments, commits, design docs in English;
  conversation in Chinese.
- **No SDK / scenario conflation**: scenario-specific logic never inside
  `agentm.core`.
- **§11 atom contract**: enforced by `extensions.validate`.
- **No preset enums for subjective fields** — free-text + LLM-decided.
- **Auto-commit awareness**: `agentm` auto-commits during sessions; run
  E2E in a sandbox, never on `main`.

## Requirements index

`project-index.yaml` (repo root) is the single source of truth for product
requirements. Every code change keeps `code` / `tests` paths and `status`
in sync — many entries currently have stale paths from the harness-collapse
migration (e.g. `src/agentm/harness/`, `src/agentm/llm/`); fix them as you
touch the affected requirements. Validation runs through the autoharness
skill. Distinct from `.claude/index.yaml` (design-concept graph).

<!-- auto-harness:begin -->
## Core principles

1. **Quality over quantity** — a few things done well beats many done poorly.
2. **Surface problems early** — fail fast, validate before investing, outline before drafting.
3. **Deliberate execution** — every decision traceable to a reason.

Full text: `/autoharness:guide`.

## North-star targets

| Target | Measure |
|---|---|
| Spec coverage | fraction of `project-index.yaml` requirements at `status: tested` |
| Test health | `uv run pytest --tb=short` pass rate; every `implemented` requirement has tests |
| Index integrity | `validate_index.py` reports 0 violations |
| Code health | `uv run mypy src/` + `uv run ruff check src/` both clean |

Secondary: simple code mapping cleanly to requirements > clever abstractions
serving five.

## Active skills

| Skill | Purpose |
|---|---|
| `/autoharness:guide` | methodology briefing |
| `/autoharness:dev-loop` | implement → test → vibe → review → measure |
| `/autoharness:north-star` | define and track optimization targets |
| `/autoharness:long-horizon` | autonomous decisions with escalation ladder |
| `/autoharness:existing-project` | recover `project-index.yaml` from current code/docs |
| `/autoharness:notify` | push iteration reports (not yet configured) |
| `/autoharness:skill-feedback` | file issues back to the autoharness plugin |
<!-- auto-harness:end -->

## Related plugins

- **workbuddy** — pipeline monitoring / repo setup / incident handling.
  Repo carries `.github/workbuddy/`; install with
  `/plugin install workbuddy@workbuddy-local`.
