# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentM is an agent framework written in Python. It is in early development (v0.1.0) with a scaffold entry point and no dependencies yet.

## Language Rules

- All code, comments, commit messages, design documents, and file content: **English**
- Communication with the user (conversation, explanations, questions): **Chinese**

## Build & Development Commands

This project uses **uv** as the package manager and build tool.

```bash
uv sync              # Install dependencies
uv run agentm        # Run the CLI entry point
uv add <package>     # Add a dependency
uv run pytest        # Run tests
uv run pytest tests/path/to/test_file.py::test_name  # Run a single test
uv run ruff check src/   # Lint check
uv run mypy src/         # Type check
```

## Code Quality Checks

After making code changes, **always** run lint and type checks on the modified files:

```bash
uv run ruff check <changed-files>
uv run mypy <changed-files>
```

Fix all errors before considering the task complete. For `mypy` issues on dynamic/duck-typed parameters (e.g., `broadcaster: object`), use targeted `# type: ignore[attr-defined]` annotations rather than suppressing broadly.

## Architecture

- **Package**: `src/agentm/` — source layout with `__init__.py` as the module root
- **Entry point**: `agentm:main` registered as a console script in `pyproject.toml`
- **Python**: 3.12+ required
- **Build backend**: `uv_build`

## scenarios/ layout

Each subdir is **either** a scenario (a `manifest.yaml` + optional local atoms; resolved by-name at `agentm run --scenario <name>`) **or** a sibling project that lives here for convenience. The loader (`src/agentm/extensions/loader.py`) only looks up scenarios by exact name, so non-scenario siblings are safe.

- `scenarios/<name>/manifest.yaml` — standard scenario form (e.g. `plan_mode`, `rca`, `trajectory_analysis`, `harness_monitor`)
- `scenarios/llmharness/` — **nested project**, not a scenario. It is the llmharness Python package + Claude Code plugin + AgentM extension (moved here on 2026-05-07 from its own repo). Has its own `pyproject.toml`, `CLAUDE.md`, and dev-loop. Owns the `harness_monitor` scenario above as its consumer. See `scenarios/llmharness/CLAUDE.md` for its own conventions.

Don't add scenario-loader logic that walks subdirs blindly — keep `scenarios/` open to nested projects.

## Design Documentation System

All design documents are stored in the `.claude/` directory. A `.claude/index.yaml` file maintains the conceptual index and relationships between concepts.

### Directory Structure

```
.claude/
├── index.yaml          # Concept index and relationship graph, must always stay up-to-date
├── designs/            # High-level design documents (continuously maintained, updated with changes)
├── plans/              # Complete implementation plans (append-only, never modify existing files)
└── tasks/              # Subtasks within plans (append-only, never modify existing files)
```

### File Rules

| Directory | Lifecycle | Naming Convention | Description |
|-----------|-----------|-------------------|-------------|
| `designs/` | **Continuously maintained** | `<concept>.md` | Top-level design; must sync updates when concepts change |
| `plans/` | **Append-only** | `YYYY-MM-DD-<plan-name>.md` | Complete implementation plan |
| `tasks/` | **Append-only** | `YYYY-MM-DD-<task-name>.md` | Breakdown of subtasks under a plan |

### index.yaml

`index.yaml` is the relationship graph for the entire design system, recording references between each concept and related documents.

**Structure Convention:**

```yaml
concepts:
  <concept-name>:
    description: "One-sentence description"
    design: "designs/<file>.md"        # Corresponding design document
    related_concepts: [<other-concept>] # Related concepts
    plans: ["plans/<file>.md"]          # Related plans
    tasks: ["tasks/<file>.md"]          # Related tasks
```

**Maintenance Rules:**

1. When adding design/plan/task files, must sync update `index.yaml`
2. When a concept changes, trace `related_concepts` links in `index.yaml` to find all affected documents and update them
3. If changes impact documents in `designs/`, must update the design documents

### Cross-Referencing

All markdown documents link to each other via relative paths:

- Design references other designs: `[concept-name](other-concept.md)`
- Plan references design: `[design-doc](../designs/concept.md)`
- Task references plan: `[plan](../plans/YYYY-MM-DD-plan.md)`
- Task references design: `[design](../designs/concept.md)`

### Change Propagation Workflow

When any design concept changes, execute the following workflow:

1. **Update design document** — Modify the corresponding file in `designs/`
2. **Query index.yaml** — Find all related concepts in `related_concepts`
3. **Assess impact scope** — Determine if related design documents need updating
4. **Update affected documents** — Sync changes to all affected design files
5. **Update index.yaml** — If adding/removing concepts or relationships, update the index
6. **Create new plan/task** — If changes need an implementation plan, append new files in `plans/` and `tasks/` (never modify existing files)

## Testing Philosophy

Every test must be deliberate. Adding, modifying, or removing a test requires careful justification.

### First-principles rule (load-bearing)

**Quality over quantity. Only test positions where AgentM's value proposition fails if broken.** Before adding a test, ask: "if this assertion never existed, what realistic disaster could ship?" If you can't name one tied to AgentM's identity (self-modifiability, evidence-driven evolution, pluggable SDK boundary), don't write it. Single-tool happy paths, vendor wiring, utility helpers, and framework guarantees are NOT core — even if a bug exists there, it isn't an AgentM-defining failure.

This rule supersedes any urge to broaden coverage. A test suite of 7 well-chosen files that map 1:1 to fail-stop positions beats 60 files that pad metrics.

### Core test positions (the only places that warrant tests by default)

| Position | Why it's load-bearing |
|---|---|
| Constitution boundary (`is_constitution_path`, manifest reload) | Wrong → agent self-modifies kernel → bricks itself |
| Atom hash determinism (`compute_atom_hash`) | Wrong → evidence attribution corrupt |
| Active-set fingerprint pairing | Wrong → observation can't be linked to atom version |
| Catalog freeze idempotence | Wrong → catalog state untrustworthy |
| Indexer rebuild idempotence | Wrong → evolution evidence drifts |
| Transactional reload atomicity | Wrong → live agent in inconsistent state |
| §11 extension contract validator | Wrong → bad atoms slip into catalog |

New tests outside this list require an explicit justification of which fail-stop they protect. If none, the test is decoration — delete instead of merge.

### Principles

1. **Test behavior, not structure** — Ask "what does this code do" not "what fields does it have". Never test language guarantees (enum values exist, dataclass defaults work, imports succeed).
2. **Every test must answer: "what bug does this prevent?"** — If you can't articulate a realistic failure scenario, don't write the test.
3. **Boundaries over happy paths** — Edge cases, invalid inputs, and state transitions are where bugs live. A single well-chosen boundary test beats ten happy-path assertions.
4. **One scenario per test, multiple asserts are fine** — Group related assertions that verify the same logical scenario. Don't split `assert a` and `assert b` into separate tests if they test the same thing.
5. **Don't test other people's code** — Pydantic validation, Python dataclass semantics, enum membership, `isinstance` checks — these are framework guarantees, not our responsibility.

### What to Test

- **State transitions and rules** — PhaseManager valid/invalid transitions
- **Immutability contracts** — Notebook operations return new instances, originals unchanged
- **Boundary conditions** — Empty inputs, missing keys, malformed paths
- **Cross-module contracts** — State registry resolves correct types, config validation rejects bad references
- **Invariants the design doc mandates** — e.g., HypothesisStatus values match the tool's Literal constraint

### What NOT to Test

- Field existence or default values on dataclasses/TypedDicts/Pydantic models
- Enum value counts or individual member existence
- Import success (`assert X is not None`)
- Type inheritance (`isinstance` checks)
- Stub functions that only `raise NotImplementedError`

### End-to-end testing methodology

**E2E means: drive the agent with natural-language prompts and verify by inspecting the trajectory. Do not call SDK / harness internals to "shortcut" the verification.**

The reason: AgentM's identity bugs (a tool the agent can't actually invoke, an atom that doesn't survive a session restart, a write that should have committed and didn't) only show up through the user-visible loop — `agentm` CLI in, trajectory out. SDK-level assertions (`assert "echo_shout" in session._tools`) bypass exactly the layers most likely to be wrong: the LLM tool list, the kernel dispatch index, the per-prompt tool-list snapshot. A passing SDK test next to a broken CLI experience is the failure mode this rule prevents.

**The procedure:**

1. Set up a sandbox cwd (a real git repo, not a worktree of the main checkout — `agentm` auto-commits and you don't want those commits landing on `main`).
2. Run `uv run agentm --cwd <sandbox> "<natural-language prompt>"`. Phrase the prompt the way a real user would; do not name internal tools (`install_atom`, `unload_atom`) unless the test is specifically about their NL discoverability.
3. Inspect the resulting trajectory at `<sandbox>/.agentm/observability/<trace>.jsonl`. The fields that matter:
   - `name == "emit:tool_call"` → who was called and with what args
   - `name == "emit:tool_result"` → exit text + `is_error` flag
   - `name.startswith("install:")` → which atoms were loaded at boot (catches auto-discovery regressions)
   - `name.startswith("emit:diagnostic")` → soft failures that didn't raise
4. Cross-check on-disk state if the prompt was a write: `git log --oneline` for auto-commit, `<sandbox>/.agentm/atoms/` for installed atoms.

**What "verify by trajectory" rules out:**
- Calling `session._tools` / `session._apis` / `session._reloader` from a Python script. If the trajectory doesn't show it, it didn't happen for the user.
- Asserting on `prompt()` return values from a stub provider in a unit test *as a substitute* for an e2e check. Stub-provider integration tests (like `test_install_atom_in_turn_n_is_dispatchable_in_turn_n_plus_one`) are valid as fail-stops, but they're not e2e — they don't catch LLM-prompt-format regressions, tool-description discoverability, or boot-time auto-discovery gaps.
- Running the CLI and trusting its final-text summary. The agent's self-report of success is not evidence; the trajectory's `tool_result` `is_error=False` is.

**When a real bug is found via trajectory, lock it down with a stub-provider integration test** so the regression is caught in CI without needing API keys. Trajectory inspection is for *finding* failures and confirming user-visible behavior; the integration test is what *prevents* recurrence.

## Slash Commands

Project-specific commands are located in `.claude/commands/`:

| Command | Purpose |
|---------|---------|
| `/design <concept>` | Create or update design document, maintain index.yaml relationships, propagate changes |
| `/plan` | Create implementation plan, break down into tasks, wait for user confirmation before execution |
| `/index [show\|check\|fix]` | View, verify, or fix index.yaml consistency |
| `/status` | Project status overview: design progress, plan status, index health |
| `/tdd` | TDD development workflow (use after entering implementation phase) |
| `/eval [define\|check\|report]` | Manage acceptance criteria and evaluation |
| `/checkpoint [create\|verify\|list]` | Workflow checkpoint management |
| `/learn` | Extract reusable patterns from current session |

## Agents

Project-specific agents in `.claude/agents/`:

| Agent | Role | Model | Tools | Primary Commands/Skills |
|-------|------|-------|-------|------------------------|
| **architect** | Architecture design, produce design docs, maintain index.yaml | opus | Read, Grep, Glob | `/design`, `/index` |
| **planner** | Break designs into executable plans and tasks | sonnet | Read, Grep, Glob | `/plan`, `/status` |
| **tdd** | Write tests first (RED), guide TDD cycle | sonnet | Read, Write, Edit, Bash, Grep, Glob | `/tdd`, `/eval` |
| **implementer** | Execute plans, write code following designs | sonnet | Read, Write, Edit, Bash, Grep, Glob | `/checkpoint` |
| **reviewer** | Verify implementation matches design, code quality | opus | Read, Grep, Glob, Bash | `/eval`, `/index check` |

### Agent Workflow

```
architect → planner → tdd (write tests) → implementer (write code) → reviewer
    ↑                                                                     |
    └─────────────── feedback (design issues) ────────────────────────────┘
```

### When to Use Each Agent

- **Design discussion / new concept** → architect
- **Break work into tasks** → planner
- **Write tests for a module** → tdd
- **Implement a plan task** → implementer
- **Verify completed work** → reviewer
