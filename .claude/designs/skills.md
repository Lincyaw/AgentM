# Design: Skill System (Progressive-Disclosure Knowledge Files)

**Status**: DRAFT
**Created**: 2026-05-01
**Last Updated**: 2026-05-01

## Overview

Filesystem-discovered `SKILL.md` files with YAML frontmatter, advertised to the LLM as a compact `<available_skills>` index in the system prompt; the model loads any specific skill on demand via the existing `read` tool.

## Motivation

AgentM's `ResourceLoader` already returns `Skill` records (see [pluggable-architecture.md §3.4](pluggable-architecture.md#34-resource-discovery-the-project-context-boundary), [extension-as-scenario.md §6](extension-as-scenario.md#6-resourceloader-project-context)) but nothing currently injects them into the system prompt and nothing validates the on-disk shape. Without a skill loader, AgentM cannot do progressive disclosure: the only way to give the model new domain knowledge today is to inline it into the system prompt forever, which burns context every turn. Skills give us a "table of contents" pattern — cheap to advertise, lazy to load.

## Design Details

### Discovery layout

The default `ResourceLoader` already walks two roots (`<agent_dir>/skills/` and `<cwd>/.agentm/skills/`). This design adds:

1. **Standardized layout per skill**:
   ```
   <root>/<skill-name>/
     SKILL.md           # required, frontmatter + body
     references/        # optional supporting docs (model loads with `read`)
     examples/          # optional examples (model loads with `read`)
   ```
2. **Recursive scan** with `os.walk(..., followlinks=True)`. When a directory contains `SKILL.md`, that directory is a skill root — do not recurse into it. Pi rule, port verbatim. Symlink loops are guarded by tracking `Path.resolve()` of every visited dir in a set.
3. **`.gitignore` / `.agentmignore` honoring** via [`pathspec`](https://pypi.org/project/pathspec/). Patterns from each `.gitignore` encountered down the walk are accumulated. Pi uses the `ignore` npm package; `pathspec` is the closest stdlib-compatible Python equivalent. (If `pathspec` is judged too heavy, the fallback is a small in-house parser supporting only `*`, `**`, `!`, leading `/`.)

### Frontmatter spec

```yaml
---
name: my-skill           # optional; defaults to parent dir name
description: |           # required; non-empty
  One-paragraph summary the LLM uses to decide whether to read this skill.
disable-model-invocation: false   # optional; if true, only invokable explicitly via /skill:name
---

# Body — the markdown the model sees when it `read`s the file.
```

### Validation rules

| Field | Rule |
|---|---|
| `name` | If present, must equal parent dir name. Lowercase `[a-z0-9-]+`. ≤ 64 chars. No leading/trailing/double hyphens. |
| `description` | Required. Non-empty after `strip()`. ≤ 1024 chars. |
| `disable-model-invocation` | Boolean if present. |
| Body | No constraint. |

Any rule violation produces a `SkillDiagnostic(level="warning", message=..., path=...)`. **The skill is still loaded if the description is non-empty** — name/length/charset issues are warnings, not load failures (matches pi). A skill with empty/missing description is dropped.

Collisions (same `name` from two paths) produce a `level="collision"` diagnostic; first-discovered wins, later ones discarded. Symlinks resolving to the same real path are silently deduplicated.

### Public API surface (in `core/_internal/skills.py`)

```python
@dataclass(frozen=True, slots=True)
class SkillDiagnostic:
    level: Literal["warning", "collision"]
    message: str
    path: str

@dataclass(frozen=True, slots=True)
class SkillRecord:
    """Richer than ResourceLoader.Skill — adds disable_model_invocation,
    file_path (so the prompt can show the absolute location to the model),
    base_dir (the skill folder root, for relative-reference resolution)."""
    name: str
    description: str
    file_path: str
    base_dir: str
    disable_model_invocation: bool
    source: str   # "user" | "project" | "path"

def load_skills_from_dir(dir: str, source: str) -> tuple[list[SkillRecord], list[SkillDiagnostic]]: ...

def load_skills(
    *,
    cwd: str,
    agent_dir: str,
    skill_paths: list[str] = (),
    include_defaults: bool = True,
) -> tuple[list[SkillRecord], list[SkillDiagnostic]]: ...

def format_skills_for_prompt(skills: list[SkillRecord]) -> str:
    """Returns the <available_skills>...</available_skills> XML block, or
    empty string if no model-invocable skills."""
```

`format_skills_for_prompt` produces:

```
\n\nThe following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's file when the task matches its description.
When a skill file references a relative path, resolve it against the skill
directory (parent of SKILL.md / dirname of the path) and use that absolute
path in tool commands.

<available_skills>
  <skill>
    <name>skill-name</name>
    <description>...</description>
    <location>/abs/path/to/SKILL.md</location>
  </skill>
  ...
</available_skills>
```

XML escaping uses stdlib `html.escape` (handles `&`, `<`, `>`, `"`, `'`).

### Extension atom: `extensions/builtin/skill_loader.py`

```python
MANIFEST = ExtensionManifest(
    name="skill_loader",
    description="Discover SKILL.md files and inject an <available_skills> "
                "index into the system prompt.",
    registers=("event:before_agent_start", "event:resources_discover"),
    config_schema={...},   # accepts {skill_paths: [str], include_defaults: bool}
)

def install(api: ExtensionAPI, config: dict) -> None:
    extra_paths: list[str] = list(config.get("skill_paths") or [])
    include_defaults = bool(config.get("include_defaults", True))

    skills_cache: list[SkillRecord] = []
    diagnostics_cache: list[SkillDiagnostic] = []

    async def discover_and_cache() -> None:
        # 1. Ask other extensions for additional skill_paths via `resources_discover`.
        results = await api.events.emit("resources_discover", ResourcesDiscoverEvent(
            cwd=api.cwd, reason="startup",
        ))
        for r in results:
            if isinstance(r, dict):
                extra_paths.extend(r.get("skill_paths") or [])
        # 2. Run loader.
        skills, diags = load_skills(
            cwd=api.cwd,
            agent_dir=str(Path.home() / ".agentm"),
            skill_paths=tuple(extra_paths),
            include_defaults=include_defaults,
        )
        skills_cache[:] = skills
        diagnostics_cache[:] = diags

    api.on("session_ready", lambda _e: discover_and_cache())

    def inject(event: BeforeAgentStartEvent) -> None:
        if not skills_cache:
            return
        block = format_skills_for_prompt(skills_cache)
        if block:
            event.system = (event.system or "") + block

    api.on("before_agent_start", inject)
```

Rationale for using `session_ready` (not `install`) to populate the cache: `resources_discover` handlers from *other* extensions are not registered until every `install()` has run. Per [extension-as-scenario.md §10b.11](extension-as-scenario.md#10b11-session_ready-event), `session_ready` is the only timing point where every extension's handlers are visible.

### Frontmatter parsing — package vs. hand-roll

`AgentM/src/agentm/core/runtime/resource_loader.py` already has a 30-line in-house `_split_frontmatter` using `yaml.safe_load`. Two options for `core/lib/frontmatter.py`:

**Option A — adopt [`python-frontmatter`](https://pypi.org/project/python-frontmatter/)** (`uv add python-frontmatter`).
- Pros: well-maintained, handles JSON/TOML frontmatter too, used by Pelican/MkDocs ecosystem, ~1k LoC of edge cases we don't have to own.
- Cons: extra dependency for a 30-line problem; pulls `pyyaml` (already a dep) and `python-dateutil`.

**Option B — promote the existing `_split_frontmatter` to public `core/lib/frontmatter.py`**.
- Pros: zero new deps; matches pi's <40-line `frontmatter.ts`; already battle-tested in `resource_loader.py`.
- Cons: need to handle one extra case the current impl doesn't: CRLF normalization (pi does it explicitly). Trivial.

**Recommendation: Option A (`python-frontmatter`).** User-confirmed 2026-05-01: prefer the maintained package over rolling our own. `uv add python-frontmatter`. Wrap it in `core/lib/frontmatter.py` with a single `parse_frontmatter(text) -> tuple[dict, str]` function so the rest of the code base depends on AgentM's surface, not the third-party API directly. Update `harness/resource_loader.py::_split_frontmatter` to call through this wrapper.

### Python-equivalent for Node specifics

| Pi (Node.js) | AgentM (Python) |
|---|---|
| `readdirSync(dir, {withFileTypes: true})`, `entry.isSymbolicLink()` | `os.walk(top, followlinks=True)` + manual `Path.is_symlink()` / `Path.resolve()` cycle guard |
| `ignore` npm package | `pathspec` (preferred) or in-house glob matcher |
| `node:fs` `realpath` for symlink dedup | `Path.resolve(strict=False)` |
| Error swallowing on `readdir` / `readFile` | `try/except OSError` with diagnostic emission |

## Interface Definition

```python
# core/_internal/skills.py
@dataclass(frozen=True, slots=True)
class SkillRecord: ...

@dataclass(frozen=True, slots=True)
class SkillDiagnostic: ...

def load_skills(*, cwd, agent_dir, skill_paths=(), include_defaults=True) -> tuple[list[SkillRecord], list[SkillDiagnostic]]: ...

def format_skills_for_prompt(skills: list[SkillRecord]) -> str: ...

# core/lib/frontmatter.py  (option B; see decision)
def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]: ...

# extensions/builtin/skill_loader.py
MANIFEST: ExtensionManifest
def install(api: ExtensionAPI, config: dict[str, Any]) -> None: ...
```

### `resources_discover` event (Tier 1.5 attachment)

Add to `harness/events.py`:

```python
@dataclass(frozen=True, slots=True)
class ResourcesDiscoverEvent:
    cwd: str
    reason: Literal["startup", "reload"]
```

Handlers return `dict[str, list[str]]` (or None). Recognized keys: `skill_paths`, `prompt_paths`. Extra keys are forwarded to whichever consumer asked. This is the smallest-possible hook for runtime extensibility (see [pluggable-architecture.md §3.4](pluggable-architecture.md#34-resource-discovery-the-project-context-boundary)).

`AgentSession` does **not** call `resources_discover` automatically. Each consumer (skill_loader, prompt_templates) emits it themselves at `session_ready` time. Rationale: keeps the harness ignorant of which resource kinds exist.

## Acceptance Scenarios

1. A skill at `<cwd>/.agentm/skills/refactor/SKILL.md` with valid frontmatter is discovered; `<available_skills>` block contains its name/description/path; system prompt grows by exactly the block size.
2. A skill with empty `description` is dropped; its diagnostic appears in the loader output.
3. A skill with `disable-model-invocation: true` is loaded but excluded from `format_skills_for_prompt`.
4. Two skills with identical `name` (one in user dir, one in project dir) — first wins; collision diagnostic emitted.
5. A symlink loop under `<cwd>/.agentm/skills/` does not hang the loader.
6. A second extension installed alongside `skill_loader` returns `{"skill_paths": ["/tmp/extra"]}` from `resources_discover`; the extra dir's skills appear in the prompt.
7. The `read` tool can read a skill's `references/foo.md` file — i.e., progressive disclosure works end-to-end.

## Related Concepts

- [extension-as-scenario.md](extension-as-scenario.md) — atom contract this loader conforms to
- [pluggable-architecture.md](pluggable-architecture.md) §3.4 — ResourceLoader port
- [prompt-templates.md](prompt-templates.md) — sibling resource kind, similar discovery shape
- [search-tools.md](search-tools.md) — the `read` tool that consumes skill bodies on demand

## Constraints and Decisions

| Decision | Rationale | Alternative |
|---|---|---|
| Skills injected via `before_agent_start` mutation, not via `context` event | `before_agent_start` is the documented system-prompt mutation hook; `context` rewrites messages | Inject via a fake user message — pollutes history |
| Cache built at `session_ready`, not `install` | Other extensions' `resources_discover` handlers aren't visible in `install()` | Eager load in install — misses extension-contributed paths |
| Use `python-frontmatter` package (Option A) | User-chosen 2026-05-01: prefer maintained dep | Hand-roll — small but yet another in-house parser to own |
| Pathspec for `.gitignore` | Closest to pi's `ignore` semantics | Hand-roll a subset matcher |
| Silently load skills with name warnings (only drop on missing description) | Matches pi behavior; avoids breaking users who have lightly-misnamed skills | Strict mode — too brittle for v1 |

## Out of Scope

- Skill versioning, dependency declarations, capability tags.
- Auto-installing skills from a registry.
- Hot reload (the `reload()` ResourceLoader method is wired but not driven by anything yet).
- Per-mode UI for picking skills interactively (TUI work, deferred to mode layer).

## Open Questions

- [ ] Should `resources_discover` results be merged with config-supplied `skill_paths` or override them? (Recommendation: merge.)
- [ ] Do we want a `MAX_SKILLS` ceiling to prevent prompt blowup? (Pi has none. Defer.)
