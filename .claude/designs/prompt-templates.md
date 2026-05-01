# Design: Prompt Templates (Slash-Command Macros)

**Status**: DRAFT
**Created**: 2026-05-01
**Last Updated**: 2026-05-01

## Overview

Filesystem-discovered markdown files where the filename is a slash-command name (`<name>.md` → `/<name>`); when the user types `/<name> args`, the body is loaded, bash-style placeholders (`$1 $@ ${@:N:L} $ARGUMENTS`) are substituted with parsed arguments, and the resulting text is sent as the user prompt. Pi-mono reference: [`packages/coding-agent/src/core/prompt-templates.ts`](/tmp/pi-analysis/pi-mono/packages/coding-agent/src/core/prompt-templates.ts).

## Motivation

AgentM has slash-command dispatch (`api.register_command`) but only for code-driven commands. Users — and other agents — frequently need *text* macros: a saved prompt for "write a PR description" or "investigate this RCA pattern" that takes a few arguments and expands to a long prompt. Without a template system this is copy-paste; with one, a project ships its institutional prompts as files in `.agentm/prompts/`.

## Design Details

### File layout

Same precedent as skills (and the existing `DefaultResourceLoader`):

```
<agent_dir>/prompts/<name>.md      # global, per-user
<cwd>/.agentm/prompts/<name>.md    # project-local
```

Plus extra paths supplied via `config["prompt_paths"]` and via `resources_discover` handler returns.

### File shape

```markdown
---
description: One-line summary shown in `/help`.
argument-hint: "<file> [<dir>]"   # optional; rendered next to the command in completions
---

# Body, free markdown. Substitutions happen *only* on the body.
Refactor $1 in $2 with the following constraints: $ARGUMENTS
```

If `description` is absent, the loader uses the first non-empty line of the body, truncated to 60 chars (port of `prompt-templates.ts:111-119`).

### Argument parsing (port `parseCommandArgs`)

Bash-style tokenization with quote handling:

- Whitespace-separated tokens.
- Single or double quotes group an argument: `/cmd "hello world" foo` → `["hello world", "foo"]`.
- A quote opens; the same quote closes; mismatched closing quotes are tolerated as part of the argument.
- No backslash escaping (matches pi).

### Substitution grammar (port `substituteArgs`)

Applied **in this order** to avoid recursive substitution of placeholder-shaped argument values:

1. `$1`, `$2`, ... — replaced with the corresponding arg (`""` if missing).
2. `${@:N}` and `${@:N:L}` — bash-style slice. `N` is **1-indexed** (matches bash; `${@:0:...}` treated as `${@:1:...}`). `L` is the count.
3. `$ARGUMENTS` — all args joined by single space.
4. `$@` — same as `$ARGUMENTS` (kept for legacy compat / parity with pi).

**Important**: substitution is non-recursive. If `$1` is replaced by a string containing `$2`, that `$2` is **not** further substituted. This is critical for safety (an arg like `'$ARGUMENTS'` won't blow up the body).

### Public API (in `core/prompt_templates.py`)

```python
@dataclass(frozen=True, slots=True)
class PromptTemplateRecord:
    name: str
    description: str
    argument_hint: str | None
    body: str
    file_path: str
    source: str

def parse_command_args(args_string: str) -> list[str]: ...
def substitute_args(body: str, args: list[str]) -> str: ...

def load_prompt_templates(
    *,
    cwd: str,
    agent_dir: str,
    prompt_paths: list[str] = (),
    include_defaults: bool = True,
) -> list[PromptTemplateRecord]: ...

def expand_prompt_template(text: str, templates: list[PromptTemplateRecord]) -> str | None:
    """If `text` is `/<name>` or `/<name> <args>` matching a template,
    return the expanded body. Else return None."""
```

### Extension atom: `extensions/builtin/prompt_templates.py`

```python
MANIFEST = ExtensionManifest(
    name="prompt_templates",
    description="Expand /<name> args templates into a prompt before the agent loop runs.",
    registers=("event:input", "event:resources_discover"),
    config_schema={...},
)

def install(api: ExtensionAPI, config: dict) -> None:
    cache: list[PromptTemplateRecord] = []
    extra_paths = list(config.get("prompt_paths") or [])
    include_defaults = bool(config.get("include_defaults", True))

    async def populate() -> None:
        results = await api.events.emit("resources_discover", ResourcesDiscoverEvent(cwd=api.cwd, reason="startup"))
        for r in results:
            if isinstance(r, dict):
                extra_paths.extend(r.get("prompt_paths") or [])
        cache[:] = load_prompt_templates(
            cwd=api.cwd, agent_dir=str(Path.home() / ".agentm"),
            prompt_paths=tuple(extra_paths), include_defaults=include_defaults,
        )

    api.on("session_ready", lambda _e: populate())

    def maybe_expand(event: InputEvent) -> dict | None:
        text = event.text or ""
        if not text.startswith("/"):
            return None
        expanded = expand_prompt_template(text, cache)
        if expanded is None:
            return None
        event.text = expanded   # mutate in place; later `input` handlers see expanded text
        return None

    api.on("input", maybe_expand)
```

### Dispatch ordering vs. `CommandSpec`

`AgentSession.prompt()` already short-circuits on `text.startswith("/")` and dispatches via the slash-command runner ([extension-as-scenario.md §10b.4](extension-as-scenario.md#10b4-slash-command-runner)). The order must be:

1. **Code-registered commands** (`api.register_command`) — exact name match wins; runs handler instead of agent loop.
2. **Prompt templates** — only if no code command matched. The template is expanded *into the user text*; then the normal agent loop runs.
3. **Fall-through** — text starting with `/` that matches neither is sent to the agent verbatim.

To make this work, `AgentSession.prompt` must:

- After the code-command lookup misses, emit `input` event with the original text. The `prompt_templates` extension's `input` handler does the expansion (mutates the text in place / returns the new text via the event-result protocol).
- Use the (possibly mutated) text as the user message.

**This requires no kernel change** — it's a pure ordering decision in the existing slash-command runner. The plan task captures it as a one-line addition: "before falling through to the agent loop with `/`-prefixed text, run `input` event handlers."

If a template name collides with a registered code command, the code command wins (matches pi behavior — code is more specific than data).

## Interface Definition

See API block above. Everything is in stdlib + `agentm.core.kernel` + `agentm.harness.events` + `agentm.harness.extension`.

## Acceptance Scenarios

1. `/<cwd>/.agentm/prompts/refactor.md` with body `Refactor $1 then explain $@` → `prompt("/refactor foo.py 'why this matters'")` results in user message `Refactor foo.py then explain foo.py why this matters`.
2. `${@:2}` returns args from index 2 onward (1-indexed); `${@:2:1}` returns exactly one arg.
3. `parse_command_args('"hello world" foo')` → `["hello world", "foo"]`.
4. A template named `refactor` does not shadow a code-registered `/refactor` command.
5. `/unknown thing` (neither template nor code command) is sent verbatim to the agent loop.
6. An argument value containing `$1` is **not** recursively substituted.
7. Frontmatter parsing failure does not crash the loader; the offending file is skipped silently (matches pi).
8. Symlink to a parent dir does not infinite-loop.

## Related Concepts

- [skills.md](skills.md) — sibling resource kind; shares `resources_discover` event and frontmatter parser
- [extension-as-scenario.md](extension-as-scenario.md) §10b.4 — slash-command runner ordering this design plugs into
- [pluggable-architecture.md](pluggable-architecture.md) §3.4 — ResourceLoader port

## Constraints and Decisions

| Decision | Rationale | Alternative |
|---|---|---|
| Code commands beat templates on name collision | Code is more specific; matches pi | Template wins — surprising for users who registered a programmatic command |
| Substitution is non-recursive | Safety: arg values can't inject placeholders | Recursive — opens injection footguns |
| Templates expand via `input` event mutation | Keeps the ordering rule in the existing runner; no new kernel hook | New `before_input_dispatch` event — bigger surface for one use case |
| `$ARGUMENTS` and `$@` are aliases | Cross-tool compatibility (Claude/Codex/OpenCode use `$ARGUMENTS`; bash uses `$@`) | Drop `$@` — breaks legacy templates |

## Out of Scope

- Per-template tool/model overrides (pi's frontmatter accepts `model:`; deferred until AgentM has multi-model selection).
- Conditional / templating-engine bodies (Jinja2). The whole point is to be tiny.
- Argument validation against `argument-hint`. The hint is a UI affordance only.

## Open Questions

- [ ] Should the expansion be visible in the session log as the *original* `/cmd args` or the *expanded* text? Recommendation: log both (entry payload carries `raw` and `expanded`).
- [ ] Do we need a `model:` frontmatter key for per-template model selection? Defer until multi-model arrives.
