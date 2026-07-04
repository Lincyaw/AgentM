# Design: Slash Command Routing

**Status**: PROPOSED
**Created**: 2026-05-11
**Reference**: Claude Code's slash command system (`/clear`, `/compact`,
`.claude/commands/*.md`, `.claude/skills/*/SKILL.md`).

---

## 1. Purpose

When a user types `/something` to the gateway, the gateway must
**intercept the message before it reaches the LLM** and either change
agent state or expand a templated prompt. This is the same idea Claude
Code calls "slash commands"; we adapt it to AgentM, where atoms and
skills are explicit kernel concepts.

The key product motivation: today the only way for a user to influence
a running session is to ask the LLM in natural language. That is
expensive, ambiguous, and uncontrolled. A user who wants to start a
fresh conversation should not need the LLM to understand they want a
fresh conversation — they should type `/new`.

---

## 2. First Principles

1. **Slash commands never reach the LLM as commands.** A command the
   *gateway* registry doesn't own is routed to a live session before the
   gateway decides whether it is unknown. The gateway reads that session's
   user-invokable command set and calls `AgentSession.prompt(raw_slash_text)`
   when the name is present, so the in-session `slash_commands` atom dispatches
   session-registered commands (`/compact`, `/goal`, `/loop`, etc). Only a
   name unknown to *both* layers is rejected with a user-visible reply rather
   than forwarded. The cost of "user typo accidentally sends private prompt to
   model" is too high to forward a name nobody owns; but a real session command
   must reach its dispatcher, not be rejected at the gateway.
2. **One protocol, four handler kinds.** Commands come from four
   sources (builtin control, markdown prompt, skill, atom), but the
   dispatcher sees a single `CommandHandler` Protocol. The router is
   the mechanism; each kind is a policy.
3. **Discovery is data-driven where possible.** Builtin control
   commands are Python (they need executable side effects); the other
   three kinds are *generated* from data already on disk (`*.md`
   files, SKILL.md frontmatter, atom MANIFEST).
4. **Namespacing makes provenance visible.** `/new` is builtin; every
   non-builtin command is `/<namespace>:<name>` so `/help` users can
   immediately tell where a command came from.

---

## 3. Four handler kinds

| Handler | namespace | kind | Source | Behavior |
|---|---|---|---|---|
| `BuiltinControl` | `None` | `control` | Python file in `commands/builtins/` | Mutates gateway/session state, may emit `OutboundMessage`s. **No LLM call.** |
| `MarkdownPrompt` | `None` | `prompt` | `<cwd>/.agentm/commands/*.md` | Expands frontmatter+body with `$ARGUMENTS` substitution. Falls through to `session.prompt(expanded)`. |
| `SkillCommand` | `"skill"` | `prompt` | Skill directories (gateway config + `.agentm/skills` + `.claude/skills`) | Reads `SKILL.md`, expands into a templated prompt that injects the body for the current turn. |
| `AtomCommand` | `"atom"` | `control` | Atoms with `mountable_via_command: true` | Calls `api.extensions.install/uninstall` against the running session's runtime scenario overlay (see §7). |

All four implement the same Protocol:

```python
class CommandHandler(Protocol):
    name: str                                          # "new" / "feishu-cli" / "permission"
    namespace: str | None                              # None / "skill" / "atom" / "plugin-x"
    summary: str                                       # one-line, shown in /help
    kind: Literal["control", "prompt"]
    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult: ...
```

```python
@dataclass(frozen=True, slots=True)
class CommandInvocation:
    raw: str                                           # original message content
    name: str                                          # "new" or "feishu-cli"
    namespace: str | None                              # None or "skill"
    args: str                                          # everything after the first token
    inbound: InboundMessage                            # source message

@dataclass(frozen=True, slots=True)
class CommandResult:
    # control:
    outbound: list[OutboundMessage] = ()
    side_effect: Callable[[Gateway], Awaitable[None]] | None = None
    # prompt (mutually exclusive with above):
    expanded_prompt: str | None = None
```

`CommandContext` is the narrow facade the router gives each handler.
Mirrors the §11 `ExtensionAPI` pattern — handlers do not see `Gateway`
directly so they cannot reach into route internals:

```python
@dataclass(frozen=True, slots=True)
class CommandContext:
    route_key: str
    channel: str
    chat_id: str
    sender_id: str
    drop_route: Callable[[], Awaitable[None]]              # for /new, /end
    get_route_stats: Callable[[], dict[str, Any]]          # for /status
    list_commands: Callable[[], list[CommandHandler]]      # for /help
    approval_bridge: ApprovalBridge                        # for /approve, /deny
```

Session-owned commands use the SDK's `CommandSpec`, not gateway
`CommandHandler`. `CommandSpec.user_invokable` defaults to `True` and is a
catalog/UX contract: session snapshots and `session_ready` advertise only
those commands, and the gateway only auto-routes unknown gateway commands to
session command dispatch when the slash name is in that advertised set. The
flag is not a security boundary inside the session runtime.

Before the first live session exists, the gateway welcome capability block
pre-advertises command names declared as `command:*` in the selected scenario's
atom `MANIFEST.registers`. This is intentionally a discovery hint for chat
clients: the authoritative list still comes from the live session's
`CommandSpec` registry once `session_ready` arrives.

---

## 4. Parsing

Trivial: a message is a command iff `content.startswith("/")` and
`button_value is None`. The first token (split on whitespace) is the
name; everything after is `args` (preserved verbatim, not split).
Namespace separator is `:`, exactly one (`/skill:feishu-cli foo bar`
→ namespace=`"skill"`, name=`"feishu-cli"`, args=`"foo bar"`).

Edge cases:

- `/` alone → invalid; reply with "Type `/help` for commands."
- `/unknown` → if the gateway registry doesn't own the name, the router
  returns `None` and the gateway ensures a session exists, reads its
  user-invokable command set, and calls `session.prompt(raw)` only when that
  name is present. Otherwise the gateway replies "no such command" — **no
  fallback to the LLM for a name nobody owns.**
- `//literal/path` → not a command (first character is `/` but second
  is `/`; treated as user text). Cheap to disambiguate without
  surprising file-path users.

---

## 5. Discovery and registration order

At `Gateway.start()` (or first access), `CommandRouter._scan()` builds
the registry:

1. **Builtin Python** — `commands/builtins/*.py`, each exporting a
   `HANDLER` module-level attribute (or `BUILTINS: list[CommandHandler]`
   for multi-handler files).
2. **Markdown prompt** — `<cwd>/.agentm/commands/*.md`, frontmatter
   `name` / `summary` / optional `args` field.
3. **Skill** — `api.skills.load_skills(...)` would be ideal but is not
   reachable from outside the session; instead the router walks explicit
   gateway-configured paths first, then the project/user AgentM skill
   directories (`<cwd>/.agentm/skills/`, `$AGENTM_HOME/skills/`) and
   Claude-compatible directories (`<cwd>/.claude/skills/`,
   `~/.claude/skills/`). Each skill becomes one `SkillCommand` in namespace
   `skill`.
4. **Atom** — at session-construction time, the runtime scenario
   overlay (see §7) lists mountable atoms. Their MANIFEST entries
   surface as `AtomCommand` instances in namespace `atom`. Deferred to
   PR #2; the design is locked here.

Welcome-time capability discovery also inspects the selected scenario's atom
`MANIFEST.registers` for `command:*` tags so a terminal can show `/goal`,
`/loop`, and similar session commands before the user's first prompt creates a
session. It does not install the atom or execute command handlers; it only
imports manifests and emits names/summaries as hints.

**Collision policy**: priority is builtin > markdown > skill > atom.
A skill named `new` does *not* shadow `/new`. `/help` shows every
discovered command with its source; shadowed commands are listed once
under their effective source plus a `(also: skill)` note.

`/help` is the only place the user sees the full registry. The router
**does not** auto-rescan; restart (or `/help --rescan`, future) picks
up new markdown files / skills. This is a deliberate trade-off — live
filesystem watching is more code than it earns at this stage.

Gateway-level skill-command paths are configured in
`$AGENTM_HOME/config.toml`:

```toml
[gateway.commands.skills]
paths = ["~/.codex/skills", "~/.agents/skills"]
```

Those paths are opt-in because they may expose personal command surfaces to
every gateway peer attached to the daemon.

---

## 6. Dispatch flow inside `Gateway._dispatch`

```python
async def _dispatch(self, msg: InboundMessage) -> None:
    if msg.button_value:
        ... # approval routing
        return

    if msg.content.startswith("/") and not msg.content.startswith("//"):
        result = await self._command_router.try_dispatch(msg, ctx=...)
        if result is None:
            # No GATEWAY handler owns this name. Ensure the route has a live
            # session, read its user-invokable CommandSpec names, and forward
            # only if the slash name is a known session command.
            sess = await self._get_or_create_session_for_input(msg)
            if slash_name(msg) in sess.command_names:
                await sess.prompt(msg.content)         # session dispatches it
            else:
                await self._publish_unknown_command(msg)
            return
        for out in result.outbound:
            await self._bus.publish_outbound(out)
        if result.side_effect is not None:
            await result.side_effect(self)
        if result.expanded_prompt is None:
            return                                     # control: done
        msg = replace(msg, content=result.expanded_prompt)
        # fall through to normal session.prompt path

    route = await self._get_or_create_route(msg)
    ...
```

Two invariants:

1. `result is None` means no gateway handler owns the name. The gateway
   resolves a live session and forwards the raw slash text only when that
   session advertises the command as user-invokable, so a session-registered
   command runs through `slash_commands`. A name nobody owns never reaches the
   LLM silently as a command.
2. `result.expanded_prompt is not None` means it is a prompt command;
   the gateway re-enters the normal path with rewritten content. The
   session sees the expanded text as if the user had typed it.

---

## 7. Skill-as-command: the activation contract

A user sends `/skill:feishu-cli list group members`. The handler:

1. Locates `SKILL.md` at the path recorded during discovery.
2. Reads body (skipping YAML frontmatter).
3. Returns:

   ```
   expanded_prompt = (
       "<<system>>You are now operating with the following skill loaded "
       "for this turn. Follow its guidance.\n\n"
       "--- SKILL: feishu-cli ---\n"
       "{body}\n"
       "--- end skill ---\n"
       "<<user>>{args}"
   )
   ```

   The actual prefix/suffix wording is defined in
   `commands/skill_command.py` so it can iterate without touching
   every skill.

4. The gateway sends `expanded_prompt` to `session.prompt(...)`. The
   skill content lives in **one user-turn message**, not in the
   system prompt — this avoids permanently inflating the session's
   prompt cache and keeps the skill scoped to this single turn.

**Why one turn?** Default is one-shot to keep semantics simple. A
"sticky skill" variant (`/skill:foo --pin`) that re-injects the body
on each subsequent turn is plausible but adds route state we do not
need yet. Listed as P2 in the plan.

**Auto-activation vs explicit activation.** AgentM already has the
`skill_loader` atom that lists every available SKILL.md in the system
prompt so the LLM can choose when to "load" one based on context.
That mechanism is unchanged. `/skill:foo` is the *explicit override*:
when the LLM has not picked the right skill, the user can force it.

---

## 8. Atom-as-command (PR #2 design lock)

### 8.1 MANIFEST extension

```python
@dataclass(frozen=True, slots=True)
class ExtensionManifest:
    name: str
    description: str
    registers: tuple[str, ...] = ()
    config_schema: dict[str, Any] = ...
    requires: tuple[str, ...] = ()
    mountable_via_command: bool = False              # NEW, default False
```

`False` default = strictly backwards compatible. Existing atoms behave
exactly as before. An atom author opts in by setting `True` *and*
documenting any safe-to-invoke config defaults.

### 8.2 Gateway whitelist

In gateway YAML:

```yaml
commands:
  atoms:
    enabled: false                # master switch, default off
    allow: [permission, cost_budget]   # only listed atoms surface as /atom:* commands
```

Even if an atom opts in via MANIFEST, deployment must allow it. Both
gates are required for `/atom:install <name>` to be discoverable.

### 8.3 Runtime overlay: `<cwd>/.agentm/atoms/`

The "runtime directory" the gateway should mutate is not a new
construct — it is the AgentM SDK's existing convention:
**`ExtensionAPI.install_atom` writes installed-atom source to
`<cwd>/.agentm/atoms/<name>.py`** via `ResourceWriter` (git-committed
through the constitution-aware path), and the user-atom
auto-discovery at session start
(`discover_user_atoms`) re-mounts everything under that directory.

Consequences for ``/atom:*``:

1. The original scenario manifest under `contrib/scenarios/...` is
   **never modified**. Source files stay read-only. ✓
2. Source-of-truth catalog locations
   (`src/agentm/extensions/builtin/...`, `contrib/extensions/...`)
   are also untouched — the install reads atom source from there but
   writes a copy into `<cwd>/.agentm/atoms/`.
3. Across daemon restarts, the next session auto-rediscovers files in
   `<cwd>/.agentm/atoms/` and re-mounts them.
4. Per-chat isolation. Tracked but not implemented — today every
   chat sharing a gateway-`cwd` shares the runtime atoms directory.
   When per-chat isolation matters, run separate gateway processes
   with separate `cwd`s. A future per-route subdirectory layered onto
   `discover_user_atoms` would close this, but is out of scope here.
5. `/atom:reset` (future) clears `<cwd>/.agentm/atoms/`.

`AtomCommand.handle` for `install`:

```python
api = ctx.get_extension_api()          # ExtensionAPI for this route's session
src = Path(_resolve_source(atom_name)).read_text()
result = api.install_atom(
    name=atom_name,
    source=src,
    target_path=None,                  # → <cwd>/.agentm/atoms/<name>.py
    config=user_config,
    rationale=f"User {sender_id} invoked /atom:install {atom_name}",
    agent_initiated=False,             # user-initiated, not the LLM
)
```

`AtomCommand.handle` for `uninstall` calls `api.unload_atom` — the
on-disk source file is intentionally left in place (it can be
`/atom:install`'d again without re-resolving the source). To wipe the
runtime overlay completely the operator removes
`<cwd>/.agentm/atoms/` manually.

### 8.4 What stays off the table

- Installing arbitrary code paths (`pip install foo`) via chat. Atoms
  must already exist in the catalog (`agentm.extensions.discover`
  must already find them).
- Modifying `core/` paths. The SDK's `is_constitution_path` validator
  rejects this regardless of how the call gets in.
- Per-user atom mounts within a shared chat. `install_atom` is scoped
  to the route's `AgentSession`, but the *on-disk* overlay is the
  gateway-wide `<cwd>/.agentm/atoms/` — see point 4 above.

---

## 9. /help rendering

```
Commands (12)

control
  /new           start a fresh session for this chat
  /end           shut down this chat's session
  /status        session id, turn count, pending approvals
  /help          this message

prompt (markdown, from .agentm/commands/)
  /standup       generate today's standup

skill (5)
  /skill:feishu-cli    operate Feishu via lark-cli
  /skill:pdf           PDF read / edit / merge
  …

atom (2, disabled by config)
  /atom:install <name> [config-json]
  /atom:list           installed atoms in this session

session (1)
  /compact
```

Grouping by source makes provenance immediately legible. Disabled
groups (e.g. `atom` if `commands.atoms.enabled: false`) are still
shown with a `(disabled by config)` note so users understand why the
command does not work. The `session` group lists commands registered
*inside* the session (dispatched by the in-session `slash_commands`
atom, e.g. `/compact`) — the gateway learns their bare names from the
`session_ready` frame and folds them in here so they are discoverable;
they carry no gateway-side summary.

---

## 10. Comparison with Claude Code

| Aspect | Claude Code | AgentM |
|---|---|---|
| Markdown commands | `.claude/commands/*.md` | `<cwd>/.agentm/commands/*.md` (different dir to keep Claude Code projects untouched) |
| Skill activation | `Skill` tool + `/<skill-name>` | `SkillCommand` + the existing `skill_loader` atom |
| Plugin namespace | `/plugin:command` | Same shape |
| Builtin control | hardcoded | `commands/builtins/*.py`, registered like atoms |
| Unknown command | reject | forward to session if a session-registered command owns it; reject only when unknown to both gateway and session |
| Kernel mutation | `Skill` tool + plugin system (opaque) | `AtomCommand` over MANIFEST-opt-in atoms with runtime overlay (transparent and auditable) |

The biggest divergence is `/atom:*`. Claude Code conflates "skill" and
"agent" and "plugin" at the user surface; AgentM keeps the kernel
concept (atom = single-file extension with MANIFEST) explicit. This
trades surface uniformity for auditability — every atom mount produces
a catalog event under `self_modifiable_architecture`.

---

## 11. Phasing

| Phase | Scope | Status |
|---|---|---|
| PR #1 | router + protocol + builtin control (`/new`, `/end`, `/status`, `/help`) + markdown prompt + `SkillCommand` | This PR |
| PR #2 | `AtomCommand` + MANIFEST `mountable_via_command` + runtime scenario overlay + `/atom:install` / `/atom:uninstall` / `/atom:list` / `/atom:reset` | Next |
| P2 | Sticky skills (`--pin`), live filesystem rescan, `/approve` / `/deny` text fallback, namespace aliases (no-prefix shortcuts) | Future |

---

## 12. Cross-references

- `gateway-channels.md` — the gateway layer commands ride on top of.
- `pluggable-architecture.md` — `ExtensionAPI` is the same facade
  pattern as `CommandContext`.
- `self-modifiable-architecture.md` — atom-as-command sits on top of
  the catalog + transactional reload that already exists.
