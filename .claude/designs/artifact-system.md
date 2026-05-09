# Artifact System

Status: design (not yet implemented)
Owner: harness layer + new builtin extension

Reaches into: `extensions/builtin/artifact_store.py` (new), session-tree
state, optional integration with `sub_agent_lifecycle` notification format,
optional integration with `evolution_substrate` tier-3 vault.

## Problem

Today every sub-agent invocation in AgentM follows the same shape:

1. Parent dispatches a worker with a freeform prompt.
2. Worker runs N tool calls of real investigation, accumulating intermediate
   findings in its private message history.
3. Worker emits one final assistant message; harness extracts the text.
4. Worker session is destroyed.
5. Parent receives the text as a tool result (or, with sub_agent_lifecycle, as
   a notification block before agent_end).

Every step from (3) onward is **lossy compression**. The worker's
intermediate reasoning, the tool results it cited, the queries it tried and
abandoned — all of it is collapsed into one prose blob. The parent has no way
to ask "show me the SQL that backed this finding" or "what queries did you
abandon and why?". Once the child shuts down its message history is gone.

The literature names this exactly: "Every hand-off between agents puts your
workflow's shared memory at risk. When one model's reply exceeds another's
context window, critical details vanish, and the next agent starts reasoning
from a partial snapshot." Field studies of multi-agent failure traces
(MAST taxonomy, 1600+ traces) attribute **36.9% of failures to inter-agent
misalignment** rooted in inconsistent shared state.

The operational footprint in rca:

- Scout finds 5 services on the abnormal call chain. Returns 5701-character
  summary. Verify worker, dispatched two turns later, must rederive every
  topology query because scout's actual SQL is gone.
- Orchestrator's context grows linearly with worker count.
- Coordinator and workers cannot peer at each other's intermediate notebook
  pages — every cross-reference round-trips through the orchestrator's text.

The fix is not "longer summaries" or "smarter compression." It is **artifacts
as first-class objects, references as the handoff currency.**

## Principle: filesystem is the store

Claude Code's lesson is the relevant one: when a subagent needs to hand off
non-trivial output, the simplest correct mechanism is **write a file, return
the path**. No database, no serialization layer, no cross-process schema —
the OS already provides atomic writes, listing, grep, content-addressing
via the path namespace, and post-mortem inspection (you can `cat` the
file).

We adopt the same shape. Artifacts are files under
`<cwd>/.agentm/artifacts/<root_session_id>/`. Tools read, write, list, and
grep them. Provenance lives in a small `.meta.json` sidecar per artifact.
There is no SQLite, no FTS, no ORM.

This is also a hybrid memory architecture in the literature's vocabulary:

- **Tier 1 — private**: each session's own message history (already exists).
- **Tier 2 — team shared (this design)**: artifact directory, session-tree-scoped.
- **Tier 3 — long-term**: cross-run vault (deferred to `evolution_substrate`).

## Layout

```
<cwd>/.agentm/artifacts/<root_session_id>/
├── art_001__topology__abnormal-call-graph.md
├── art_001__topology__abnormal-call-graph.meta.json
├── art_002__query_result__service-span-counts.md
├── art_002__query_result__service-span-counts.meta.json
├── art_003__finding__ts-auth-disappearance.md
├── art_003__finding__ts-auth-disappearance.meta.json
└── art_004__brief_rejection__missing-scope.md
    ...
```

File-name structure: `<artifact_id>__<kind>__<slugified-title>.md`. The
filename is itself a partial index — `ls .agentm/artifacts/<root>/ | grep
__topology__` enumerates all topology artifacts without opening the
sidecar files.

The `.meta.json` carries the provenance:

```json
{
  "artifact_id": "art_001",
  "parent_id": null,
  "kind": "topology",
  "title": "Abnormal call graph (DOT format)",
  "tags": ["scout", "round1"],
  "parent_artifact_ids": [],
  "created_by": {
    "session_id": "sess_abc",
    "task_id": "task_xyz",
    "persona": "scout",
    "timestamp": 1740000000.0
  }
}
```

The artifact body is a plain UTF-8 file. `.md` is the default extension
because most artifacts are prose findings; binary artifacts are out of
scope (text only, same as the original constraint).

Append-only: the writer never modifies an existing file. Revisions are new
artifacts that reference the predecessor via `parent_id`. Deletion is also
out of scope — operators can `rm` the directory after the root session ends
if retention pressure demands it.

## Storage backend

Plain filesystem. No database, no library dependency.

- `artifact_id` allocation: monotonically incremented per root session,
  zero-padded (`art_001`, `art_002`, ...). The store keeps a tiny
  `.next_id` file in the directory; allocation is a read-modify-write under
  a `fcntl.flock` (or asyncio lock — workers are in-process today).
- Atomic writes: write to `<id>__<kind>__<slug>.md.tmp`, then `os.rename`
  to the final name. POSIX rename is atomic; readers never see partial
  files.
- Listing: `os.scandir` of the directory.
- Filtering by `kind`: filename prefix match (`art_*__topology__*.md`).
- Searching by content: `re.search` over each candidate file's body. We
  rely on the OS page cache; for the artifact volumes we expect (tens to
  hundreds per session), a linear scan is fast enough. If a scenario ever
  produces thousands, the user can `grep -l` from a shell — same shape.

Cross-root sharing is not supported. Different root sessions get
different directories.

## Inheritability

`artifact_store` registers as an inheritable extension in the same
`available_inherited_extensions` map `sub_agent` already uses. When a
parent dispatches a child, the child's `artifact_store` install reads the
parent's `root_session_id` from the dispatch metadata and opens the
**same directory**. All descendants of one root share one folder.

The live store object is per session and is exposed through
`ExtensionAPI.set_service("artifact_store", store)`. Peer extensions resolve it
with `api.get_service("artifact_store")`; there is no module-level session
registry. This is the only piece of cross-session coupling. Everything else is
read/write of files in a known directory.

## Tools registered (one extension)

A single new builtin: `extensions/builtin/artifact_store.py`. Tier 1
(no permission). Inheritable into worker sessions.

**`artifact_write({kind, title, body, tags?, parent_artifact_ids?}) -> {artifact_id, path}`**

Append-only. Allocates the next id, writes body atomically to
`<id>__<kind>__<slug>.md`, writes the sidecar. Provenance is captured
automatically from the session context. Returns id and absolute path.

**`artifact_read({artifact_id, range?}) -> {kind, title, created_by, body}`**

Returns the content. **`range` parameter is mandatory in spirit, optional
in syntax**:

- omitted → returns body if it's under a per-call cap (default 8 KiB),
  else returns the first 8 KiB plus a truncation marker with byte offsets
  and a recommendation to call again with `range`.
- `{lines: [start, end]}` → 1-indexed, end-inclusive line slice.
- `{bytes: [start, end]}` → byte slice for non-line-oriented bodies.
- `{head: N}` / `{tail: N}` → first/last N lines.

Without bounded reads, an artifact system delays context bloat instead
of preventing it; the cap is load-bearing.

**`artifact_list({kind?, tags?, created_by_task?, since?, limit?}) -> {artifacts: [{id, kind, title, path, created_by, tags, size_bytes}]}`**

Lightweight listing — no `body`. Filters by frontmatter fields. Default:
latest 50, sorted by timestamp descending. Implementation reads
`.meta.json` files; for typical session sizes (≤ a few hundred artifacts)
this is microseconds.

**`artifact_grep({pattern, kind?, tags?, max_hits?, snippet_lines?}) -> {hits: [{id, title, line_no, snippet}]}`**

Linear regex scan over filtered artifacts. Returns matched lines with
configurable surrounding context (default 2 lines each side). Bounded by
`max_hits` (default 20) to prevent runaway results blowing context.

The four-tool surface mirrors `tool_read` / `tool_grep` / `tool_ls` from
the file-system world. Deliberately no `artifact_edit` / `artifact_delete`.

## Worker return convention

Combined with `sub_agent_lifecycle`, the notification block format is
extended:

```xml
<subagent_result task_id="abc123" purpose="Map service topology">
  <summary>Topology mapped. 5 services on abnormal chain. ts-auth absent
  in abnormal period (was 30K spans normal); ts-verification-code took
  its place (34K spans abnormal). Resource scan shows ts-route-service
  CPU 6.5x normal.</summary>
  <artifacts>
    <ref id="art_001" kind="topology" title="Abnormal call graph (DOT format)" />
    <ref id="art_002" kind="query_result" title="Service span counts (abnormal vs normal)" />
    <ref id="art_003" kind="finding" title="ts-auth-service disappearance evidence" />
  </artifacts>
</subagent_result>
```

The orchestrator gets a tight summary plus a list of artifact references.
To inspect any one in detail it calls `artifact_read(id, range=...)`.
To enumerate everything a particular worker produced it calls
`artifact_list(created_by_task=task_id)`.

The `<summary>` is what the worker chose to highlight; the artifacts are
what it actually produced. They can disagree, and the orchestrator
resolves discrepancies by reading the artifact.

## Persona file convention

Worker personas (`agents/<name>.md`) gain an optional frontmatter field:

```yaml
artifact_kinds: [finding, query_result, trace]
```

Documented expectation only — no runtime enforcement. The persona body
should reinforce in prose what kinds it produces.

## Integration with sub_agent_lifecycle

`_ChildTask` gains a fourth state field:

```python
@dataclass
class _ChildTask:
    ...
    summary: str | None = None
    artifact_ids: list[str] = field(default_factory=list)
```

When a worker calls `agent_end`:

1. The harness extracts its final assistant message text → `summary`.
2. The artifact directory is scanned for sidecars with
   `created_by.task_id == task_id` → `artifact_ids`.
3. `decide_turn_action` (parent side, on a voluntary `Stop(ModelEndTurn)`)
   builds the notification block from these two fields, not from the full
   `final_messages` history, and returns it via `Inject(messages=[...])`.

If a worker writes zero artifacts, the block degrades to the
`sub_agent_lifecycle.md` baseline (just `<summary>` text). If a worker
writes artifacts but emits no final text, the block carries only
`<artifacts>`. Both ends are graceful.

## Failure modes addressed

| Failure mode (literature) | Defense (this design) |
|---|---|
| Context loss at handoff | Worker's full work survives in artifact files; not compressed into final_text |
| Cascade failure / context poisoning | Artifacts immutable; later workers cannot retroactively corrupt earlier ones |
| Communication overhead (re-explain) | Workers read each other's artifacts directly, no orchestrator round-trip |
| Summarization drift | Original artifacts persist alongside any summary that references them |
| Supervisor context bloat | Notification block is summary + ids; bounded `artifact_read` prevents blow-up |
| Inconsistent shared state | Append-only model; provenance lets later readers detect contradictions |

What this design does **not** address:

- Inter-agent disagreement / arbitration: still the orchestrator's job.
- Long-running cross-session evolution: tier 3 / `evolution_substrate`.
- Binary artifacts: out of scope (text only).
- ACL / privacy: every session in the tree sees every artifact. Use a
  separate root session for confidential work.
- Garbage collection: operator-driven `rm -rf .agentm/artifacts/<old_root>/`.
  Optional time-based sweep can be a follow-up.
- Concurrent writers from multiple processes: today's `sub_agent` runs
  children in-process, so an asyncio.Lock around id allocation suffices.
  If we ever support out-of-process workers, swap in `fcntl.flock`.

## Compatibility and migration

- Backward compatible: scenarios that don't load `artifact_store` continue
  to work exactly as today; sub_agent_lifecycle notification format
  degrades gracefully when `artifact_ids` is empty.
- The `inherit_extensions` mechanism already handles propagating into child
  sessions.
- Persona files without `artifact_kinds` frontmatter remain valid.
- No data migration; the directory is created on first write.
- Operator post-mortem: `cd .agentm/artifacts/<root>; ls; cat art_003*.md`
  works without any tooling.

## Effort estimate

- `extensions/builtin/artifact_store.py`: ~150 lines (filesystem layer +
  4 tools + MANIFEST + session-ready handler). Lower than the SQLite
  version because there is no schema, no FTS5 fallback, no ORM.
- `_ChildTask` extension in sub_agent: ~30 lines.
- Notification block formatter update: ~20 lines in `sub_agent` itself.
- rca scenario update: replace `update_hypothesis` / `remove_hypothesis`
  with thin wrappers over `artifact_write(kind="hypothesis", ...)`,
  preserving the existing tool API for compatibility.

Total: roughly 200–250 lines of code plus this design doc.

## Open questions

1. **Sharing across root sessions.** RCA on dataset A and RCA on dataset B
   today produce two independent root sessions. Should they share a
   "RCA technique vault"? Probably yes, but as tier-3 evolution_substrate.

2. **Artifact references in artifact bodies.** `body` is plain text. If a
   worker writes "see art_002 for details" we are not enforcing the link
   structurally — only via the explicit `parent_artifact_ids` field.
   Tradeoff: enforcing creates a typed graph but pushes complexity into
   tooling. Keep loose for v1.

3. **Streaming artifacts.** A worker midway through a long query result
   may want to commit a partial artifact. `artifact_write` is currently
   one-shot. Streaming append would require a write-handle abstraction;
   filesystem actually makes this trivial (open file in append mode), but
   we defer until there is a concrete use case.

4. **Slug collisions.** Two artifacts with title "Service span counts"
   produce different ids but the same slug suffix; the id prefix
   disambiguates the filename. No collision possible.
