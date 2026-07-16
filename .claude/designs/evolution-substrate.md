# Evolution Substrate

Status: current.

## 1. Purpose

The evolution substrate binds every observed run to the exact atom source that
produced it. Its job is identity and evidence attribution, not optimization
policy.

Three invariants are load-bearing:

1. One source byte sequence has one deterministic version id.
2. A trace fingerprint never points at a version whose source cannot be
   recovered and verified.
3. Derived metrics and run links are rebuildable from raw observability.

`compute_atom_hash(source)` is the only atom version function. It returns the
first 12 lowercase hexadecimal characters of SHA-256 over UTF-8 source.

## 2. Ownership and boundaries

| Responsibility | Owner |
|---|---|
| Emit exact loaded-set fingerprints | `observability` atom |
| Freeze immutable source and manifest snapshots | runtime catalog |
| Validate and browse snapshots | `core._internal.catalog` through `CatalogService` |
| Attribute traces and metrics | runtime catalog indexer |
| Expose read and rollback tools | contrib `tool_catalog` atoms |

`.agentm/catalog/**` is constitution-protected. Normal `ResourceWriter`
implementations reject direct writes there. Only catalog mechanism writes
snapshot and derived-attribution files.

`ResourceWriter` is a mutation transport, not a version store. It reports path
classification and errors only. Atom reload/install owns transactional source
snapshots, byte verification, runtime rollback, and catalog freezing.

## 3. On-disk contract

```text
.agentm/catalog/
└── atoms/
    └── <atom_name>/
        ├── current
        └── <content_hash>/
            ├── source.py
            ├── manifest.json
            ├── metrics.jsonl
            ├── runs/
            │   └── <trace_id> -> observability trace
            └── decisions.jsonl
```

`current` is a text file containing one content hash and a trailing newline.
It is replaced atomically after the immutable snapshot is complete.

### 3.1 Immutable snapshot

`source.py` is byte-identical UTF-8 source.

`manifest.json` is a canonical JSON projection of `ExtensionManifest`:

```json
{
  "name": "tool_read",
  "description": "Read files",
  "registers": ["tool:read"],
  "config_schema": "package.ReadConfig",
  "requires": [],
  "conflicts": [],
  "api_version": 1,
  "affects": [],
  "tier": 1,
  "mountable_via_command": false,
  "provides_role": [],
  "effects": {},
  "content_hash": "e5f6a1b2c3d4"
}
```

Freeze is idempotent only when existing bytes match exactly. A mismatching
`source.py` or `manifest.json` at an existing content hash is corruption, not a
new version and not a recoverable compatibility case.

### 3.2 Validated reads

Every source/manifest read performs all of these checks before returning data:

- atom name is a Python identifier;
- version is exactly 12 lowercase hex characters;
- both snapshot files exist;
- source is UTF-8;
- recomputed source hash equals the directory name;
- manifest is a JSON object;
- manifest `name` equals the atom directory;
- manifest `content_hash` equals the version directory.

`list_versions` validates every version directory rather than silently hiding a
damaged entry. `current_version` validates the pointer and its target snapshot.

### 3.3 Derived files

`metrics.jsonl`, `runs/`, and `decisions.jsonl` are not part of content
identity.

- The indexer is the only writer of `metrics.jsonl`.
- Run links are derived from raw trace fingerprints.
- Decision records are append-only mediated policy output.
- Rebuilding derived state must not rewrite `source.py`, `manifest.json`, or
  `current`.

## 4. Freeze and activation flow

At session readiness, observability enumerates all loaded atoms and calls
`freeze_current` for each. It compares the returned content hash with
`list_atoms().current_hash`; a mismatch aborts fingerprint emission.

Reload follows this sequence:

1. Validate proposed source and manifest before mutation.
2. Read current source and compute old/new content hashes.
3. Freeze the old source.
4. Write through the bootstrap `ResourceWriter`.
5. Read the real target file and require exact proposed bytes.
6. Activate the module transactionally.
7. Freeze the activated source.
8. Emit `ExtensionReloadEvent` with the same content hash used by the catalog.

Any failure restores source bytes, reactivates the previous module, resets the
catalog `current` pointer, and restores the captured runtime registry snapshot
if rollback itself fails.

Install uses the same write verification and snapshot freeze. It additionally
captures any pre-existing target bytes so failed import/install/freeze restores
the original file or removes a newly created file.

## 5. Active-set fingerprint

Observability records:

```json
{
  "core": null,
  "scenario": "local",
  "atoms": {
    "tool_read": "tool_read@e5f6a1b2c3d4",
    "tool_bash": "tool_bash@112233aabbcc"
  }
}
```

The indexer accepts only valid content hashes and requires each referenced
snapshot to pass validated reading before appending metrics or run links.
Missing or contradictory identity is a hard error.

Mid-session reload emits a new fingerprint in `agentm.atom.reload`; attribution
after that boundary uses the new active set.

## 6. Atom-facing query surface

`CatalogService` exposes:

- `list_versions(name)`
- `current_version(name)`
- `get_source_at(name, version)`
- `get_manifest_at(name, version)`
- `runs_for(fingerprint)`
- deterministic hash/fingerprint helpers

The contrib `tool_catalog_browse` atom exposes validated read tools only. It
does not expose Git history or arbitrary path lookup.

The contrib `tool_catalog_mutate` atom exposes lifecycle mutation separately.
`rollback_atom(atom, version, rationale)` reads a validated catalog source and
calls transactional `reload_atom`. Success additionally requires the activated
`new_hash` to equal the requested version. Arbitrary resource rollback is not a
catalog operation.

## 7. Non-goals

- Git commits, branches, SHAs, or repository history as runtime identity.
- Versioning arbitrary files through the atom catalog.
- Silently migrating or accepting legacy catalog layouts.
- Using catalog metrics as the sole decision engine.
- Hiding catalog corruption by skipping malformed entries.

Optimization policy belongs to tuner scenarios. Raw observability remains the
source of truth for trajectory analysis; this catalog supplies exact,
recoverable identity and a rebuildable attribution index.
