# RFC 0001 — Remote Data-Plane: a DuckDB-over-S3 query lib + service

| | |
|---|---|
| **Status** | Draft (for review) |
| **Author** | Aoyang Fang |
| **Date** | 2026-06-05 |
| **Scope** | `agentm` (consumer: `duckdb_sql` atom) + `aegis` (producer: a new `duckdbquery` lib, a generic query endpoint on the **blob** service, and a thin refactor of the injection datapack-query) |
| **Supersedes** | — (this draft revises its own earlier "reuse the injection endpoint" framing; see *Design evolution*) |

## Summary

Let an agent query remotely-stored parquet with DuckDB SQL **without downloading
it and without requiring the data to be managed by aegislab**, by factoring the
existing DuckDB-over-S3 query engine into:

1. a reusable **`duckdbquery` Go library** — the query primitive: given resolved
   parquet URLs + read-only SQL, run DuckDB and stream Arrow;
2. a **generic query endpoint on the `aegis-blob` service** — `query any
   parquet by S3 bucket+key/prefix`, deployment-agnostic, the natural fit for
   agentm tool-calls and for *your own* S3-uploaded data; and
3. the **existing injection `datapack-query` reduced to a thin wrapper** over the
   same lib (resolve injection id → parquet keys → call lib).

The agent's `list_tables` / `query_sql` tools gain a remote mode that targets the
blob query endpoint (or, for aegis-managed data, the injection endpoint) — **with
no change to the agent's prompt or tool schema**.

This is mostly *re-layering existing code*: the blob service already exists as a
standalone pod with S3 credentials, bucket routing and auth; the injection
query engine already takes resolved parquet URLs, not injections, at its core.

## Design evolution

An earlier cut of this RFC proposed reusing `aegis-api`'s injection-scoped
`datapack-query` directly. That is rejected as the primary design because **not
all data is under aegislab's injection/dataset management** — a user may upload
their own parquet to S3 and must still be able to query it. The injection
endpoint is inherently keyed by an aegis MySQL injection PK, which arbitrary S3
data does not have. So the generic capability moves down into a lib and is
exposed by the blob service (already deployment-agnostic and S3-credentialed),
with the injection endpoint kept only as a convenience wrapper.

## Motivation

### 1. Data is not always aegis-managed
The consumer must be able to query parquet that lives on S3 but was never
registered as an aegis injection/dataset (own uploads, third-party dumps,
ad-hoc exports). Anything keyed by injection PK cannot serve that case. The
addressing primitive must be **S3 bucket + key/prefix**, with injection-id as an
optional convenience on top.

### 2. Data-locality coupling
Today every consumer needs the parquet on local disk: `duckdb_sql` reads a local
`data_dir` (`contrib/scenarios/rca/src/agentm_rca/tools/duckdb_sql.py:265`). For
ops-lite / rcabench-scale data that means copying gigabytes to every machine and
keeping copies in sync with what the producer just generated.

### 3. Per-agent resource blow-up (observed)
On 2026-06-05 a verifier eval at `--case-parallel 200 --parallel 20` drove a
24-core host to **load average 1088** (peak ~3000): each hop is a separate
`agentm` subprocess, each opens its own DuckDB whose worker pool defaults to the
core count (24) and loads parquet into its own RAM — ~1300 subprocesses × 24
threads ≈ 31k workers, while the real bottleneck (the LLM call) sat idle. We
shimmed it with an env-gated thread cap (`AGENTM_DUCKDB_THREADS`). The
structural fix is to stop spawning N local query engines: one server holds data
and bounds concurrency centrally; agents carry zero DuckDB footprint.

### 4. Producer/consumer lifecycle coupling
aegis is the source of truth for datapacks; an agent holding a hand-copied
snapshot is querying a stale fork. Querying the producer's storage in place
makes "the data for X" unambiguous and reproducible.

## Goals

- Query remotely-stored parquet over HTTP/SQL with **no local copy**, whether or
  not it is aegis-managed (address by S3 bucket+key/prefix).
- One **shared query primitive** (lib) behind both the generic blob endpoint and
  the injection wrapper — a single SQL-safety guard, Arrow path, and macro set,
  with no behavioural drift between surfaces.
- **Zero change to the agent's prompt / tool schema**; remote vs local is an
  install-time config of one atom (the env axis).
- Central, single-knob bound on query concurrency (server-side), removing the
  per-subprocess oversubscription class of bug.
- Reuse the blob service's existing S3 creds, bucket routing, auth, and pod.
- Backward compatible: endpoint unset ⇒ today's local mode, unchanged.

## Non-goals

- Replacing local mode (kept for offline / air-gapped / unit-test use).
- A general OLAP/BI service, cross-datapack joins, or write paths. Read-only,
  single-statement, parquet-over-S3 — the envelope the engine already enforces.
- Materialization/caching tiers, query cost accounting, autoscaling — parked
  until load demands them.

## Current state

### Consumer — agentm `duckdb_sql` atom
`contrib/scenarios/rca/src/agentm_rca/tools/duckdb_sql.py`: `connect()` (`:199`)
opens in-memory DuckDB, registers each local `*.parquet` as a view, installs
`p50/p90/p95/p99` macros (`_HELPER_MACROS` `:166`). `list_tables` → `describe()`
(`:244`); `query_sql` (`:298`) enforces read-only + single-statement + `LIMIT`
wrap + token-budget truncation + a dedup cache, returning JSON rows.

### Producer — `aegis-blob` service (the right host, already exists)
- Standalone microservice: `aegis/aegislab/src/cmd/aegis-blob/main.go`,
  `serve --port 8085`, **separate pod** (`helm/charts/blob/templates/deployment.yaml`).
- General object-storage gateway over `minio-go`
  (`src/crud/storage/blob/driver_s3.go`): presign-put/get, inline get, stat,
  list, copy, batch-delete, zip, lifecycle, HMAC token access
  (`src/crud/storage/blob/routes.go`).
- **Already holds S3 credentials, endpoint, and bucket routing**
  (`[blob.buckets.<name>]` in `config.dev.toml`: `endpoint`, `access_key_env`,
  `path_style`, …; buckets `aegis-datapack`, `aegis-dataset`, …). JWT/SSO auth.
- **No DuckDB/Arrow today** — grep of `src/crud/storage/blob/` is clean. That
  capability lives only in the injection domain.

### Producer — injection datapack query (the engine to extract)
`aegis/aegislab/src/core/domain/injection/query_datapack_arrow.go`
(build tag `//go:build duckdb_arrow`):
- `newDuckDBConnector()` (`:199`) — connector whose init `INSTALL/LOAD httpfs`
  (per-connection, needed for S3-presigned `read_parquet`). **Pure, reusable.**
- `validateDatapackSQL()` + `sqlBlacklist` (`:309`) — read-only allowlist
  (SELECT/WITH only; bans `read_parquet`, `attach`, `copy`, `pragma`, DDL/DML…).
  **Pure, reusable.**
- `buildSafeParquetSQL()` / `describeParquetColumns()` / `sanitizeViewName()` —
  path-driven schema + projection. **Pure, reusable.**
- `runDatapackQuery()` (`:356`): `CREATE VIEW … read_parquet('<url>')` per file →
  DuckDB → `ipc.NewWriter` Arrow stream. The **only** injection-coupled step is
  `listDatapackParquets(injectionName)` → `store.ParquetReaderPath(...)` (which
  itself calls `blobclient` to mint presigned URLs). Everything downstream is
  parameterized by resolved URLs.
- Engine: embedded DuckDB via `github.com/duckdb/duckdb-go/v2` (CGo, statically
  linked); results as Apache Arrow IPC via `arrow-go/v18` — written
  **uncompressed** for cross-language/browser codecs (`#423`).
- Consumers: `@x-api-type {"portal":"true"}` (portal-only, not in SDKs) and in
  practice **no active caller** — the engine is proven but nothing depends on it,
  so re-layering carries near-zero blast radius.

**Seam summary:** the query core is already a path-in / Arrow-out function with
an injection-specific *resolver* bolted on the front. The lib boundary is that
exact line.

## Proposal

### Three layers

```
        ┌──── duckdbquery lib (Go pkg, build-tag duckdb_arrow) ────┐
        │  Query(ctx, sources []Source{View, ParquetURL}, sql)     │
        │        → Arrow IPC stream                                 │
        │  Schema(ctx, sources) → []Table{name, rows, columns}     │
        │  owns: validateSQL(allowlist) · newConnector(httpfs)     │
        │        · buildSafeParquetSQL · describe · sanitizeView   │
        │        · p50..p99 macro registration                     │
        └──────────────┬───────────────────────────┬──────────────┘
         integrated by │             wrapped by     │
   ┌──────────────────────────────┐   ┌──────────────────────────────────┐
   │ aegis-blob (:8085) — GENERIC  │   │ aegis-api injection — CONVENIENCE │
   │ POST /blob/buckets/{b}/query  │   │ POST /injections/{id}/datapack-   │
   │   body {keys[]|prefix, sql}   │   │      query   (path unchanged)     │
   │ presign keys via own S3 creds │   │ resolve id→name→files→presign,    │
   │  → build sources → lib        │   │  build sources → lib              │
   │ ONE endpoint; list_tables is  │   │ (thin wrapper; UX unchanged)      │
   │ a discovery query through it  │   │                                   │
   └──────────────────────────────┘   └──────────────────────────────────┘
```

- **lib** is creds-free: it receives already-resolved (typically presigned) URLs
  and reads them via httpfs. SQL safety, schema, macros, and Arrow streaming all
  live here, once.
- **blob endpoint** is the primary, general surface: address parquet by
  `bucket + keys[]` (explicit files → views) or `bucket + prefix` (all `*.parquet`
  under a prefix → one view per file). It reuses blob's existing presign + S3
  creds + bucket auth; it does **not** add DuckDB to blob's storage drivers (a
  new handler imports the lib; drivers stay pure object storage).
- **injection endpoint** keeps its exact route and behaviour but its body becomes
  "resolve → presign → sources → `lib.Query`". `aegislab 那个功能只做一次封装`.

### Library API (Go, sketch)

```go
package duckdbquery // //go:build duckdb_arrow ; no-arrow stub mirrors injection

type Source struct { View string; ParquetURL string } // URL: presigned https or s3://

func Query(ctx, sources []Source, userSQL string) (io.ReadCloser, error) // Arrow IPC
func Schema(ctx, sources []Source) ([]Table, error)
// internal: validateSQL, newConnector(httpfs), buildSafeParquetSQL,
//           registerPercentileMacros (p50/p90/p95/p99), sanitizeViewName
```

Both callers build `[]Source` from their own addressing and delegate.

### Consumer side (agentm)

Add a **remote mode** to the `duckdb_sql` atom (not a fork — keep the tool
contract identical). Selected by config/env:

```toml
- module: agentm_rca.tools.duckdb_sql
  config:
    endpoint: https://aegis.example:8082   # the GATEWAY/edge base, NOT aegis-blob:8085
    bucket:   my-bucket                     # → /api/v2/blob/buckets/{bucket}/query
    dataset:  cases/batch-01KQ.../          # prefix; or keys[]; or AGENTM_DUCKDB_DATASET
    # AGENTM_DUCKDB_TOKEN: a machine bearer (OIDC client_credentials or
    # service-account token); row_limit/token_limit unchanged
```

`endpoint` is the **gateway / edge-proxy** entrypoint, never a direct upstream.
The atom appends `/api/v2/blob/buckets/{bucket}/query` (or
`/api/v2/injections/{id}/datapack-query`) and attaches `Authorization: Bearer
<token>`. See *Gateway routing & auth* below for why direct-to-blob is forbidden.

- `endpoint` set ⇒ `connect()` is replaced by an HTTP client; `data_dir` optional.
- **One server endpoint — schema is just a query.** There is no separate
  `/schema` endpoint; `list_tables` is a discovery query
  (`SELECT table_name, column_name, data_type FROM information_schema.columns`)
  sent through the same `/query`, reshaped into the existing
  `{tables:[{table,columns}]}` payload (row counts dropped; the agent can
  `count(*)` itself).
- `query_sql` → `POST …/query` with `{prefix|keys, sql}`. The **client keeps the
  UX-shaping guards** it owns (single-statement, `LIMIT` wrap, token-budget
  truncation, dedup cache); the **lib keeps the security guard** (read-only
  allowlist) — defense in depth across the now-untrusted wire.
- Result: read Arrow IPC with `pyarrow.ipc.open_stream` → rows → existing
  `_serialize`/`_compact_ids`/`_truncate` → **byte-identical agent output**. (A
  light `pyarrow` extra; or aegis offers `Accept: application/json` to skip Arrow.)
- For aegis-managed data the same atom can instead target
  `…/injections/{id}/datapack-query` — same Arrow response, no code difference.

### Addressing — two tiers

| Tier | Handle | Who serves | Use |
|---|---|---|---|
| Generic | `bucket + key[] / prefix` | **blob `/query`** | own S3 uploads, any parquet, agentm default |
| Convenience | injection `id` | aegis-api injection wrapper | aegis-managed datapacks (id is in each case's `injection.json`, see below) |

aegis-managed cases already carry their handle: every local case ships an
`injection.json` with the aegis injection PK (`"id": 1908`) and `name`
(`batch-<ULID>` = the dir name). Same-instance queries use `id` directly; the
portable cross-instance handle is `name` (needs `POST /injections/search`).

### Your-own-S3 flow (the case that motivated this)

1. `POST /blob/buckets` (or a configured `[blob.buckets.mine]`) → a bucket.
2. `presign-put` → upload your `*.parquet`.
3. Point the agent at `…/blob/buckets/mine` + a prefix; `list_tables` /
   `query_sql` just work. No injection, no MySQL, no aegis dataset registration.

### Where each responsibility lives

| Concern | Local mode | Remote mode |
|---|---|---|
| Read-only enforcement | atom | **lib** (server) + atom (early reject) |
| Single statement / LIMIT wrap | atom | atom |
| `p50..p99` helpers | atom | **lib registers** |
| Token-budget truncation, dedup, id-compaction | atom | atom |
| DuckDB worker-thread bound | per-subprocess env | **one server-side setting** |
| S3 creds / presign / bucket auth | n/a | **blob service** |
| Data residence | every consumer's disk | S3, queried in place |

## API surface (remote)

All paths are reached **through the gateway** (`{gw}` = edge-proxy/gateway base,
e.g. `https://<edge>:8082`), never the upstream pod directly.

```
# ONE generic endpoint — list_tables is a discovery query through it, not a 2nd route
POST {gw}/api/v2/blob/buckets/{bucket}/query     Accept: arrow | json
       body: { "prefix": "cases/…/", "sql": "SELECT … FROM abnormal_traces …" }
        or:  { "keys": ["…/abnormal_traces.parquet", …], "sql": "…" }
       → Arrow IPC stream  (or { row_count, rows:[…] } when Accept: json)
       list_tables sends sql = SELECT … FROM information_schema.columns
# convenience (unchanged path, lib-backed; injection keeps its own schema route for the portal):
POST {gw}/api/v2/injections/{id}/datapack-query   body { "sql": "…" }
GET  {gw}/api/v2/injections/{id}/datapack-schema
Auth:  Authorization: Bearer <token>             (gateway verifies via SSO JWKS)
```

The `/api/v2/blob/` prefix already routes to `aegis-blob:8085` with `auth=jwt`,
and `/` (catch-all) routes to `aegis-api:8082` with `auth=jwt`, so **no new
gateway route is needed** — the query/schema endpoints inherit existing routing.

## Gateway routing & auth (all traffic transits the gateway)

Every request goes through the L7 gateway (`aegis-gateway:8086`, fronted
externally by the Caddy edge-proxy + LoadBalancer on the public IP:8082). Agents
**must not** hit `aegis-blob:8085` / `aegis-api:8082` directly.

**Why direct-to-upstream is forbidden.** The gateway is the authn chokepoint: it
verifies the bearer (JWT/service token) against the SSO JWKS, then injects
*trusted headers* (`X-Aegis-User-Id`, … + an `X-Aegis-Signature` HMAC over the
canonical claim set, keyed by `gateway.trusted_header_key`). Upstreams trust that
signature and **do not re-verify the JWT**. So a client that bypasses the gateway
either (a) is rejected for missing/forged trusted headers, or (b) — if the pod is
network-reachable — slips past authz entirely. Mitigation is twofold:
**NetworkPolicy** so only the gateway can reach upstream pods, and the existing
**trusted-header HMAC** so forged `X-Aegis-*` headers are detected.

**Machine authentication (the real gap).** There is *no* self-service token
issuance; a human must provision a credential once, out-of-band. Three
non-interactive options, best-first for an agent:
- **OIDC `client_credentials`** — `POST {gw}/token` with `client_id`+`client_secret`
  → 24h service token; the SSO client auto-refreshes (~60s skew). Give the agent
  the client_id/secret via env; it manages tokens itself. **Recommended.**
- **Service-account token** — `POST {gw}/v1/service-accounts/{name}/issue` →
  long-lived (365d default, no refresh — re-issue). Good for baking a static
  credential into an eval environment.
- **API-key → token exchange** — HMAC-signed headers → 24h user JWT. Heavier;
  needs a human-created API key.

**Auth-mode & audience gotchas (will silently 401 if wrong):**
- The `/api/v2/blob/` route is `auth=jwt` today. To accept *machine* service
  tokens, set the query route (or blob route) to **`jwt-or-service`** — the
  gateway supports `none | jwt | service-token | jwt-or-service`.
- Service tokens carry `aud=["sso"]`; user tokens default `aud=["portal"]`. If
  the query route declares `audiences=["portal"]`, a machine service token is
  **rejected**. Either leave the query route audience-unrestricted or mint tokens
  with the matching audience.
- **Authz, not just authn:** a SELECT can read every column — same exposure as a
  download — so the agent's identity needs **read ACL on the target bucket**
  (the own-S3 bucket, or the datapack/injection scope). Provision that with the
  credential.

**Streaming & limits through the gateway:**
- The gateway uses `net/http.ReverseProxy` — it **streams, does not buffer, and
  does not rewrite Content-Type** — so `application/vnd.apache.arrow.stream`
  passes through cleanly. ✅
- **`ResponseHeaderTimeout` defaults to 30s** (per-route `timeout_seconds`): a
  heavy scan whose first byte lands after 30s is cut off. DuckDB emits the Arrow
  schema early so first-batch latency is usually low, but the query route should
  carry a **raised `timeout_seconds`** for big datapacks.
- **Rate limit is global 200 RPS / burst 400.** This collides head-on with the
  agent fan-out from the CPU-incident story — 100–200 concurrent agents firing
  queries will trip the gateway's limiter (429). The eval path needs a **higher
  limit or a dedicated route bucket**, or client-side pacing. (Same concurrency
  budget, now visible at the gateway instead of the CPU.)

## Performance: closing the CPU incident

In remote mode an agent subprocess holds **no** DuckDB connection, spawns **no**
24-thread pool, loads **no** parquet into RAM — one HTTP POST, then it waits (the
I/O-bound work it was always meant to be). The blob/injection server runs DuckDB
with a single operator-set `threads`/`memory_limit`, so concurrency is bounded by
design. The `AGENTM_DUCKDB_THREADS` shim becomes unnecessary in remote mode
(kept for local).

## Risks & trade-offs (honest accounting)

This design reuses existing infra, which is its strength and its weakness. The
risks below are real; two of them (R5, R6) are serious enough that their
mitigations are promoted into Phase 1 below rather than left as options.

- **R1 — Bearer is bucket-wide; blast radius is the whole bucket.** A SELECT
  reads any column the token's bucket ACL allows, so a leaked agent token
  exfiltrates the entire bucket, not one datapack. Least-privilege is violated.
  *Mitigation: task-scoped capability (see north-star + Phase 1).*
- **R2 — We overload a human SSO system for machine-to-machine.** 24h tokens,
  refresh, audiences, api-keys are built for interactive users. Almost every
  gotcha (audience-mismatch 401s, provisioning friction, secret distribution to
  a 200-way fan-out) is an artefact of bending human OIDC onto agents, not an
  intrinsic need.
- **R3 — Auth correctness depends on NetworkPolicy actually being enforced.**
  The gateway verifies the JWT then upstreams trust an HMAC header and *don't
  re-verify*. If upstream pods are directly reachable in-cluster (common in
  kind/dev), any pod bypasses the gateway and authz collapses. The trusted-header
  scheme is only as strong as the network isolation around it.
- **R4 — Token / presigned-URL leakage into traces.** The atom's tool-calls are
  logged to `.agentm/observability`; endpoint, bearer, and presigned URLs must be
  redacted or they land in trace files.
- **R5 — Centralisation trades per-process oversubscription for shared-fate
  DoS.** The thing that fixes the CPU incident — one shared engine — means one
  pathological query (`range(1e12)`, cross join, deep recursive CTE) can OOM/peg
  the engine for *everyone*. In the per-subprocess model a bad query only hurt
  itself. *Mitigation: per-session resource limits + isolation (Phase 1).*
- **R6 — SQL safety is a keyword denylist — wrong polarity, maintenance
  treadmill.** `validateDatapackSQL` regex-bans `read_parquet`/`attach`/… on
  lowercased text. Denylists miss new DuckDB functions (`read_text`, `read_blob`,
  lambdas, `glob`), false-positive on column names (`copy`), and are evadable via
  comments/aliases. "Default-allow, enumerate-forbidden" is the wrong default.
  *Mitigation: self-confining engine (Phase 1) makes the denylist non-load-bearing.*
- **R7 — Statelessness loses connection warmth → possible per-query
  regression.** Local mode keeps one DuckDB connection for the whole agent
  session (views registered once, page cache warm, dedup cache). An agent runs
  10–20 queries per case; a request-per-query remote model re-presigns and
  re-`CREATE VIEW read_parquet(https://…)` every time, over the network instead
  of mmap. Removing process-startup cost can be undone by added per-query
  latency unless the server keeps the dataset connection warm.
- **R8 — Engine-version drift breaks "same engine" reproducibility.** The local
  atom uses the Python `duckdb` package; aegis uses Go `duckdb-go v2.10502`.
  Different builds/versions can differ in function availability and type
  coercion, so local and remote results may diverge — bad for a *verifier* whose
  verdicts must be reproducible. Pin/verify version parity, or make remote the
  single canonical engine.

## A more elegant north-star (where this should grow)

Reuse-optimal ≠ optimal. Abstracting away from the current aegis implementation,
the elegant shape is **a per-task, capability-scoped, self-confining data
session** — and the MVP above is its degenerate special case (bucket-wide bearer
+ stateless query), so it can grow into this without a rewrite.

1. **Capability, not identity — the agent holds no aegis credential.** The
   orchestrator (which already has creds) mints, per dispatch, a capability good
   for *exactly the one datapack* the agent will investigate, with a short TTL,
   and hands the agent an opaque `{session_url, scoped_token}`. Blast radius
   shrinks from "bucket / 24h" to "datapack / 1h"; provisioning friction
   disappears (minted per-task, not per-agent). Presigned URLs are already this
   pattern at file granularity — this just lifts it to a query session.
2. **Stateful warm session, not request-per-query.** The orchestrator opens a
   dataset session (authenticating as itself, scoped to X); the agent runs many
   queries against a connection whose views are already registered and cache is
   warm; the session closes with the task. Fixes R7 and the auth-scope problem at
   once — and maps cleanly onto AgentM's *env* concept (the data env is
   provisioned per-task, like a sandbox).
3. **Self-confining engine — security by construction, not denylist (cheap,
   do it now).** The principle: the engine holds **no ambient credentials** and
   is **locked to a fixed source allowlist** after view setup, so a missed
   denylist entry has nothing to reach. The concrete mechanism is
   backend-dependent — and note the trap that *naively* `SET
   enable_external_access=false` also breaks the views, since
   `CREATE VIEW … read_parquet(url)` is lazy and re-invokes external I/O on every
   `SELECT`:
   - **S3-backed (the common case):** set **no S3 secret** on the connection, so
     only the caller's presigned URLs resolve — an arbitrary `s3://` or
     other-bucket URL fails on signature. The current connector already does this
     (it only `LOAD httpfs`, configures no secret), so S3 confinement is largely
     free.
   - **Local / JuiceFS-backed:** `SET allowed_directories=['<datapack_dir>']` +
     `SET lock_configuration=true` fences filesystem reads to the datapack dir;
     keep the keyword denylist only as a backstop against local file-reader
     functions (`read_text`/`read_blob`/local `read_csv`).
   - **Fully airtight alternative:** materialise views into `TEMP` tables before
     locking the connection — then external access can be disabled outright, at a
     RAM/latency cost (acceptable for small per-case parquet, not for the large
     histogram files). Make this a per-deploy toggle.
   This demotes the denylist from load-bearing to backstop, neutralising the
   bulk of R6.
4. **Per-session resource fence — isolate DoS (cheap, do it now).** Per-session
   connection + `SET memory_limit`, a statement timeout (Go context), and
   `SET threads=N`. A pathological query bounds to its own session instead of
   taking down the shared engine (R5).
5. **Consumer protocol stays generic and producer-agnostic.** The atom speaks one
   "SQL session" protocol — `{session_url, bearer}` in, `query/list/close` out —
   and is *auth-scheme-agnostic*. Every aegis specific (gateway path, token
   minting, presign, engine confinement) hides behind the session endpoint, so a
   different producer (your own S3, a third party) implements the same protocol
   and the atom changes nothing. This is the real "data-plane as a pluggable env".

Together these are strictly better on least-privilege, no standing machine
credential, connection warmth, DoS isolation, and consumer/producer boundary.

## Backward compatibility & rollout

- **Phase 0 (done):** `AGENTM_DUCKDB_THREADS` thread-cap for local mode.
- **Phase 1:** extract the `duckdbquery` lib from `query_datapack_arrow.go`
  (pure pieces move as-is); add `POST /blob/.../query` + `/schema` using blob's
  presign; build `aegis-blob` with `-tags duckdb_arrow` (+ no-arrow stub).
  **Two north-star mitigations are must-haves here, not later** (both are a few
  lines and kill the two serious risks):
  - **Self-confining engine (R6):** no ambient S3 secret (so only caller-minted
    presigned URLs resolve) + `allowed_directories` + `lock_configuration` for
    local-backed data, so the keyword denylist drops to a backstop. (Not a naive
    `enable_external_access=false` — that would also disable the lazy
    `read_parquet` views; see north-star §3 for the correct mechanism.)
  - **Per-session resource fence (R5):** per-query `SET memory_limit`, a
    statement timeout via the request context, and `SET threads=N`, so one bad
    query cannot take down the shared engine.
  - Redact endpoint/bearer/presigned URLs from the atom's logged tool args (R4).
- **Phase 2:** refactor injection `datapack-query` to call the lib; register
  `p50..p99` macros in the lib; flip `sdk:true` on the query/schema handlers so
  non-browser clients are first-class; optional `Accept: application/json`.
- **Phase 3:** agentm `duckdb_sql` remote mode against the blob endpoint (Arrow→
  JSON in the atom); validate on a handful of verifier cases end-to-end; drop the
  local copy from the eval harness.
- **Phase 4 (conditional):** scale/replicate the query path, JSON fast-path,
  per-bucket query quotas — when fan-out load justifies it.

## Alternatives considered

- **Reuse the injection endpoint directly (earlier draft).** Can't serve
  non-aegis S3 data (keyed by injection PK); see *Design evolution*.
- **Put the generic query on `aegis-api` / a brand-new pod.** `aegis-blob`
  already has S3 creds, bucket routing, auth, and its own pod — it *is* the
  general data-plane service; adding a handler beats standing up infrastructure.
- **Keep local + thread cap only (Phase 0).** Treats the symptom; still copies
  data and still spawns a query engine per subprocess.
- **Arrow Flight SQL / Trino / Presto.** Heavier, new infra/auth; the monorepo
  already speaks DuckDB+Arrow — reusing it is far cheaper.
- **Expose via MCP instead of the native atom.** Possible later (an `mcp_bridge`
  fronting the same endpoint); the native atom already carries the prompt-tuned
  tool descriptions and result shaping, so remote mode reuses all of it.

## Open questions

1. **Presigned URLs vs DuckDB S3 secret.** Today injection passes presigned
   HTTPS URLs to httpfs (creds-free engine, TTL-bounded). Keep that in the lib,
   or let the lib hold a DuckDB S3 secret and read `s3://` directly? Recommend
   presigned URLs — keeps the lib creds-free.
2. **Authz model: bucket ACL now, task-scoped capability later.** MVP reuses
   blob's bucket read-ACL (a SELECT = a download in exposure). The *north-star*
   replaces the standing bucket-wide bearer with an orchestrator-minted,
   per-datapack, short-TTL capability (R1). Open: do we ship a minimal signed
   `{prefix, exp}` capability already in Phase 1, or defer the full session?
3. **Prefix → view naming.** With `prefix`, one view per `*.parquet`
   (`sanitizeViewName(filestem)`), matching local mode's table names so agent SQL
   is portable. Confirm collision handling.
4. **Glob / many files.** Should `keys` allow a DuckDB glob (`read_parquet([...])`)
   for partitioned datapacks, or stay one-view-per-file? Start one-per-file.
5. **Build-tag reach.** `aegis-blob`'s image/CI must build with `-tags
   duckdb_arrow` to include the endpoint (no-arrow stub otherwise).
6. **Machine credential for the eval:** OIDC `client_credentials` (24h,
   auto-refresh — best for live agents) vs a long-lived service-account token
   (365d, no refresh — simplest to bake into an eval env). Operational choice;
   both need one-time human provisioning with read ACL on the target bucket.
7. **Route auth-mode change:** flip the query route to `jwt-or-service` and keep
   it audience-unrestricted so machine service tokens (`aud=["sso"]`) are
   accepted — a small `[[gateway.routes]]` config change that needs sign-off.
8. **Gateway limits under fan-out:** raise the query route's `timeout_seconds`
   (big-scan first-byte) and give it rate-limit headroom (or a dedicated bucket)
   so eval fan-out doesn't hit the global 200 RPS / 30s defaults.
9. **Engine-version parity (R8):** the local Python `duckdb` and aegis Go
   `duckdb-go v2.10502` are different builds. Pin them to matching DuckDB
   versions, or declare remote the single canonical engine for verifier runs so
   verdicts stay reproducible?

## Appendix — references

- Consumer atom: `contrib/scenarios/rca/src/agentm_rca/tools/duckdb_sql.py`
  (`connect` :199, `_HELPER_MACROS` :166, `query_sql` :298, `_resolve_data_dir` :265).
- Query engine to extract:
  `aegis/aegislab/src/core/domain/injection/query_datapack_arrow.go`
  (`newDuckDBConnector` :199, `validateDatapackSQL`/`sqlBlacklist` :309,
  `buildSafeParquetSQL` :109, `runDatapackQuery` :356, `getDatapackSchema` :227).
- Injection↔blob path resolution:
  `…/domain/injection/datapack_store_s3.go` (uses `blobclient`),
  `…/src/clients/blob/module.go` (local/remote modes).
- Blob service: `…/src/cmd/aegis-blob/main.go`,
  `…/src/crud/storage/blob/{service,handler,routes,driver_s3,registry}.go`,
  `…/helm/charts/blob/templates/deployment.yaml`, `config.dev.toml [blob.buckets.*]`.
- Case→datapack identity: each case's `injection.json` (`id`, `name`).
- Gateway routing/auth: `aegis/aegislab/src/cmd/aegis-gateway/main.go`,
  `…/src/clients/gateway/{router,middleware_auth,proxy,types}.go`,
  `helm/templates/configmap.yaml` `[[gateway.routes]]` (blob/injection routes),
  edge-proxy `helm/templates/edge-proxy-config.yaml`.
- Machine auth: `…/src/crud/iam/sso/{oidc_grants,service_account}.go`,
  `…/src/crud/iam/auth/handler.go` (api-key→token), JWKS
  `…/src/platform/jwtkeys/module.go`, `…/src/platform/crypto/jwt.go`.
- AgentM env axis: `.claude/designs/pluggable-architecture.md`
  (Tool+Operations as the env port).
