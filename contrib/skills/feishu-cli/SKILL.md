---
name: feishu-cli
description: |
  Operate Feishu / Lark resources (messages, docs, calendar, bitable, mail,
  wiki, tasks, OKR, attendance, ...) through the official `lark-cli` binary
  invoked via the `bash` tool. Trigger when the user asks the agent to read
  or write anything in their Feishu / Lark workspace, or mentions "飞书" /
  "Lark" alongside an action verb (read messages, send, create doc, schedule
  meeting, search wiki, draft mail reply, build dashboard, etc.).
type: skill
tags:
  - feishu
  - lark
  - integration
---

# Feishu / Lark CLI

`lark-cli` is the official Feishu / Lark CLI (https://github.com/larksuite/cli).
Once it is installed and authenticated on the host, it gives this agent a
direct hand into the user's Feishu workspace: messages, docs, calendar,
bitable, mail, wiki, tasks, OKR, attendance, video conferences, sheets,
slides — see `lark-cli help` for the full domain list.

You do **not** call a wrapper tool for this. You shell out via `bash`:

```
lark-cli <subcommand> [args...] [--as user|bot] [--dry-run]
```

Output is JSON on stdout; non-zero exit means the request failed (read
stderr for the reason).

## Discovery first — never guess argument shape

Feishu's OpenAPI surface is large and changes. Before composing any non-
trivial call:

1. `lark-cli help` — top-level domain map.
2. `lark-cli <domain> --help` — subcommands inside a domain.
3. `lark-cli schema <api>` — full request / response schema for a specific
   endpoint (e.g. `lark-cli schema im.v1.message.create`). Read this
   instead of guessing field names; the schema is authoritative.
4. `lark-cli api <METHOD> <path> --dry-run` — resolves identity and the
   final URL without sending. Use as a sanity check before any write.

If a call returns `code != 0`, the response body usually carries a
`msg` and an actionable hint. Read it before retrying with different
arguments.

## Identity: `--as user` vs `--as bot`

Two execution identities, and the choice is **load-bearing** — wrong
choice fails silently with permission errors or, worse, succeeds against
the wrong inbox.

| Identity | Token | Use when |
|---|---|---|
| `--as user` | UAT (set up via `lark-cli auth login` once) | Reading or writing on behalf of *this human*: their calendar, their DMs, their drafts, their drive, their mail. The thing the user means when they say "my". |
| `--as bot`  | TAT (app credentials)                       | Acting as the application itself: posting from a bot to a group it is in, reading bitable rows the app has scope for, sending automated notifications. No personal inbox access. |

Defaults: if the host has both UAT and TAT configured, `lark-cli` picks
based on the API; **always pass `--as` explicitly** when the user's
intent is "operate as me" — relying on the default is how agents end up
sending things from a bot account when the user wanted personal-account
behavior.

Quick rule: if the prompt contains "my <inbox|calendar|drive|messages|
mail>", use `--as user`. If it says "the bot" / "the app" / "from the
team account", use `--as bot`. When ambiguous, ask once.

## Safety rules

These are non-negotiable. Treat them as invariants of the skill.

- **Never invoke `lark-cli auth login`, `auth logout`, or `config init`
  yourself.** They are interactive (browser / TTY prompts) and mutate
  user-global credential state. If `auth check` reports missing scope,
  print the suggested command and `auth login --domain <domain>`
  invocation for the user to run, and stop.
- **Dry-run writes first.** Before any `api POST/PUT/PATCH/DELETE`, run
  the same call with `--dry-run` to confirm the URL and identity. Only
  then send the real request.
- **Confirm before fan-out writes.** Sending to many people, deleting
  records, modifying shared documents — surface the list of targets to
  the user and wait for explicit go-ahead. Single self-targeted writes
  (creating a doc in your own drive, adding a calendar event for
  yourself) do not need confirmation.
- **Never echo tokens.** stdout / stderr from `lark-cli` is generally
  safe, but if you ever see an `app_secret`, `tenant_access_token`, or
  `user_access_token` value in output, redact it before quoting back.

## Common shapes

These are seeds, not exhaustive recipes. Look up exact fields with
`lark-cli schema <api>` before each new endpoint you touch.

```bash
# Read recent messages from a chat (user identity)
lark-cli api GET /open-apis/im/v1/messages \
  --params 'container_id_type=chat&container_id=<chat_id>&page_size=20' \
  --as user

# Send a text message as the bot
lark-cli api POST /open-apis/im/v1/messages \
  --params 'receive_id_type=chat_id' \
  --data '{"receive_id":"<chat_id>","msg_type":"text","content":"{\"text\":\"hi\"}"}' \
  --as bot

# List my calendars
lark-cli api GET /open-apis/calendar/v4/calendars --as user

# Create a doc from markdown (delegates to lark-cli's high-level helper
# when present; check `lark-cli help` for the docs subtree)
lark-cli docs create --from-markdown ./draft.md --as user

# Search the wiki
lark-cli api POST /open-apis/wiki/v1/nodes/search \
  --data '{"query":"onboarding","page_size":10}' \
  --as user
```

The exact subcommand layout for docs / bitable / mail evolves; rely on
`lark-cli <domain> --help` rather than memory.

## Cross-time-zone / fan-out workflows

For multi-step orchestration (poll many users' free/busy, classify a
backlog of meeting transcripts, batch-tag bitable rows), prefer:

1. **Plan before execute.** Sketch the exact API calls and the loop
   structure first. Show the user the plan; get one approval; then
   execute the loop. Don't approve each iteration.
2. **Cache list responses.** A `users.list` or `chats.search` round-trip
   that returns IDs you'll reuse 20× should be saved once; don't refetch
   inside the loop.
3. **Stop on first 4xx.** A permission error on iteration 3 of 50
   usually means iterations 4–50 will also fail; stop, surface, ask.

## When `lark-cli` is not the right hand

`lark-cli` is one process per call — fine for a few dozen calls. When
the task crosses ~hundreds of API hits, has long-lived event-subscription
needs, or wants to handle interactive card callbacks, escalate to the
official Python SDK:

- **`lark-oapi` (https://github.com/larksuite/oapi-sdk-python)** —
  programmatic OpenAPI client (`pip install lark-oapi`), with a
  `lark_oapi.channel` submodule that wraps event subscription, message
  normalization, outbound sending, streaming card updates, dedup, and
  SSRF-guarded media upload. Reach for this when building a long-running
  worker, not for one-shot reads / writes. Inside an AgentM session,
  isolate it in a contrib atom or scenario worker — do not import it
  from agent-authored bash one-liners.

The split: `lark-cli` for everything an agent does conversationally;
`lark-oapi` for everything a service does in the background.

## Quick auth sanity check

Before a session that will hit Feishu heavily, confirm credentials are
healthy with a single read call:

```bash
lark-cli auth status
lark-cli api GET /open-apis/calendar/v4/calendars --as user --dry-run
```

If `auth status` reports the user is not logged in but the user wants
"my" operations, stop and ask them to run `lark-cli auth login`. Do not
attempt to bypass with `--as bot` — that operates against the wrong
inbox.
