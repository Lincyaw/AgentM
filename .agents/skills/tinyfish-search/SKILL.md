---
name: tinyfish-search
description: Use when a chatbot request needs current web search or clean page fetching through TinyFish. This skill uses the local tinyfish CLI via the existing bash tool, reusing credentials from `tinyfish auth login`.
---

# TinyFish Search

Use the existing `bash` tool to call the local TinyFish CLI. Do not ask for or print API keys.

## Check Auth

If TinyFish has not been used in the current session, first check:

```bash
tinyfish auth status
```

If it reports unauthenticated, tell the user to run `tinyfish auth login` in their shell.

If the `tinyfish` command is missing, install the CLI with:

```bash
npm install -g @tiny-fish/cli
```

## Search

Run:

```bash
tinyfish search query "<query>"
```

Optional hints:

```bash
tinyfish search query "<query>" --location "<location>" --language "<language>"
```

The command returns JSON with `query`, `results`, and `total_results`. Each result includes fields such as `position`, `title`, `url`, `site_name`, and `snippet`.

When the user asks for a result count, trim the JSON result list yourself after the command returns.

## Fetch Content

Run:

```bash
tinyfish fetch content get "<url>" --format markdown
```

For multiple URLs:

```bash
tinyfish fetch content get "<url1>" "<url2>" --format markdown
```

Use `--format json` when the user wants structured extraction, `--links` to include extracted links, and `--image-links` to include image links.

## Response Rules

- Cite result URLs when answering from search or fetched pages.
- Keep the response to the fields the user requested.
- If the CLI fails, report the TinyFish error concisely and do not retry blindly.
