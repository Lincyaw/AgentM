# Anthropic OAuth Manual Smoke Test

This repo now ships a registry-backed Anthropic OAuth login flow plus persistent auth storage.

## Preconditions

- `uv sync`
- A browser session that can log into Claude Pro/Max
- Working local loopback access to `http://localhost:53692/callback`

## Smoke steps

1. Run `uv run agentm auth providers` and confirm `anthropic` is listed.
2. Run `uv run agentm auth login anthropic`.
3. Complete the browser flow, or paste the final redirect URL / auth code back into the CLI if prompted.
4. Confirm the CLI reports stored credentials at `~/.agentm/auth.json` (or `$AGENTM_AUTH_PATH`).
5. Run `uv run agentm auth status anthropic` and confirm `configured=True`.
6. Run:
   `ANTHROPIC_OAUTH_TOKEN='' uv run agentm run --scenario general_purpose --provider anthropic --model claude-sonnet-4-6 "Say hello in one sentence."`
7. Confirm the session boots using the stored OAuth credential without requiring `ANTHROPIC_API_KEY`.

## Expected result

- Login succeeds.
- Credentials persist across process restarts.
- A subsequent `agentm run ...` resolves Anthropic through the provider registry and authenticates from stored OAuth state.
