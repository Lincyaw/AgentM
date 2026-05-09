# Issue 85 Session Refactor

- Extracted slash-command and cost-budget prompt policy from `AgentSession.prompt` into builtin atoms.
- Added explicit provider selection via `ProviderResolver` and harness `LastRegisteredWins`.
- Moved construction wiring into `session_factory.py` and runtime dependencies into `SessionRuntime`.
- Replaced boundary implementation checks with published Protocol methods and deep-copied `fork_at` payloads.

- Follow-up: pinned slash-command input handling at `BusPriority.PRE` and added CLI trajectory coverage for command/template collisions.
