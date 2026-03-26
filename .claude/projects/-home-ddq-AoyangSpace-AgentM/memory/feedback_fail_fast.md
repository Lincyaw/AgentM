---
name: fail-fast-principle
description: User prefers exposing errors early over hiding them with fallbacks; avoid silent defaults that mask bugs
type: feedback
---

Expose errors as early as possible; do not hide them with fallbacks.

**Why:** The user's engineering principle is that silent fallbacks (default values, swallowed exceptions) push bugs downstream where they're harder to diagnose. It's better to fail loudly at the point of error than to produce subtly wrong results.

**How to apply:**
- Prefer raising exceptions over returning default values when a missing field or config indicates a bug
- Use `except Exception: pass` only in true teardown paths (file close, connection cleanup)
- When a `.get(key, default)` default would mask a configuration error, use `[]` lookup and let it raise
- Log warnings at minimum when fallback paths are taken during execution
- In tool functions called by LLM: returning error strings is OK (Layer 1 self-heal). In SDK internals: raise.
