# Issue 79 Atom Reloader Hardening

Status: implemented and tested

- Preserved previous `LoadedAtom` registry entries when rollback activation fails after a reload install failure.
- Replaced per-call coroutine bridge loops with native async reload/install/freeze implementations and sync API-boundary facades.
- Added `AtomReloader.shutdown()` for reloader-owned EventBus subscriptions and wired `AgentSession.shutdown()` to call it before clearing the bus.
- Added typed `LoadedAtom.import_kind` and uniform `owners_by_kind` registration ownership tracking.
