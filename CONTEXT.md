# AgentM Glossary

Terms with precise meaning inside this project. Implementation details belong elsewhere — this file is a glossary, not a spec.

## Session

The single user-facing entry point: `from agentm import Session; Session.create(...)`. Wraps everything — substrate construction, atom installation, override handling. Users do not need to know about "floor atoms", "scenarios", or the extension API to use AgentM.

Power-users override defaults two ways:

- pass an axis object directly: `Session.create(session_manager=MyOne(), ...)`
- pass a replacement floor tuple: `Session.create(floor=("my.atoms.foo", ...))`

The CLI consumes `Session.create` the same way a notebook does — there is no separate entry path.

## Floor

The Python tuple `FLOOR_ATOMS` in `agentm/session.py` listing the atoms `Session.create` installs before any user scenario. Editing the tuple adds or replaces a floor atom. There is no manifest file, no "required" vs "optional" tier; every floor atom is on equal footing.

Floor atoms are responsible for being polite — each one checks `if api.has_X(): return` before registering, so a user-supplied axis object (pre-registered by `Session.create` from kwargs) silently wins.

If a floor atom is removed and downstream code needs the slot it filled, the first call to that slot raises a clear error. There is no freeze-time required-axis check.

## Substrate-only

Two kernel singletons that are not pluggable and are not floor atoms: **CatalogService** and **ProviderResolver**. They have one implementation, instantiated by the substrate itself. The boundary axiom ("substrate never instantiates a default") does not bind them because there is no axis to plug.
