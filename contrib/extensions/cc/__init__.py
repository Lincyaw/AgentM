"""Claude Code compatibility atoms (opt-in, ``tier=2``).

Lives in a subdirectory so flat-file auto-discovery
(:func:`agentm.extensions.discover.discover_contrib_atoms`) does NOT
walk in here — auto-discovery refuses ``tier>=2`` atoms by design.
Scenarios that want these load them explicitly via their manifest's
``available_inherited_extensions``.
"""
