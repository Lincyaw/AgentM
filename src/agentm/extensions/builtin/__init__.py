"""Built-in extensions catalog.

Each ``.py`` file under this package is a single self-contained extension
following the §11 contract in ``.claude/designs/extension-as-scenario.md``.
Phase 2 atom modules land here. This package itself contains no logic;
``agentm.extensions.discover`` enumerates the catalog at runtime.
"""
