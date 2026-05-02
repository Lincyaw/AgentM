"""Opt-in contrib extensions.

Atoms in this package are NOT auto-discovered by ``discover_builtin``.
They are general-purpose contrib code (e.g. Claude Code compatibility
adapters) that scenarios load explicitly via their manifest's
``available_inherited_extensions``. Keeping them here separates "core
SDK builtins, always available" from "third-party-tool adapters that
the SDK does not assume".
"""
