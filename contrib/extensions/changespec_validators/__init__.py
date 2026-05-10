"""ChangeSpec per-kind validators (B-3).

Each module in this package validates a single ``ChangeSpec.kind`` and
exposes ``validate(change_spec, cwd, target_scenario) -> dict``.

Return shape: ``{"ok": bool, "error": str | None, "resolved_path": str | None}``.
``resolved_path`` is the absolute on-disk path the apply step should write
to (omitted on rejection).

Validators live under ``contrib/`` (not ``src/agentm/``) because they
encode scenario-shape policy, not SDK mechanism. The dispatch is in
``tool_propose_change._execute``.
"""
