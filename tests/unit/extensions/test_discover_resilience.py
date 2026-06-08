"""Fail-stop: one broken atom must not deny the whole catalog.

The unreplaceable-substrate axiom says the *only* thing that cannot be a
replaceable extension is the act of loading replacements. If
``discover_builtin`` re-raises on the first import error, a single
typo / missing optional dep / bad MANIFEST in any one builtin atom kills
every ``agentm`` CLI invocation before any scenario can load — the
substrate has lost its only job.

This test drops a syntactically broken Python file into
``agentm/extensions/builtin/`` for the duration of the test, calls
``discover_builtin``, and asserts:

* Discovery returns a non-empty dict that still contains other real
  atoms (e.g. ``operations``).
* The broken atom shows up in ``last_discovery_failures()`` with its
  module path and the captured exception.
* A second call to ``discover_builtin`` is cache-served (does not
  re-raise, does not re-walk).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import agentm.extensions.builtin as _builtin_pkg
from agentm.extensions.discover import (
    discover_builtin,
    last_discovery_failures,
    reset_cache,
)


def test_broken_atom_does_not_break_catalog(tmp_path: Any) -> None:
    pkg_dir = Path(_builtin_pkg.__file__).parent  # type: ignore[arg-type]
    broken = pkg_dir / "zz_broken_for_resilience_test.py"
    # Top-level ImportError — the most common real-world failure shape
    # (missing optional dep / typo in a sibling module reference).
    broken.write_text(
        "from __future__ import annotations\n"
        "raise ImportError('synthetic failure for resilience test')\n"
    )
    try:
        reset_cache()
        entries = discover_builtin()

        # Other real atoms still loaded.
        assert "operations" in entries, (
            "broken atom denied unrelated atoms — substrate axiom violated"
        )
        # Broken atom did not appear in the entries dict.
        assert "zz_broken_for_resilience_test" not in entries

        # Failure surfaced via the documented inspection hook.
        failures = last_discovery_failures()
        assert any(
            mod.endswith("zz_broken_for_resilience_test")
            and isinstance(exc, ImportError)
            for mod, exc in failures
        ), f"expected the synthetic failure in last_discovery_failures(); got {failures!r}"

        # Cache hit on second call — no re-walk, same dict.
        entries2 = discover_builtin()
        assert entries2 is entries
    finally:
        broken.unlink(missing_ok=True)
        reset_cache()
