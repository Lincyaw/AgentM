"""Host-side driver tools for llmharness.

This subpackage holds modules that legitimately import
``agentm.core.runtime.*`` because they construct standalone AgentM
sessions for offline replay / fork-and-replay workflows. They are NOT
§11 atoms — there is no ``MANIFEST`` / ``install`` pair, they are never
named in a scenario manifest, and they MUST NOT be imported from any
atom under :mod:`llmharness.audit`, :mod:`llmharness.adapters`, etc.

The ``tools/`` placement is the structural signal that says "host
driver, not API surface". The boundary test
``tests/test_replay_engine_boundary.py`` enforces that no atom imports
:mod:`llmharness.tools.engine`.
"""
