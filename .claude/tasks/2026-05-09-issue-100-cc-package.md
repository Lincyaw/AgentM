# Issue 100 — Claude Code compatibility package consolidation

- Requirement: REQ-100-cc-extension-package
- Design concepts touched: extension_as_scenario, single_file_extension_contract, self_modifiable_architecture, skills, agent_team
- Change: converted `contrib/extensions/cc/` from flat `cc_*.py` atoms into an explicitly mounted package (`contrib.extensions.cc`) containing tier-2 atom files, package-internal markdown parsing helpers, and a shared `core.lib.available_agents_block` XML renderer.
- Validation plan: §11 validator for the three public atoms, package/resource discovery tests, ruff, mypy, and full pytest.
