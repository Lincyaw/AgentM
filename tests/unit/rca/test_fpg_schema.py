from __future__ import annotations

from rca import SCENARIO_ROOT
from agentm.extensions.validate import validate_atom_file


def test_fpg_rca_atoms_satisfy_section_11_contract() -> None:
    atom_paths = [
        SCENARIO_ROOT / "src/rca/default/finalize.py",
        SCENARIO_ROOT / "src/rca/default/fpg_contract.py",
        SCENARIO_ROOT / "src/rca/default/rcabench_contract.py",
    ]

    issues = []
    for path in atom_paths:
        issues.extend(
            validate_atom_file(
                path,
                module_path=f"rca.default.{path.stem}",
                known_extension_names={"finalize", "fpg_contract", "rcabench_contract"},
            )
        )

    assert issues == [], "\n".join(
        f"{issue.module_path} [{issue.rule}]: {issue.message}" for issue in issues
    )
