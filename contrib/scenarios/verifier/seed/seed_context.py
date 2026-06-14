"""Build the user prompt for seed verification agents."""
from __future__ import annotations

_FAULT_CONTEXT_TEMPLATE = "Injected fault: {fault_kind} on {target}"


def build_seed_prompt(
    *,
    target: str,
    fault_kind: str,
    params: str,
    fault_doc: str,
) -> str:
    """Build the user prompt for verifying an injection target."""
    sections: list[str] = []

    # -- Fault scenario --
    line = _FAULT_CONTEXT_TEMPLATE.format(fault_kind=fault_kind, target=target)
    if params:
        line += f" ({params})"
    fault_lines = [line]
    if fault_doc:
        fault_lines.append(fault_doc)
    sections.append("## Fault scenario\n" + "\n\n".join(fault_lines))

    # -- Target --
    sections.append(
        f"## Target\n"
        f"Service: **{target}**\n"
        f"This is the injection target. Verify that the fault actually "
        f"took effect by comparing its behavior in the normal vs "
        f"abnormal windows."
    )

    return "\n\n".join(sections)
