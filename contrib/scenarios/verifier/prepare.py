"""Assemble workflow inputs from a case directory.

Pure data preparation — no LLM, no workflow engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .lib.graph import (
    SYNTHETIC,
    build_neighbor_graph,
    get_infra_edges,
    get_infra_nodes,
    get_relationships,
)
from .lib.injection import get_injections, load_fault_doc
from .lib.fpg import (
    REL_MECHANISM,
    load_injection_meta,
)

FAULT_KINDS_DIR = Path(__file__).resolve().parent / "fault_kinds"


@dataclass
class CaseContext:
    """Everything the propagation workflow needs for one case."""

    data_dir: str
    window: dict[str, str]
    injections: list[dict[str, str]]
    graph: dict[str, list[list[str]]]
    infra_nodes: list[str]
    fault_docs: dict[str, str]
    meta: dict[str, Any]
    rel_mechanism: dict[str, str] = field(default_factory=lambda: dict(REL_MECHANISM))

    def to_workflow_args(
        self,
        *,
        out_dir: str,
        budget: int = 15,
        skip_judge: bool = False,
    ) -> dict[str, Any]:
        """Serialize to the dict the workflow script expects."""
        return {
            "data_dir": self.data_dir,
            "window": self.window,
            "injections": self.injections,
            "graph": self.graph,
            "infra_nodes": self.infra_nodes,
            "fault_docs": self.fault_docs,
            "budget": budget,
            "out_dir": out_dir,
            "skip_judge": skip_judge,
            "rel_mechanism": self.rel_mechanism,
        }

def prepare_case(case_dir: Path) -> CaseContext:
    """Build a CaseContext from a case directory."""
    data_dir = case_dir.resolve()

    injections = get_injections(data_dir)
    if not injections:
        raise ValueError(f"{data_dir.name}: no injections found")

    meta = load_injection_meta(data_dir)

    rels = get_relationships(data_dir)
    infra_nodes = get_infra_nodes(data_dir)
    rels.extend(get_infra_edges(data_dir, infra_nodes))
    neighbor_graph = build_neighbor_graph(rels)

    graph_serializable: dict[str, list[list[str]]] = {
        svc: [[n, r] for n, r in neighbors]
        for svc, neighbors in neighbor_graph.items()
        if svc not in SYNTHETIC
    }

    fault_docs: dict[str, str] = {}
    for inj in injections:
        fk = inj["chaos_type"]
        if fk not in fault_docs:
            doc = load_fault_doc(fk, FAULT_KINDS_DIR)
            if doc:
                fault_docs[fk] = doc

    clean_injections = [
        i for i in injections
        if i.get("target") and i["target"] not in SYNTHETIC
    ]

    return CaseContext(
        data_dir=str(data_dir),
        window=meta["window"],
        injections=clean_injections,
        graph=graph_serializable,
        infra_nodes=sorted(infra_nodes),
        fault_docs=fault_docs,
        meta=meta,
    )
