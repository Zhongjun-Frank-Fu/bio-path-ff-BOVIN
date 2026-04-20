"""M1 · T1.2 — Pydantic v2 schema for the BOVIN-Pathway graph JSON.

The schema has one job: **fail loudly** if the JSON emitted by
``tools/parse_graph_v0.py`` has drifted from what the rest of the stack
expects (right node/edge counts, valid module IDs, every edge endpoint
defined). Plan §5 · M1 DoD is encoded in ``Graph.dod_check``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ModuleId = Literal[
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11"
]
Side = Literal["host", "therapy"]
Direction = Literal["+", "-", "±"]


class Module(BaseModel):
    id: ModuleId
    name: str


class Node(BaseModel):
    id: str = Field(min_length=1)
    symbol: str
    name: str
    type: str = Field(min_length=1)
    side: Side
    observable: bool
    module: ModuleId
    refs: str = ""


class Edge(BaseModel):
    source: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    target: str = Field(min_length=1)
    direction: Direction
    module_from: str | None = None
    module_to: str | None = None
    evidence: str = ""


class Graph(BaseModel):
    version: str
    modules: list[Module]
    nodes: list[Node]
    edges: list[Edge]

    @field_validator("nodes")
    @classmethod
    def _nodes_ids_unique(cls, v: list[Node]) -> list[Node]:
        ids = [n.id for n in v]
        if len(set(ids)) != len(ids):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate node ids: {dup}")
        return v

    @model_validator(mode="after")
    def _edges_reference_known_nodes(self) -> "Graph":
        known = {n.id for n in self.nodes}
        missing: list[tuple[str, str, str]] = [
            (e.source, e.relation, e.target)
            for e in self.edges
            if e.source not in known or e.target not in known
        ]
        if missing:
            raise ValueError(
                "edges reference undefined nodes — the parser should have "
                f"dropped these: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return self

    def dod_check(self) -> None:
        """Plan §5 DoD guard — 82 nodes, ≥99 edges, ≥5 node types, ≥5 edge types."""
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        n_ntypes = len({n.type for n in self.nodes})
        n_etypes = len({e.relation for e in self.edges})
        assert n_nodes == 82, f"expected 82 nodes, got {n_nodes}"
        assert n_edges >= 99, f"expected >= 99 edges, got {n_edges}"
        assert n_ntypes >= 5, f"expected >= 5 node types, got {n_ntypes}"
        assert n_etypes >= 5, f"expected >= 5 edge types, got {n_etypes}"
