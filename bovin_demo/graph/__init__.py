"""BOVIN-Pathway graph loading + HeteroData conversion (M1)."""

from __future__ import annotations

from bovin_demo.graph.loader import GraphDict, load_graph

__all__ = ["GraphDict", "load_graph", "to_heterodata"]


def to_heterodata(graph, **kwargs):
    """Lazy wrapper so importing ``bovin_demo.graph`` doesn't pull in torch."""
    from bovin_demo.graph.heterodata import to_heterodata as _impl

    return _impl(graph, **kwargs)
