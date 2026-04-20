"""Heterogeneous GNN + module-attention readout + baseline MLP (M3)."""

from __future__ import annotations

from bovin_demo.model.baseline_mlp import BaselineMLP
from bovin_demo.model.hetero_gnn import HeteroGNN
from bovin_demo.model.readout import (
    DEFAULT_MODULE_IDS,
    HeteroGNNClassifier,
    ModuleAttentionPool,
)

__all__ = [
    "DEFAULT_MODULE_IDS",
    "BaselineMLP",
    "HeteroGNN",
    "HeteroGNNClassifier",
    "ModuleAttentionPool",
    "build_classifier",
]


def build_classifier(
    data,
    *,
    hidden_dim: int = 64,
    num_intra_layers: int = 2,
    num_inter_layers: int = 1,
    heads: int = 4,
    dropout: float = 0.25,
) -> HeteroGNNClassifier:
    """One-call factory: HeteroData → fully wired HeteroGNNClassifier.

    The classifier's module list is inferred from ``data`` — any module
    absent from the graph (empty M10 in a test sub-sample, say) is still
    given a zero-slot in the readout so downstream indexing is stable.
    """
    backbone = HeteroGNN.from_heterodata(
        data,
        hidden_dim=hidden_dim,
        num_intra_layers=num_intra_layers,
        num_inter_layers=num_inter_layers,
        heads=heads,
        dropout=dropout,
    )
    observed: set[str] = set()
    for nt in data.node_types:
        mods = getattr(data[nt], "module", [])
        # Batched HeteroData wraps list-attrs as list-of-lists (one per graph).
        if mods and isinstance(mods[0], list):
            for sub in mods:
                observed.update(sub)
        else:
            observed.update(mods)
    module_ids = tuple(m for m in DEFAULT_MODULE_IDS if m in observed)
    pool = ModuleAttentionPool(hidden_dim=hidden_dim, module_ids=module_ids)
    return HeteroGNNClassifier(backbone=backbone, pool=pool)
