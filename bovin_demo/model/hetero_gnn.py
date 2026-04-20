"""M3 · T3.1 — HeteroGNN backbone.

Pipeline (one patient graph at a time at M3; batched in M4):

    HeteroData.x_dict
        │
        ├── per-node-type Linear encoder → hidden_dim
        ├── HeteroConv(GATv2Conv) × num_intra_layers  ("intra" message passing)
        ├── HGTConv            × num_inter_layers     ("inter" cross-type attention)
        │
        └── returns x_dict  (dict[node_type → (N_t, hidden)])

Design notes
------------
* The "intra / inter" split in PLAN §1 is a *story*, not a hard PyG
  constraint. GATv2Conv is per-edge-type (handles src→dst pairs along each
  relation); HGTConv is an explicit cross-type attention. Stacking them
  gives a 3-layer GNN where the last layer has the strongest module-crossing
  bandwidth — that's the "inter-module" half.
* ``in_channels=(-1, -1)`` makes GATv2Conv lazy on input dim so the same
  convolution instance works for both same-type (single tensor) and
  bipartite (tuple) edges routed by ``HeteroConv``.
* ``add_self_loops=False`` because heterogeneous self-loops are inserted
  optionally by the HeteroData builder; baking them into the conv would
  double-count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


def _node_feat_dims(data: "HeteroData") -> dict[str, int]:
    return {nt: int(data[nt].x.size(-1)) for nt in data.node_types}


class HeteroGNN(nn.Module):
    """Heterogeneous GNN with per-type encoder + GATv2 + HGT stack."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        feat_dims: dict[str, int],
        *,
        hidden_dim: int = 64,
        num_intra_layers: int = 2,
        num_inter_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.25,
    ) -> None:
        from torch_geometric.nn import GATv2Conv, HeteroConv, HGTConv

        super().__init__()
        node_types, edge_types = metadata
        self.metadata_ = (list(node_types), list(edge_types))
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        self.encoder = nn.ModuleDict(
            {
                nt: nn.Sequential(
                    nn.Linear(feat_dims[nt], hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for nt in node_types
            }
        )

        self.intra_layers = nn.ModuleList(
            [
                HeteroConv(
                    {
                        et: GATv2Conv(
                            (-1, -1),
                            hidden_dim,
                            heads=heads,
                            concat=False,
                            add_self_loops=False,
                            dropout=dropout,
                        )
                        for et in edge_types
                    },
                    aggr="sum",
                )
                for _ in range(num_intra_layers)
            ]
        )

        self.inter_layers = nn.ModuleList(
            [
                HGTConv(
                    hidden_dim,
                    hidden_dim,
                    metadata=self.metadata_,
                    heads=heads,
                )
                for _ in range(num_inter_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: "HeteroData") -> dict[str, torch.Tensor]:
        x_dict = {nt: self.encoder[nt](data[nt].x) for nt in self.encoder}
        edge_index_dict = data.edge_index_dict

        for layer in self.intra_layers:
            # HeteroConv only produces output for node types that appear as
            # ``dst`` in at least one edge type. For node types with no
            # inbound edges we carry the previous hidden forward unchanged,
            # otherwise the next layer gets ``x_dict[nt]=None`` and PyG's
            # GATv2Conv trips its ``x[0].dim() == 2`` assertion.
            out = layer(x_dict, edge_index_dict)
            merged: dict[str, torch.Tensor] = {}
            for nt, h_prev in x_dict.items():
                h_new = out.get(nt)
                merged[nt] = (
                    self.dropout(torch.relu(h_new)) if h_new is not None else h_prev
                )
            x_dict = merged

        for layer in self.inter_layers:
            out = layer(x_dict, edge_index_dict)
            x_dict = {nt: (out[nt] if out.get(nt) is not None else x_dict[nt])
                      for nt in x_dict}

        return x_dict

    @classmethod
    def from_heterodata(
        cls,
        data: "HeteroData",
        *,
        hidden_dim: int = 64,
        num_intra_layers: int = 2,
        num_inter_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.25,
    ) -> "HeteroGNN":
        """Convenience ctor — pulls metadata + feat dims straight off ``data``."""
        return cls(
            metadata=data.metadata(),
            feat_dims=_node_feat_dims(data),
            hidden_dim=hidden_dim,
            num_intra_layers=num_intra_layers,
            num_inter_layers=num_inter_layers,
            heads=heads,
            dropout=dropout,
        )
