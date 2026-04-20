"""M1 · T1.4 — GraphDict → ``torch_geometric.data.HeteroData``.

Conventions
-----------
* **Node types** come from the node's ``type`` field (enzyme/TF/receptor/...).
  Every node type becomes a PyG node store with its own x/index mapping.
* **Edge types** group by ``relation`` (activates/inhibits/binds/...). The
  full PyG edge-type tuple is ``(src_ntype, relation, dst_ntype)``; ``binds``
  between ``receptor`` and ``receptor`` and between ``DAMP`` and ``receptor``
  therefore become two distinct edge types, which is exactly what HGTConv
  wants (one attention weight per semantic relation type).
* ``x`` is initialized to small random features of shape ``(N_t, feat_dim)``
  per node type; real TCGA features are swapped in at M2.
* ``add_self_loops`` mirrors the config flag from ``configs/default.yaml``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from bovin_demo.graph.loader import GraphDict

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


def to_heterodata(
    graph: GraphDict,
    *,
    feat_dim: int = 8,
    add_self_loops: bool = False,
    generator: "torch.Generator | None" = None,  # noqa: F821  (TYPE_CHECKING)
) -> "HeteroData":
    """Convert a validated ``GraphDict`` into a ``HeteroData`` object.

    Heavy imports (``torch``, ``torch_geometric``) are deferred to call time so
    ``bovin_demo.graph.loader`` stays importable on boxes without PyG.
    """
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()

    # --- nodes: one store per ``type`` -----------------------------------
    nodes_by_type: dict[str, list[dict]] = defaultdict(list)
    node_local_index: dict[str, tuple[str, int]] = {}
    for n in graph["nodes"]:
        ntype = n["type"]
        idx = len(nodes_by_type[ntype])
        nodes_by_type[ntype].append(n)
        node_local_index[n["id"]] = (ntype, idx)

    for ntype, nlist in nodes_by_type.items():
        data[ntype].x = torch.randn(len(nlist), feat_dim, generator=generator)
        data[ntype].node_ids = [n["id"] for n in nlist]
        data[ntype].module = [n["module"] for n in nlist]
        data[ntype].observable = torch.tensor(
            [n["observable"] for n in nlist], dtype=torch.bool
        )

    # --- edges: one store per (src_type, relation, dst_type) --------------
    buckets: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for e in graph["edges"]:
        s_type, s_idx = node_local_index[e["source"]]
        t_type, t_idx = node_local_index[e["target"]]
        buckets[(s_type, e["relation"], t_type)].append((s_idx, t_idx))

    for (s_type, relation, t_type), pairs in buckets.items():
        src_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        dst_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        edge_index = torch.stack([src_idx, dst_idx], dim=0)
        data[s_type, relation, t_type].edge_index = edge_index

    if add_self_loops:
        for ntype, nlist in nodes_by_type.items():
            n = len(nlist)
            idx = torch.arange(n, dtype=torch.long)
            data[ntype, "self_loop", ntype].edge_index = torch.stack([idx, idx], dim=0)

    return data
