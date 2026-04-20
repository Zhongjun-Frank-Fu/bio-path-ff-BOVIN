"""M3 · T3.2 / T3.3 — module-level attention pool + ICD-readiness head.

For each of the 11 modules we learn (i) a 1-layer attention scorer that
weighs nodes within the module and (ii) a linear projection that collapses
the weighted mean into a scalar. The 11 scalars are the "module embedding"
the plan §1 pipeline block promises: directly interpretable, feeds the
readiness head with Linear(11, 1).

Batching (added in M4)
----------------------
``forward`` accepts either a single HeteroData (M3 sanity / M5 XAI path)
or a PyG-batched HeteroData where each node store carries a ``.batch``
tensor mapping rows to graph index. The batched path uses segment softmax
+ segment sum so all ``B`` patients pool in one GPU-friendly pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


DEFAULT_MODULE_IDS: tuple[str, ...] = tuple(f"M{i}" for i in range(1, 12))


def _segment_softmax(scores: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Softmax over ``scores`` within each batch group (using PyG utility)."""
    from torch_geometric.utils import softmax as pyg_softmax

    return pyg_softmax(scores, batch, num_nodes=num_graphs)


def _segment_sum(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Sum values within each batch group → (num_graphs,)."""
    from torch_geometric.utils import scatter

    return scatter(values, batch, dim=0, dim_size=num_graphs, reduce="sum")


class ModuleAttentionPool(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        module_ids: tuple[str, ...] = DEFAULT_MODULE_IDS,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.module_ids = tuple(module_ids)
        self.attn = nn.ModuleDict({mid: nn.Linear(hidden_dim, 1) for mid in self.module_ids})
        self.proj = nn.ModuleDict({mid: nn.Linear(hidden_dim, 1) for mid in self.module_ids})

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        module_of_node: dict[str, list[str]],
        batch_dict: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Pool node-level features into per-(graph, module) scalars.

        Returns
        -------
        module_emb : Tensor, shape ``(B, len(module_ids))`` (B = 1 for single-graph path).
        attn : list of length B, each a dict ``{mid → Tensor(n_in_module_in_graph)}``
               of softmax weights, usable by M5 XAI.
        """
        device = next(iter(x_dict.values())).device

        # Flatten per-type tensors → a single (N_total, hidden) matrix, remember
        # module + batch index per flat row.
        Hs: list[torch.Tensor] = []
        batches: list[torch.Tensor] = []
        mods_flat: list[str] = []
        for nt, h in x_dict.items():
            Hs.append(h)
            mods_flat.extend(module_of_node[nt])
            if batch_dict is not None:
                batches.append(batch_dict[nt])
            else:
                batches.append(torch.zeros(h.size(0), dtype=torch.long, device=device))
        H = torch.cat(Hs, dim=0)
        batch_vec = torch.cat(batches, dim=0)
        assert H.size(0) == len(mods_flat) == batch_vec.size(0)
        B = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1

        # Pre-bucket flat rows by module.
        by_module: dict[str, list[int]] = {mid: [] for mid in self.module_ids}
        for i, m in enumerate(mods_flat):
            if m in by_module:
                by_module[m].append(i)

        module_emb = torch.zeros(B, len(self.module_ids), device=device)
        per_graph_attn: list[dict[str, torch.Tensor]] = [dict() for _ in range(B)]

        for col, mid in enumerate(self.module_ids):
            idx = by_module[mid]
            if not idx:
                continue
            idx_t = torch.tensor(idx, dtype=torch.long, device=device)
            Hm = H.index_select(0, idx_t)                       # (N_mid, hidden)
            batch_m = batch_vec.index_select(0, idx_t)          # (N_mid,)
            raw = self.attn[mid](Hm).squeeze(-1)                # (N_mid,)
            alpha = _segment_softmax(raw, batch_m, num_graphs=B)
            proj_vals = self.proj[mid](Hm).squeeze(-1)          # (N_mid,)
            pooled = _segment_sum(alpha * proj_vals, batch_m, num_graphs=B)  # (B,)
            module_emb[:, col] = pooled

            for b in range(B):
                mask_b = (batch_m == b)
                per_graph_attn[b][mid] = alpha[mask_b].detach() if mask_b.any() else torch.empty(
                    0, device=device
                )

        # Single-graph call — collapse to the shapes M3 sanity / M5 XAI expect.
        if batch_dict is None:
            return module_emb.squeeze(0), per_graph_attn[0]  # type: ignore[return-value]
        return module_emb, per_graph_attn


class HeteroGNNClassifier(nn.Module):
    """Backbone → pool → head. Returns dict; shape ``(B, 1)`` logit."""

    def __init__(self, backbone: nn.Module, pool: ModuleAttentionPool) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = nn.Linear(len(pool.module_ids), 1)

    def forward(self, data: "HeteroData") -> dict[str, object]:
        x_dict = self.backbone(data)

        def _flat(mods):
            # Batched HeteroData wraps list-attrs as list-of-lists.
            if mods and isinstance(mods[0], list):
                return [m for sub in mods for m in sub]
            return list(mods)

        module_of_node = {
            nt: _flat(getattr(data[nt], "module", [])) for nt in data.node_types
        }

        # Detect whether HeteroData is batched (PyG's DataLoader sets ``.batch``
        # on each node store when it concatenates a list of HeteroData).
        batch_dict: dict[str, torch.Tensor] | None = None
        if all(hasattr(data[nt], "batch") for nt in data.node_types):
            batch_dict = {nt: data[nt].batch for nt in data.node_types}

        # Pool already collapses single-graph output to flat shapes; batched
        # output stays (B, M). head(Linear) handles both uniformly.
        module_emb, attn = self.pool(x_dict, module_of_node, batch_dict=batch_dict)
        logit = self.head(module_emb)  # (1,) or (B, 1)
        return {"logit": logit, "module_emb": module_emb, "attn": attn}
