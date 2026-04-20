"""M3 · T3.4 — flat gene-vector → 2-layer MLP baseline.

Purpose (Plan §7 · DoD #3): the HeteroGNN must beat this by ≥ 0.03 AUC on
test. The baseline is deliberately simple — it gets the exact same
observable-gene vector the GNN sees, but no graph structure — so any AUC
gap is attributable to the biology prior, not to capacity.
"""

from __future__ import annotations

import torch
from torch import nn


class BaselineMLP(nn.Module):
    """2-layer MLP with dropout, binary logit output."""

    def __init__(
        self,
        in_features: int,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape ``(B, in_features)`` → logits ``(B, 1)``."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)
