"""M5 · T5.1 — Captum ``IntegratedGradients`` on node-level inputs.

Why attribute to node features (not ``module_emb`` directly)?
------------------------------------------------------------
The readout's head is ``Linear(11, 1)``; IG through that alone is just
``w · (x - baseline)`` — a tautology, no explanation. We instead attribute
to the original per-node input ``x_dict`` (shape ``(N_t, feat_dim)`` per
type) and let IG flow through the encoder + GATv2 + HGT + pool + head
stack. The resulting (node, feat) attributions roll up two ways:

    node-level :  sum over feat_dim        → (N_total,)
    module-level: sum over nodes in module → (11,)

Both are stored so the M5 sanity can ask either "CRT in top-5 nodes?" or
"M4 DAMP in top-3 modules?" (PLAN §7 DoD items #4 and #5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


@dataclass
class NodeAttribution:
    """Per-patient attribution packaged for downstream aggregation / viz."""

    node_ids: list[str]          # length N_total, in flat order
    node_types: list[str]        # length N_total
    modules: list[str]           # length N_total
    attributions: np.ndarray     # shape (N_patients, N_total) — sum over feat_dim
    logits: np.ndarray           # shape (N_patients,)
    probs: np.ndarray            # shape (N_patients,)
    labels: np.ndarray           # shape (N_patients,)
    sample_ids: list[str]        # length N_patients


class _FlatInputWrapper(nn.Module):
    """Wrap a ``HeteroGNNClassifier`` so its input becomes a single flat tensor.

    Captum's ``IntegratedGradients`` wants a callable where the input has a
    well-defined leading dimension it can interpolate over. HeteroData is a
    dict of tensors plus edge indices plus list attrs — not directly
    interpolatable. We concatenate the per-type ``x`` matrices along one
    axis, remember the slice offsets, and rebuild the HeteroData inside
    ``forward``.
    """

    def __init__(self, classifier: nn.Module, template: "HeteroData") -> None:
        super().__init__()
        self.classifier = classifier
        self.template = template

        self.node_types: list[str] = list(template.node_types)
        self.offsets: dict[str, tuple[int, int]] = {}
        off = 0
        for nt in self.node_types:
            n = int(template[nt].num_nodes)
            self.offsets[nt] = (off, off + n)
            off += n
        self.total_nodes = off
        self.feat_dim = int(template[self.node_types[0]].x.size(-1))

    def build_flat_input(self, data: "HeteroData") -> torch.Tensor:
        parts = [data[nt].x for nt in self.node_types]
        return torch.cat(parts, dim=0)

    def _single(self, x_flat: torch.Tensor) -> torch.Tensor:
        data = self.template.clone()
        for nt, (a, b) in self.offsets.items():
            data[nt].x = x_flat[a:b]
        out = self.classifier(data)
        return out["logit"].view(-1)  # (1,)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # Captum's IG batches ``n_steps`` interpolated copies of the input
        # along the leading axis; we don't try to run them as a PyG batch
        # (that would require rebuilding edge_index_dict too). Loop instead.
        if x_flat.dim() == 3:
            outs = [self._single(x_flat[i]) for i in range(x_flat.size(0))]
            return torch.cat(outs, dim=0)  # (B,)
        return self._single(x_flat)


def compute_node_attributions(
    classifier: nn.Module,
    patient_batches: list["HeteroData"],
    *,
    baseline: str = "zero",
    n_steps: int = 20,
) -> NodeAttribution:
    """Run Captum IG per patient; return a packaged attribution.

    Parameters
    ----------
    classifier : a trained ``HeteroGNNClassifier`` (``eval()`` is set
        internally — IG needs deterministic forward).
    patient_batches : list of single-graph HeteroData (one per patient).
    baseline : ``"zero"`` — the only sensible default for z-scored input where
        0 literally means "no signal". Other baselines (mean / random noise)
        are left as a v1 lever.
    n_steps : IG's interpolation steps (config ``xai.n_samples``).

    Returns
    -------
    NodeAttribution
        node / module annotations + (P, N_total) attribution matrix ready
        for the M5 heatmap and sanity checks.
    """
    from captum.attr import IntegratedGradients

    classifier.eval()
    template = patient_batches[0]
    wrapper = _FlatInputWrapper(classifier, template)

    node_ids: list[str] = []
    node_types: list[str] = []
    modules: list[str] = []
    for nt in wrapper.node_types:
        store = template[nt]
        node_ids.extend(list(store.node_ids))
        node_types.extend([nt] * int(store.num_nodes))
        modules.extend(list(store.module))

    ig = IntegratedGradients(wrapper)

    attrs = np.zeros((len(patient_batches), wrapper.total_nodes), dtype=np.float32)
    logits = np.zeros(len(patient_batches), dtype=np.float32)
    probs = np.zeros(len(patient_batches), dtype=np.float32)
    labels = np.zeros(len(patient_batches), dtype=np.float32)
    sample_ids: list[str] = []

    for i, data in enumerate(patient_batches):
        x_flat = wrapper.build_flat_input(data).unsqueeze(0).detach().requires_grad_(True)
        if baseline == "zero":
            baseline_t = torch.zeros_like(x_flat)
        else:
            raise ValueError(f"unsupported baseline: {baseline}")

        attribution = ig.attribute(x_flat, baselines=baseline_t, n_steps=n_steps)
        # attribution shape: (1, N_total, feat_dim) — sum over feat_dim
        attrs[i] = attribution.squeeze(0).sum(dim=-1).detach().cpu().numpy()

        with torch.no_grad():
            out = classifier(data)
        logits[i] = float(out["logit"].view(-1).item())
        probs[i] = float(torch.sigmoid(out["logit"].view(-1)).item())
        labels[i] = float(data.y.view(-1).item())
        sample_ids.append(str(getattr(data, "sample_id", f"sample_{i}")))

    return NodeAttribution(
        node_ids=node_ids,
        node_types=node_types,
        modules=modules,
        attributions=attrs,
        logits=logits,
        probs=probs,
        labels=labels,
        sample_ids=sample_ids,
    )
