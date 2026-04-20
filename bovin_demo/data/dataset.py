"""M4 · T4.2 — one ``HeteroData`` per patient, ready for PyG DataLoader.

Feature encoding
----------------
For every node, ``x`` is a 2-tuple ``[z_expr, observed_flag]``:

* ``z_expr`` — z-scored expression (across patients) for that node, or 0.
* ``observed_flag`` — 1.0 if the patient had a measurement for this node,
  0.0 otherwise (non-observable therapy nodes, or genes absent from the
  alignment).

The flag lets the encoder distinguish "zero is a real measurement" from
"zero is padding". Without it the two cases collapse and the encoder bias
has to carry both, which hurts small-dataset learning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from bovin_demo.graph.loader import GraphDict

if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset
    from torch_geometric.data import HeteroData


def _patient_dataset_class():
    """Return the concrete class — instantiated lazily so ``import bovin_demo.data``
    does not force ``torch`` on boxes that only need the pure-pandas helpers."""

    import torch
    from torch.utils.data import Dataset as _Dataset

    from bovin_demo.graph.heterodata import to_heterodata

    class PatientGraphDataset(_Dataset):
        FEAT_DIM = 2  # [z_expr, observed_flag]

        def __init__(
            self,
            graph: GraphDict,
            aligned_expr: pd.DataFrame,
            labels: pd.Series,
        ) -> None:
            super().__init__()
            if list(aligned_expr.index) != list(labels.index):
                raise ValueError("aligned_expr and labels must share the same sample index")
            self.graph = graph
            self.sample_ids = list(aligned_expr.index)
            self.labels = labels.astype(np.float32).to_numpy()

            # Z-score per node across patients, preserving NaN for "no measurement".
            mu = aligned_expr.mean(axis=0, skipna=True)
            sd = aligned_expr.std(axis=0, skipna=True, ddof=0).replace(0, 1.0)
            self._z = ((aligned_expr - mu) / sd).fillna(0.0).astype(np.float32)
            self._observed = aligned_expr.notna().astype(np.float32)

            # Build the template HeteroData once and remember the per-type node
            # order so __getitem__ can lay out features deterministically.
            gen = torch.Generator().manual_seed(0)
            self._template = to_heterodata(graph, feat_dim=self.FEAT_DIM, generator=gen)
            self._ntype_node_ids: dict[str, list[str]] = {
                nt: list(self._template[nt].node_ids) for nt in self._template.node_types
            }

        def __len__(self) -> int:
            return len(self.sample_ids)

        def __getitem__(self, idx: int) -> "HeteroData":
            sample_id = self.sample_ids[idx]
            z_row = self._z.loc[sample_id] if sample_id in self._z.index else None
            obs_row = self._observed.loc[sample_id] if sample_id in self._observed.index else None

            data = self._template.clone()
            for ntype, node_ids in self._ntype_node_ids.items():
                feats = np.zeros((len(node_ids), self.FEAT_DIM), dtype=np.float32)
                for i, nid in enumerate(node_ids):
                    if z_row is not None and nid in z_row.index:
                        feats[i, 0] = float(z_row[nid])
                        feats[i, 1] = float(obs_row[nid]) if obs_row is not None else 0.0
                data[ntype].x = torch.from_numpy(feats)
            data.y = torch.tensor([self.labels[idx]], dtype=torch.float32)
            data.sample_id = sample_id
            return data

    return PatientGraphDataset


def build_patient_dataset(graph: GraphDict, aligned_expr: pd.DataFrame, labels: pd.Series):
    """Public factory — keeps ``torch`` import lazy."""
    cls = _patient_dataset_class()
    return cls(graph, aligned_expr, labels)
