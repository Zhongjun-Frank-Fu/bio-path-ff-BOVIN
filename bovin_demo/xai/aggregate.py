"""M5 · T5.2 — aggregate per-node attributions into patient-picking + module rolls.

Two rollups we need for the demo:

1. **Module-level** per-patient: sum of ``|attr|`` over all nodes in each
   module → ``(P, 11)`` matrix. This is what the heatmap renders and what
   DoD #4 ("M4 DAMP in top-3 modules") reads off.
2. **Node-level** population: mean of ``|attr|`` over patients → a single
   ranking of node_ids. DoD #5 ("CRT/HMGB1 in top-5") reads off this.

Patient selection (top-TPR per PLAN §6 · T5.2): we want patients the model
calls *confidently right* on the positive class — those are the clearest
XAI subjects. Concretely: filter to ``label == 1``, sort by ``prob``
descending, take the top-N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bovin_demo.xai.ig_captum import NodeAttribution


DEFAULT_MODULE_ORDER: tuple[str, ...] = tuple(f"M{i}" for i in range(1, 12))


@dataclass
class ModuleRollup:
    module_ids: list[str]                # ordered columns of ``matrix``
    matrix: np.ndarray                   # (P, 11) — sum of |attr| per module
    mean_per_module: np.ndarray          # (11,) — mean over patients
    top_modules: list[str]               # module_ids sorted by mean desc


@dataclass
class NodeRanking:
    node_ids: list[str]                  # ordered by mean |attr| desc
    mean_abs_attr: np.ndarray            # (N_total,) in that order
    modules: list[str]                   # per-node module, aligned with node_ids


def select_top_tpr_patients(
    attr: NodeAttribution,
    n: int = 20,
    require_positive: bool = True,
) -> np.ndarray:
    """Return the flat indices of patients to render in the heatmap.

    Strategy: filter by ``label == 1`` (true positives only — we want XAI
    that explains "why did the model call this ICD-ready?"), then sort by
    predicted probability descending.
    """
    order = np.argsort(-attr.probs)
    if require_positive:
        mask = attr.labels[order] == 1
        order = order[mask]
    return order[: min(n, order.size)]


def aggregate_by_module(
    attr: NodeAttribution,
    module_order: tuple[str, ...] = DEFAULT_MODULE_ORDER,
) -> ModuleRollup:
    """Sum |attr| within each module per patient. Module ordering is fixed
    (M1..M11) so the heatmap axis is stable across runs."""
    # (P, N_total) → (P, 11)
    matrix = np.zeros((attr.attributions.shape[0], len(module_order)), dtype=np.float32)
    modules_arr = np.array(attr.modules)
    abs_attr = np.abs(attr.attributions)
    for j, mid in enumerate(module_order):
        col_mask = modules_arr == mid
        if not col_mask.any():
            continue
        matrix[:, j] = abs_attr[:, col_mask].sum(axis=1)

    mean_per_module = matrix.mean(axis=0) if matrix.size else np.zeros(len(module_order))
    order = np.argsort(-mean_per_module)
    return ModuleRollup(
        module_ids=list(module_order),
        matrix=matrix,
        mean_per_module=mean_per_module,
        top_modules=[module_order[i] for i in order],
    )


def rank_nodes(attr: NodeAttribution) -> NodeRanking:
    """Mean |attr| over patients, descending by node."""
    mean_abs = np.mean(np.abs(attr.attributions), axis=0)  # (N_total,)
    order = np.argsort(-mean_abs)
    return NodeRanking(
        node_ids=[attr.node_ids[i] for i in order],
        mean_abs_attr=mean_abs[order],
        modules=[attr.modules[i] for i in order],
    )
