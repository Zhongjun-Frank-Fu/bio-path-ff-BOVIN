"""Integrated Gradients on HeteroGNN + matplotlib heatmap (M5)."""

from __future__ import annotations

from bovin_demo.xai.aggregate import (
    aggregate_by_module,
    rank_nodes,
    select_top_tpr_patients,
)
from bovin_demo.xai.ig_captum import NodeAttribution, compute_node_attributions
from bovin_demo.xai.runner import run_xai
from bovin_demo.xai.viz import plot_heatmap

__all__ = [
    "NodeAttribution",
    "aggregate_by_module",
    "compute_node_attributions",
    "plot_heatmap",
    "rank_nodes",
    "run_xai",
    "select_top_tpr_patients",
]
