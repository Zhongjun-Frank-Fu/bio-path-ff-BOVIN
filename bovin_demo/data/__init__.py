"""TCGA data loading + pathway gene alignment + stratified split (M2)."""

from __future__ import annotations

from bovin_demo.data.gene_mapping import HitReport, map_to_pathway_nodes
from bovin_demo.data.labels import (
    LabelReport,
    icd_readiness_label,
    icd_readiness_signature,
)
from bovin_demo.data.split import Split, stratified_split
from bovin_demo.data.tcga_loader import CoadBundle, load_coad

__all__ = [
    "CoadBundle",
    "HitReport",
    "LabelReport",
    "Split",
    "icd_readiness_label",
    "icd_readiness_signature",
    "load_coad",
    "map_to_pathway_nodes",
    "stratified_split",
]
