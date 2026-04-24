"""TCGA + ICI data loading · gene alignment · splits · label helpers."""

from __future__ import annotations

from bovin_demo.data.gene_mapping import HitReport, map_to_pathway_nodes
from bovin_demo.data.ici_loader import (
    ICIBundle,
    ICIPoolBundle,
    TIER_A_COHORTS,
    load_gene_aliases,
    load_ici_cohort,
    load_ici_pool,
)
from bovin_demo.data.labels import (
    LabelReport,
    crt_quartile_label,
    icd_readiness_label,
    icd_readiness_signature,
    recist_binary_label,
)
from bovin_demo.data.split import Split, leave_one_cohort_out, stratified_split
from bovin_demo.data.tcga_loader import CoadBundle, load_coad

__all__ = [
    "CoadBundle",
    "HitReport",
    "ICIBundle",
    "ICIPoolBundle",
    "LabelReport",
    "Split",
    "TIER_A_COHORTS",
    "crt_quartile_label",
    "icd_readiness_label",
    "icd_readiness_signature",
    "leave_one_cohort_out",
    "load_coad",
    "load_gene_aliases",
    "load_ici_cohort",
    "load_ici_pool",
    "map_to_pathway_nodes",
    "recist_binary_label",
    "stratified_split",
]
