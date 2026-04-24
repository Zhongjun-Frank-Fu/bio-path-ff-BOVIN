"""A2-M6 · shared dataset/split build pipeline.

``build_data_and_split(cfg, seed, …)`` consolidates the load-align-label-split
flow that used to live inline in ``train/loop.py``, ``xai/runner.py``, and
``eval/report.py``. Extracting it avoids maintaining three copies as the
ici_pool and LOCO code paths diverge from the original TCGA-COAD demo.

Public surface
--------------
- :class:`PreparedData` — bundle of graph / aligned / label / split / cohort_ids
- :func:`build_data_and_split` — config-dispatched builder

Consumers stay decoupled: they receive ``aligned`` (samples × node_id DataFrame)
+ ``label`` (samples × {0,1}) + ``split`` (Split dataclass), regardless of
whether the source is TCGA-COAD or the ICI pool.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from bovin_demo.data.split import Split
    from bovin_demo.graph.loader import GraphDict


@dataclass
class PreparedData:
    graph: "GraphDict"
    aligned: "pd.DataFrame"             # samples × graph-node-id columns
    label:   "pd.Series"                # samples × {0.0, 1.0} (NaN filtered)
    split:   "Split"
    cohort_ids: "pd.Series | None" = None  # only populated for ici_pool
    align_hit_rate: float = 0.0
    label_pos_rate: float = 0.0
    data_source: str = ""


def build_data_and_split(
    cfg: Any,
    *,
    seed: int,
    raw_dir_override: str | Path | None = None,
    holdout_cohort_override: str | None = None,
) -> PreparedData:
    """Dispatch on ``cfg.data.source`` — return a :class:`PreparedData`.

    Supported ``cfg.data.source`` values:
        ``"tcga_coad_xena"`` (default) — demo COAD path with surrogate label
        ``"ici_pool"``                 — Aim 2 6-cohort pool with RECIST binary

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Already loaded via :func:`bovin_demo.train.loop._load_config`.
    seed : int
        Seed for stratified and LOCO val carve-out.
    raw_dir_override : path-like, optional
        Override ``cfg.paths.raw_dir`` (COAD) or ``cfg.data.ici.raw_dir`` (ICI).
    holdout_cohort_override : str, optional
        If given and source is ``ici_pool``, forces LOCO mode with this
        holdout cohort (matches ``run_training``'s override semantics).
    """
    import pandas as pd

    from bovin_demo.data import (
        icd_readiness_label,
        leave_one_cohort_out,
        load_coad,
        load_ici_pool,
        map_to_pathway_nodes,
        recist_binary_label,
        stratified_split,
    )
    from bovin_demo.graph import load_graph

    data_cfg = cfg.get("data", {}) if "data" in cfg else {}
    data_source = str(data_cfg.get("source", "tcga_coad_xena"))

    graph = load_graph()
    cohort_ids: "pd.Series | None" = None

    if data_source == "ici_pool":
        ici_cfg = data_cfg.get("ici", {})
        raw_dir = Path(raw_dir_override or ici_cfg.get("raw_dir", "data/raw_ici"))
        aliases_csv = ici_cfg.get(
            "aliases_csv", "bovin_demo/data/static/bovin_gene_aliases.csv"
        )
        pool_kwargs: dict[str, Any] = dict(
            raw_dir=raw_dir,
            aliases_csv=aliases_csv,
            filter_timepoint=ici_cfg.get("filter_timepoint", "pre"),
            require_label=bool(ici_cfg.get("require_label", True)),
        )
        if ici_cfg.get("cohorts", None) is not None:
            pool_kwargs["cohorts"] = list(ici_cfg["cohorts"])
        pool = load_ici_pool(**pool_kwargs)

        aligned, align_rep = map_to_pathway_nodes(pool.expr, graph)
        label, label_rep = recist_binary_label(
            pool.clinical,
            response_col=str(cfg.label.get("response_col", "response_raw")),
            mapping=cfg.label.get("mapping", None),
        )
        cohort_ids = pool.clinical["cohort_id"]
    else:
        raw_dir = Path(raw_dir_override or cfg.paths.get("raw_dir", "data/raw"))
        bundle = load_coad(raw_dir)
        aligned, align_rep = map_to_pathway_nodes(bundle.expr, graph)
        label, label_rep = icd_readiness_label(bundle.expr)

    common = aligned.index.intersection(label.index)
    aligned = aligned.loc[common]
    label = label.loc[common]
    if cohort_ids is not None:
        cohort_ids = cohort_ids.loc[common]

    # --- split dispatch ---
    split_cfg = cfg.get("split", {}) if "split" in cfg else {}
    split_kind = str(split_cfg.get("kind", "stratified"))
    if holdout_cohort_override is not None:
        split_kind = "loco"

    if split_kind == "loco":
        if cohort_ids is None:
            raise ValueError("split.kind='loco' requires data.source='ici_pool'")
        loco_cfg = split_cfg.get("loco", {})
        holdout = holdout_cohort_override or loco_cfg.get("holdout_cohort")
        if holdout is None:
            raise ValueError(
                "LOCO mode needs split.loco.holdout_cohort or an override"
            )
        split = leave_one_cohort_out(
            cohort_ids, label,
            holdout_cohort=str(holdout),
            val_frac=float(loco_cfg.get("val_frac", 0.15)),
            seed=seed,
        )
    else:
        split = stratified_split(label, seed=seed)

    return PreparedData(
        graph=graph,
        aligned=aligned,
        label=label,
        split=split,
        cohort_ids=cohort_ids,
        align_hit_rate=float(align_rep.hit_rate),
        label_pos_rate=float(label_rep.pos_rate),
        data_source=data_source,
    )
