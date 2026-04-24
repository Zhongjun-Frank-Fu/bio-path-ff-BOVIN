"""M5 · T5.1–T5.4 — end-to-end XAI run driven by a trained checkpoint.

``run_xai(run_dir)`` loads the best Lightning checkpoint from a previous
``run_training`` directory, rebuilds the same dataset / split / classifier,
runs Captum IG on the top-N true-positive validation patients, and writes:

    {run_dir}/xai/
        xai_heatmap.png       ← PLAN §1 deliverable
        node_attributions.csv ← (patient × node_id) long table
        module_summary.csv    ← (module, mean_abs_attr, rank)
        sanity.json           ← DoD #4 and #5 pass/fail snapshot
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


LANDMARK_NODES: tuple[str, ...] = ("crt", "hmgb1", "st6gal1")


def _load_checkpoint(ckpt_path: Path, model):
    """Populate ``model`` from a Lightning checkpoint state dict.

    We rebuild ``model`` from config (same hparams as training) and only
    use the ckpt for weights — this side-steps Lightning's
    ``load_from_checkpoint`` dance which wants the LitModule class present.
    """
    import torch

    state = torch.load(ckpt_path, map_location="cpu")
    sd = state["state_dict"] if "state_dict" in state else state
    # Lightning prefixes everything with ``model.`` — strip it.
    trimmed = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(trimmed, strict=False)
    log.info("loaded %d params (missing=%d unexpected=%d) from %s",
             len(trimmed), len(missing), len(unexpected), ckpt_path)
    return model


def run_xai(
    run_dir: str | Path,
    *,
    config_path: str | Path,
    raw_dir_override: str | Path | None = None,
    seed: int = 42,
    top_n_patients: int = 20,
    n_steps: int = 20,
    holdout_cohort_override: str | None = None,
) -> dict:
    """Orchestrate the full M5 pipeline against the ckpt produced by M4.

    Works for both ``data.source=tcga_coad_xena`` and ``data.source=ici_pool``
    via the shared :func:`bovin_demo.data.build.build_data_and_split` helper.
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from bovin_demo.data.build import build_data_and_split
    from bovin_demo.data.dataset import build_patient_dataset
    from bovin_demo.model import build_classifier
    from bovin_demo.train.loop import _load_config
    from bovin_demo.xai.aggregate import (
        aggregate_by_module,
        rank_nodes,
        select_top_tpr_patients,
    )
    from bovin_demo.xai.ig_captum import compute_node_attributions
    from bovin_demo.xai.viz import plot_heatmap

    run_dir = Path(run_dir)
    cfg = _load_config(config_path)
    prep = build_data_and_split(
        cfg, seed=seed,
        raw_dir_override=raw_dir_override,
        holdout_cohort_override=holdout_cohort_override,
    )
    graph = prep.graph
    aligned, label, split = prep.aligned, prep.label, prep.split

    dataset = build_patient_dataset(graph, aligned, label)
    val_samples = [dataset[int(i)] for i in split.val_idx]

    # Build the classifier shell from one batch, then load weights from ckpt.
    probe_loader = PyGDataLoader(val_samples[: min(16, len(val_samples))], batch_size=16)
    probe = next(iter(probe_loader))
    classifier = build_classifier(
        probe,
        hidden_dim=int(cfg.model.hidden_dim),
        num_intra_layers=int(cfg.model.num_intra_layers),
        num_inter_layers=int(cfg.model.num_inter_layers),
        heads=int(cfg.model.attention_heads),
        dropout=float(cfg.model.dropout),
    )
    # Warm lazy layers so load_state_dict sees every parameter slot.
    import torch
    with torch.no_grad():
        classifier(probe)

    ckpts = sorted((run_dir / "ckpt").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/ckpt")
    classifier = _load_checkpoint(ckpts[-1], classifier)
    classifier.eval()

    attr = compute_node_attributions(classifier, val_samples, n_steps=n_steps)

    # Pick top-N true positives and restrict the heatmap to those rows.
    idx = select_top_tpr_patients(attr, n=top_n_patients, require_positive=True)
    if idx.size == 0:
        log.warning("no label==1 val patients — falling back to top-prob overall")
        idx = select_top_tpr_patients(attr, n=top_n_patients, require_positive=False)

    module_roll = aggregate_by_module(attr)
    node_rank = rank_nodes(attr)

    out_dir = run_dir / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap
    heatmap_path = plot_heatmap(
        matrix=module_roll.matrix[idx],
        patient_ids=[attr.sample_ids[i] for i in idx],
        module_ids=module_roll.module_ids,
        out_path=out_dir / "xai_heatmap.png",
    )

    # CSVs
    node_rows = []
    for pi in idx:
        for ni, nid in enumerate(attr.node_ids):
            node_rows.append({
                "sample_id": attr.sample_ids[pi],
                "node_id": nid,
                "node_type": attr.node_types[ni],
                "module": attr.modules[ni],
                "attr": float(attr.attributions[pi, ni]),
            })
    pd.DataFrame(node_rows).to_csv(out_dir / "node_attributions.csv", index=False)
    pd.DataFrame({
        "module": module_roll.module_ids,
        "mean_abs_attr": module_roll.mean_per_module,
        "rank": [module_roll.top_modules.index(m) + 1 for m in module_roll.module_ids],
    }).to_csv(out_dir / "module_summary.csv", index=False)

    # ---- DoD sanity (PLAN §7 items #4 and #5) ---------------------------
    top5_nodes = node_rank.node_ids[:5]
    top3_modules = module_roll.top_modules[:3]
    landmarks_in_top5 = [n for n in LANDMARK_NODES if n in top5_nodes]
    sanity = {
        "dod_4_m4_damp_in_top3_modules": "M4" in top3_modules,
        "dod_5_landmark_in_top5_nodes": len(landmarks_in_top5) > 0,
        "top5_nodes": top5_nodes,
        "top3_modules": top3_modules,
        "landmarks_found": landmarks_in_top5,
        "n_patients": int(idx.size),
        "heatmap": str(heatmap_path),
    }
    (out_dir / "sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    log.info("wrote XAI artifacts under %s", out_dir)
    return sanity
