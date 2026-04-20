"""M6 · T6.4 — TCGA-LUAD zero-shot transfer.

"Zero-shot" means: the model was trained on TCGA-COAD; we run inference on
LUAD without any fine-tuning. This is the cheapest falsifiability check we
can afford in the demo — if AUC collapses to ~0.5 on LUAD, the COAD result
was mostly cohort-specific overfitting.

PLAN §5 · M6 DoD: ``luad_auc ≥ 0.55``. The bar is deliberately low — we're
not claiming cross-cohort performance, we're ruling out "the model only
learned COAD-specific batch artifacts".
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def run_luad_zero_shot(
    run_dir: str | Path,
    *,
    config_path: str | Path = "configs/tcga_coad.yaml",
    luad_raw_dir: str | Path = "data/raw_luad",
) -> dict:
    """Load the best ckpt from ``run_dir``, score every LUAD sample.

    Returns a small dict summarizing the transfer: ``{n_samples, label_pos_rate,
    auc, accuracy, brier}``. Writes nothing to disk; the caller (report.py)
    embeds the result in the main run report.
    """
    import torch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from bovin_demo.data import (
        icd_readiness_label,
        load_coad,
        map_to_pathway_nodes,
    )
    from bovin_demo.data.dataset import build_patient_dataset
    from bovin_demo.eval.metrics import compute_metrics
    from bovin_demo.graph import load_graph
    from bovin_demo.model import build_classifier
    from bovin_demo.train.loop import _load_config
    from bovin_demo.xai.runner import _load_checkpoint

    run_dir = Path(run_dir)
    cfg = _load_config(config_path)

    graph = load_graph()
    # LUAD files share Xena format with COAD — load_coad is named by legacy but
    # reads any sampleMap directory it's handed.
    bundle = load_coad(luad_raw_dir)
    log.info("LUAD bundle: %d samples × %d genes", bundle.n_samples, bundle.n_genes)

    aligned, align_report = map_to_pathway_nodes(bundle.expr, graph)
    log.info("LUAD alignment hit_rate=%.3f", align_report.hit_rate)

    # Recompute the surrogate label on LUAD data — it's the same formula, just
    # scored against LUAD's gene-expression distribution.
    label, lrep = icd_readiness_label(bundle.expr)
    log.info("LUAD label pos_rate=%.3f thr=%.4f", lrep.pos_rate, lrep.threshold)

    common = aligned.index.intersection(label.index)
    aligned, label = aligned.loc[common], label.loc[common]

    dataset = build_patient_dataset(graph, aligned, label)
    samples = [dataset[i] for i in range(len(dataset))]

    probe = next(iter(PyGDataLoader(samples[: min(16, len(samples))], batch_size=16)))
    clf = build_classifier(
        probe,
        hidden_dim=int(cfg.model.hidden_dim),
        num_intra_layers=int(cfg.model.num_intra_layers),
        num_inter_layers=int(cfg.model.num_inter_layers),
        heads=int(cfg.model.attention_heads),
        dropout=float(cfg.model.dropout),
    )
    with torch.no_grad():
        clf(probe)

    ckpts = sorted((run_dir / "ckpt").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/ckpt")
    clf = _load_checkpoint(ckpts[-1], clf)
    clf.eval()

    probs: list[float] = []
    labels: list[int] = []
    loader = PyGDataLoader(samples, batch_size=32)
    with torch.no_grad():
        for batch in loader:
            logit = clf(batch)["logit"].view(-1)
            probs.extend(torch.sigmoid(logit).cpu().tolist())
            labels.extend(batch.y.view(-1).to(torch.int64).cpu().tolist())

    m = compute_metrics(np.asarray(labels), np.asarray(probs), bootstrap=0)
    return {
        "n_samples": len(labels),
        "label_pos_rate": float(np.mean(labels)),
        "auc": m["auc"]["mean"],
        "accuracy": m["accuracy"]["mean"],
        "f1": m["f1"]["mean"],
        "brier": m["brier"]["mean"],
        "alignment_hit_rate": align_report.hit_rate,
    }
