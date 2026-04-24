"""A2-M5 · T5.2 — external cohort zero-shot transfer for an ICI pool ckpt.

Dual of :mod:`bovin_demo.eval.luad_transfer` but for the Aim 2 pool model:
load the best ckpt produced by ``run_training(cfg=ici_pool.yaml)``, run
inference on a held-out cohort (currently **Sade-Feldman GSE120575**
scRNA pseudobulk), and emit patient-level AUC + bootstrap CI.

Usage as a library::

    from bovin_demo.eval.external_transfer import run_sade_feldman_external
    metrics = run_sade_feldman_external(
        run_dir="outputs/20260423_042824_seed42",
        config_path="configs/ici_pool.yaml",
    )

Usage as a script::

    python -m bovin_demo.eval.external_transfer \\
        --run-dir outputs/20260423_042824_seed42 \\
        --config  configs/ici_pool.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def run_sade_feldman_external(
    run_dir: str | Path,
    *,
    config_path: str | Path = "configs/ici_pool.yaml",
    sade_raw_dir: str | Path | None = None,
    timepoint: str = "pre",
    bootstrap: int = 1000,
) -> dict:
    """Score every Sade-Feldman patient with the pool-trained ckpt.

    Returns a dict with ``{n_patients, label_pos_rate, auc, auc_ci,
    accuracy, brier, alignment_hit_rate, therapy_breakdown, caveat}``.

    The plan-pre-registered caveat (§2.1) is **CD45+ enrichment strips tumor
    cells, so CRT/HMGB1-driven predictions can't fully express**. The DoD
    threshold (≥ 0.55) is deliberately soft — a higher number would be nice
    but is not expected.
    """
    import torch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from bovin_demo.data.dataset import build_patient_dataset
    from bovin_demo.data.sade_loader import load_sade_feldman
    from bovin_demo.eval.metrics import compute_metrics
    from bovin_demo.graph import load_graph
    from bovin_demo.data import map_to_pathway_nodes
    from bovin_demo.model import build_classifier
    from bovin_demo.train.loop import _load_config
    from bovin_demo.xai.runner import _load_checkpoint

    run_dir = Path(run_dir)
    cfg = _load_config(config_path)

    graph = load_graph()
    bundle = load_sade_feldman(
        raw_dir=sade_raw_dir if sade_raw_dir else None,
        timepoint=timepoint,
    )
    log.info(
        "Sade bundle: %d patients · hit_rate=%.3f",
        bundle.n_patients, bundle.hit_rate,
    )

    # Drop unlabeled rows (Sade maps to R/NR via _DEFAULT_RECIST_TO_BINARY +
    # title-case aliases; anything else is NaN and can't be scored).
    labeled = bundle.clinical["ici_response"].notna()
    expr = bundle.expr.loc[labeled]
    clin = bundle.clinical.loc[labeled]
    if len(expr) == 0:
        raise RuntimeError("no labeled Sade-Feldman patients found")

    # bundle.expr has HGNC symbol columns; route through map_to_pathway_nodes
    # to get graph node_id columns (matches ici_pool forward-pass layout).
    aligned, align_report = map_to_pathway_nodes(expr, graph)
    log.info(
        "graph-alignment hit_rate=%.3f (%d/%d observable)",
        align_report.hit_rate, align_report.n_hits, align_report.n_observable,
    )

    label = clin["ici_response"].astype(float).rename("label")
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

    metrics = compute_metrics(
        np.asarray(labels), np.asarray(probs),
        bootstrap=bootstrap, seed=42,
    )
    auc_block = metrics["auc"]
    therapy_breakdown = clin["treatment"].value_counts().to_dict()

    return {
        "cohort": "sade_feldman_gse120575",
        "n_patients":        int(len(labels)),
        "label_pos_rate":    float(np.mean(labels)),
        "auc":               float(auc_block["mean"]),
        "auc_ci_95":         [float(auc_block["ci_lo"]), float(auc_block["ci_hi"])],
        "accuracy":          float(metrics["accuracy"]["mean"]),
        "f1":                float(metrics["f1"]["mean"]),
        "brier":             float(metrics["brier"]["mean"]),
        "alignment_hit_rate":float(align_report.hit_rate),
        "therapy_breakdown": therapy_breakdown,
        "timepoint":         timepoint,
        "passes_dod_4":      float(auc_block["mean"]) >= 0.55,
        "caveat": (
            "Sade-Feldman uses CD45+ sorting — tumor cells are excluded, so the "
            "BOVIN ICD axis (CRT/HMGB1/HSP) is systematically under-represented "
            "in the input. Low AUC here is the pre-registered expectation "
            "(plan §2.1)."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--config",  default="configs/ici_pool.yaml")
    p.add_argument("--sade-raw-dir", default=None,
                   help="override data/raw_ici/sade_feldman_gse120575/")
    p.add_argument("--timepoint", default="pre",
                   choices=["pre", "post", "both"])
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    metrics = run_sade_feldman_external(
        args.run_dir,
        config_path=args.config,
        sade_raw_dir=args.sade_raw_dir,
        timepoint=args.timepoint,
        bootstrap=args.bootstrap,
    )
    out = args.output or (Path(args.run_dir) / "external_sade_feldman.json")
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[done] wrote {out}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
