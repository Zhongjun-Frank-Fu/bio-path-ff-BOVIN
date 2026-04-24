"""Train + XAI on a cleaner label: CRT top-25% vs bot-25%.

Rationale: our composite surrogate fails Peng Fig.1C replication under
median split. CRT-alone + quartile split does replicate Peng. This
experiment asks: when we train on the *cleaner* label, does the XAI story
stay the same (M4 DAMP top module, CRT top node)?

Writes to outputs/crt_quartile_seed42/ parallel to the main run.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import torch

from bovin_demo.data import (
    crt_quartile_label,
    load_coad,
    map_to_pathway_nodes,
    stratified_split,
)
from bovin_demo.data.dataset import build_patient_dataset
from bovin_demo.graph import load_graph
from bovin_demo.model import build_classifier
from bovin_demo.train.lit_module import build_lit_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


SEED = 42
MAX_EPOCHS = 80
HIDDEN = 64
BATCH = 32
LR = 2e-3
WD = 1e-4
PATIENCE = 15


def _subset(dataset, indices):
    return torch.utils.data.Subset(dataset, indices.tolist())


def main() -> None:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch_geometric.loader import DataLoader as PyGDataLoader

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"outputs/{stamp}_crt_quartile_seed{SEED}")
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    pl.seed_everything(SEED, workers=True)
    graph = load_graph()
    bundle = load_coad("data/raw")

    aligned, align_rep = map_to_pathway_nodes(bundle.expr, graph)
    log.info("alignment hit_rate = %.3f", align_rep.hit_rate)

    label, lrep = crt_quartile_label(bundle.expr)
    keep = label.dropna().index
    aligned = aligned.loc[keep]
    label = label.loc[keep].astype(np.int64)
    log.info("CRT-quartile cohort: %d patients (top25%% + bot25%%), pos_rate=%.3f",
             len(label), lrep.pos_rate)

    split = stratified_split(label, seed=SEED)
    log.info("split: %s", split.sizes())

    dataset = build_patient_dataset(graph, aligned, label)
    train_ds = _subset(dataset, split.train_idx)
    val_ds = _subset(dataset, split.val_idx)
    test_ds = _subset(dataset, split.test_idx)

    train_loader = PyGDataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=BATCH, shuffle=False)
    test_loader = PyGDataLoader(test_ds, batch_size=BATCH, shuffle=False)

    probe = next(iter(train_loader))
    model = build_classifier(
        probe, hidden_dim=HIDDEN, num_intra_layers=2, num_inter_layers=1,
        heads=4, dropout=0.25,
    )
    with torch.no_grad():
        model(probe)

    y_train = label.to_numpy()[split.train_idx]
    pos_w = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
    log.info("pos_weight = %.3f", pos_w)

    lit = build_lit_module(model, lr=LR, weight_decay=WD, max_epochs=MAX_EPOCHS,
                           pos_weight=pos_w)

    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir / "ckpt", filename="best",
        monitor="val_auc", mode="max", save_top_k=1,
    )
    early_cb = EarlyStopping(monitor="val_auc", mode="max", patience=PATIENCE)
    csv_logger = CSVLogger(save_dir=str(run_dir), name="logs", version="")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[ckpt_cb, early_cb],
        logger=csv_logger,
        enable_progress_bar=False,
        log_every_n_steps=1,
        accelerator="cpu", devices=1,
    )
    trainer.fit(lit, train_loader, val_loader)
    best_val = float(ckpt_cb.best_model_score.item()) if ckpt_cb.best_model_score else float("nan")
    log.info("best val_auc = %.4f @ epoch %d", best_val, trainer.current_epoch)

    test_metrics = trainer.test(lit, dataloaders=test_loader, verbose=False)[0]

    result = {
        "run_dir": str(run_dir),
        "label_kind": "crt_quartile",
        "cohort_size": int(len(label)),
        "best_val_auc": best_val,
        "test_auc": float(test_metrics.get("test_auc", float("nan"))),
        "test_loss": float(test_metrics.get("test_loss", float("nan"))),
        "seed": SEED,
        "epochs_run": int(trainer.current_epoch),
        **split.sizes(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info("wrote %s/metrics.json", run_dir)

    # ---------------------------------------------------------------------
    # Run XAI on this ckpt
    # ---------------------------------------------------------------------
    log.info("running XAI on val fold...")
    val_samples = [dataset[int(i)] for i in split.val_idx]

    # Reload best ckpt weights
    from bovin_demo.xai.aggregate import aggregate_by_module, rank_nodes, select_top_tpr_patients
    from bovin_demo.xai.ig_captum import compute_node_attributions
    from bovin_demo.xai.runner import _load_checkpoint
    from bovin_demo.xai.viz import plot_heatmap

    clf_eval = build_classifier(probe, hidden_dim=HIDDEN, num_intra_layers=2,
                                 num_inter_layers=1, heads=4, dropout=0.25)
    with torch.no_grad():
        clf_eval(probe)
    clf_eval = _load_checkpoint(ckpt_cb.best_model_path, clf_eval)
    clf_eval.eval()

    attr = compute_node_attributions(clf_eval, val_samples, n_steps=20)
    idx = select_top_tpr_patients(attr, n=min(20, len(val_samples)), require_positive=True)
    if idx.size == 0:
        idx = select_top_tpr_patients(attr, n=min(20, len(val_samples)), require_positive=False)

    module_roll = aggregate_by_module(attr)
    node_rank = rank_nodes(attr)

    (run_dir / "xai").mkdir(exist_ok=True)
    plot_heatmap(
        matrix=module_roll.matrix[idx],
        patient_ids=[attr.sample_ids[i] for i in idx],
        module_ids=module_roll.module_ids,
        out_path=run_dir / "xai/xai_heatmap.png",
        title="BOVIN · CRT-quartile label · module × patient attribution",
    )

    landmarks = ["crt", "hmgb1", "st6gal1"]
    top5 = node_rank.node_ids[:5]
    sanity = {
        "label_kind": "crt_quartile",
        "top5_nodes": top5,
        "top3_modules": module_roll.top_modules[:3],
        "landmarks_in_top5": [n for n in landmarks if n in top5],
        "dod_4_m4_in_top3": "M4" in module_roll.top_modules[:3],
        "dod_5_landmark_in_top5": any(n in top5 for n in landmarks),
        "n_patients_heatmap": int(idx.size),
    }
    (run_dir / "xai/sanity.json").write_text(json.dumps(sanity, indent=2))
    log.info("sanity = %s", sanity)

    print("\n" + "="*70)
    print("COMPARISON · surrogate vs crt_quartile")
    print("="*70)
    print(f"{'metric':30s}  {'surrogate (original)':24s}  {'crt_quartile':20s}")
    print(f"{'cohort size':30s}  {'329':24s}  {len(label)}")
    print(f"{'best val_auc':30s}  {'0.9660':24s}  {best_val:.4f}")
    print(f"{'test_auc':30s}  {'0.9688':24s}  {result['test_auc']:.4f}")
    print(f"{'top3 modules':30s}  {'[M4, M6, M9]':24s}  {module_roll.top_modules[:3]}")
    print(f"{'top5 nodes':30s}  {'[crt,cd47,hmgb1,hsp70,cd24]':24s}  {top5}")
    print(f"{'M4 DAMP in top-3':30s}  {'✅':24s}  {'✅' if sanity['dod_4_m4_in_top3'] else '❌'}")
    print(f"{'landmark in top-5':30s}  {'✅ [crt, hmgb1]':24s}  "
          f"{'✅' if sanity['dod_5_landmark_in_top5'] else '❌'}  "
          f"{sanity['landmarks_in_top5']}")


if __name__ == "__main__":
    main()
