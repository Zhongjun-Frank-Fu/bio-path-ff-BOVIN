"""M4 · T4.2–T4.5 — end-to-end training orchestrator.

Responsibilities
----------------
1. Read YAML config via OmegaConf (supports `defaults:` inheritance).
2. Seed everything (``seed_everything`` covers torch / numpy / python / PyG).
3. Build the TCGA-COAD dataset from ``data/raw``.
4. Instantiate HeteroGNNClassifier + LitBovinModule.
5. Fit with EarlyStopping (val_auc, mode=max) + ModelCheckpoint.
6. Evaluate on test fold, plus a pure-MLP baseline for DoD #3.
7. Dump ``metrics.json``, ``training.log``, ``ckpt/*.ckpt``, CSV logs under
   ``outputs/YYYYMMDD_HHMMSS_seed{N}/``.

Per the plan's "reproducibility contract" every run gets its own timestamped
output directory so seed-sweep runs (T4.5) don't clobber each other.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig


log = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    run_dir: Path
    best_val_auc: float
    test_auc: float
    test_loss: float
    baseline_test_auc: float
    num_train: int
    num_val: int
    num_test: int
    seed: int
    epochs_run: int
    # A2-M4 fields (default to None so demo TCGA runs still serialize cleanly).
    data_source: str | None = None
    split_kind: str | None = None
    holdout_cohort: str | None = None
    # A2-M4.1 · RandomForest as a second flat-feature baseline alongside MLP.
    rf_test_auc: float | None = None

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["run_dir"] = str(self.run_dir)
        return out


def _load_config(path: str | Path) -> "DictConfig":
    from omegaconf import OmegaConf

    p = Path(path)
    cfg = OmegaConf.load(p)
    defaults = cfg.pop("defaults", None)
    if defaults is not None:
        merged = OmegaConf.create({})
        for entry in defaults:
            name = entry if isinstance(entry, str) else next(iter(entry.values()))
            parent = OmegaConf.load(p.parent / f"{name}.yaml")
            merged = OmegaConf.merge(merged, parent)
        cfg = OmegaConf.merge(merged, cfg)
    return cfg


def _make_run_dir(root: Path, seed: int) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{stamp}_seed{seed}"
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def _init_file_logger(run_dir: Path) -> logging.Handler:
    fh = logging.FileHandler(run_dir / "training.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)
    return fh


def _compute_pos_weight(y) -> float:
    import numpy as np

    y = np.asarray(y)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        return 1.0
    return neg / pos


def run_training(
    config_path: str | Path,
    seed: int = 42,
    *,
    max_epochs_override: int | None = None,
    output_root: str | Path | None = None,
    raw_dir_override: str | Path | None = None,
    holdout_cohort_override: str | None = None,
) -> TrainingResult:
    """Fit → validate → test → persist artifacts. Returns a summary.

    Parameters
    ----------
    config_path : path to a YAML config under ``configs/``.
    seed : global seed (default 42; T4.5 sweeps 42/1337/2024).
    max_epochs_override : cut training short for smoke tests.
    output_root : override ``outputs/`` root; useful for tests writing to tmp.
    raw_dir_override : point at a different ``data/raw``; used in tests
        against the synthetic Xena fixture.
    """
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from bovin_demo.data import (
        icd_readiness_label,
        leave_one_cohort_out,
        load_coad,
        load_ici_pool,
        map_to_pathway_nodes,
        recist_binary_label,
        stratified_split,
    )
    from bovin_demo.data.dataset import build_patient_dataset
    from bovin_demo.graph import load_graph
    from bovin_demo.model import build_classifier
    from bovin_demo.train.lit_module import build_lit_module

    cfg = _load_config(config_path)
    raw_dir = Path(raw_dir_override or cfg.paths.get("raw_dir", "data/raw"))
    out_root = Path(output_root or cfg.paths.get("output_root", "outputs"))
    max_epochs = int(max_epochs_override or cfg.train.max_epochs)

    run_dir = _make_run_dir(out_root, seed)
    fh = _init_file_logger(run_dir)
    log.info("config = %s", dict(cfg))
    log.info("seed   = %d", seed)
    log.info("run    = %s", run_dir)

    pl.seed_everything(seed, workers=True)

    graph = load_graph()
    log.info("graph: %d nodes / %d edges", len(graph["nodes"]), len(graph["edges"]))

    # --- Data source dispatch (A2-M4 T4.1) --------------------------------
    data_cfg = cfg.get("data", {}) if "data" in cfg else {}
    data_source = str(data_cfg.get("source", "tcga_coad_xena"))
    cohort_ids: "pd.Series | None" = None  # populated only in ici_pool path

    if data_source == "ici_pool":
        import pandas as pd  # local import; rest of file already uses np only

        ici_cfg = data_cfg.get("ici", {})
        # Allow the default cohorts list in ici_loader.TIER_A_COHORTS when config omits it.
        cohorts_cfg = ici_cfg.get("cohorts", None)
        pool_raw_dir = Path(raw_dir_override or ici_cfg.get("raw_dir", "data/raw_ici"))
        aliases_csv = ici_cfg.get("aliases_csv",
                                  "bovin_demo/data/static/bovin_gene_aliases.csv")
        filter_tp = ici_cfg.get("filter_timepoint", "pre")
        require_lbl = bool(ici_cfg.get("require_label", True))

        pool_kwargs = dict(
            raw_dir=pool_raw_dir,
            aliases_csv=aliases_csv,
            filter_timepoint=filter_tp,
            require_label=require_lbl,
        )
        if cohorts_cfg is not None:
            pool_kwargs["cohorts"] = list(cohorts_cfg)

        pool = load_ici_pool(**pool_kwargs)
        log.info("ICI pool: %d samples × %d genes · %d cohorts",
                 pool.n_samples, len(pool.genes), len(pool.cohorts))
        log.info("per-cohort hit rates: %s",
                 {k: f"{v:.1%}" for k, v in pool.per_cohort_hit_rates.items()})

        # pool.expr has HGNC symbols as columns; map_to_pathway_nodes converts
        # those to graph node_ids (lowercase, e.g. "crt") that PatientGraphDataset
        # expects. Missing nodes (e.g. type1_ifn from IFNA1/IFNB1 naming mismatch)
        # become NaN columns → z_expr=0, observed_flag=0 in the dataset.
        aligned, align_report = map_to_pathway_nodes(pool.expr, graph)
        log.info("alignment hit_rate=%.3f (%d/%d observable, %d misses)",
                 align_report.hit_rate, align_report.n_hits,
                 align_report.n_observable, len(align_report.misses))

        label, label_report = recist_binary_label(
            pool.clinical,
            response_col=str(cfg.label.get("response_col", "response_raw")),
            mapping=cfg.label.get("mapping", None),
        )
        log.info("label: pos_rate=%.3f (mapping misses: %s)",
                 label_report.pos_rate, label_report.genes_missing)

        cohort_ids = pool.clinical["cohort_id"]

    else:  # legacy demo path
        bundle = load_coad(raw_dir)
        log.info("TCGA: %d samples × %d genes", bundle.n_samples, bundle.n_genes)

        aligned, align_report = map_to_pathway_nodes(bundle.expr, graph)
        log.info(
            "alignment hit_rate=%.3f (%d/%d observable, %d misses)",
            align_report.hit_rate, align_report.n_hits,
            align_report.n_observable, len(align_report.misses),
        )

        label, label_report = icd_readiness_label(bundle.expr)
        log.info("label: pos_rate=%.3f thr=%.4f",
                 label_report.pos_rate, label_report.threshold)

    common = aligned.index.intersection(label.index)
    aligned = aligned.loc[common]
    label = label.loc[common]
    if cohort_ids is not None:
        cohort_ids = cohort_ids.loc[common]

    # --- Split dispatch (A2-M4 T4.2) --------------------------------------
    split_cfg = cfg.get("split", {}) if "split" in cfg else {}
    split_kind = str(split_cfg.get("kind", "stratified"))
    # Passing holdout_cohort_override at call-time forces LOCO mode even if the
    # config defaults to stratified — this is how the LOCO driver (tools/run_ici_loco.py)
    # iterates folds without rewriting the config for each.
    if holdout_cohort_override is not None:
        split_kind = "loco"
    holdout_cohort: str | None = None
    if split_kind == "loco":
        if cohort_ids is None:
            raise ValueError("split.kind='loco' requires data.source='ici_pool'")
        loco_cfg = split_cfg.get("loco", {})
        holdout_cohort = (
            holdout_cohort_override
            or loco_cfg.get("holdout_cohort")
        )
        if holdout_cohort is None:
            raise ValueError(
                "LOCO mode needs split.loco.holdout_cohort set "
                "(or pass --holdout-cohort)"
            )
        split = leave_one_cohort_out(
            cohort_ids, label,
            holdout_cohort=str(holdout_cohort),
            val_frac=float(loco_cfg.get("val_frac", 0.15)),
            seed=seed,
        )
        log.info("LOCO fold: holdout=%s  sizes=%s", holdout_cohort, split.sizes())
    else:
        split = stratified_split(label, seed=seed)
        log.info("stratified split sizes: %s", split.sizes())

    dataset = build_patient_dataset(graph, aligned, label)

    def _subset(indices):
        return torch.utils.data.Subset(dataset, indices.tolist())

    train_ds = _subset(split.train_idx)
    val_ds = _subset(split.val_idx)
    test_ds = _subset(split.test_idx)

    pos_w = _compute_pos_weight(label.to_numpy()[split.train_idx])
    log.info("pos_weight (balanced, train fold) = %.3f", pos_w)

    bs = int(cfg.train.get("batch_size", 32))
    train_loader = PyGDataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = PyGDataLoader(test_ds, batch_size=bs, shuffle=False)

    probe = next(iter(train_loader))
    model = build_classifier(
        probe,
        hidden_dim=int(cfg.model.hidden_dim),
        num_intra_layers=int(cfg.model.num_intra_layers),
        num_inter_layers=int(cfg.model.num_inter_layers),
        heads=int(cfg.model.attention_heads),
        dropout=float(cfg.model.dropout),
    )
    with torch.no_grad():
        model(probe)  # warm up lazy layers so the optimizer sees every parameter

    lit = build_lit_module(
        model,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        max_epochs=max_epochs,
        pos_weight=pos_w,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir / "ckpt",
        filename="best",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    early_cb = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=int(cfg.train.patience),
    )
    csv_logger = CSVLogger(save_dir=str(run_dir), name="logs", version="")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        default_root_dir=str(run_dir),
        callbacks=[ckpt_cb, early_cb],
        logger=csv_logger,
        enable_progress_bar=False,
        log_every_n_steps=1,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_val = (
        float(ckpt_cb.best_model_score.item())
        if ckpt_cb.best_model_score is not None
        else float("nan")
    )
    log.info("best val_auc = %.4f @ epoch %d", best_val, trainer.current_epoch)

    test_metrics = trainer.test(lit, dataloaders=test_loader, verbose=False)[0]

    baseline_auc = _train_baseline(
        aligned=aligned, label=label, split=split,
        seed=seed, max_epochs=max_epochs, bs=bs,
    )
    log.info("BaselineMLP test_auc = %.4f", baseline_auc)

    rf_auc = _train_baseline_rf(
        aligned=aligned, label=label, split=split, seed=seed,
    )
    log.info("BaselineRF  test_auc = %.4f", rf_auc)

    result = TrainingResult(
        run_dir=run_dir,
        best_val_auc=best_val,
        test_auc=float(test_metrics.get("test_auc", float("nan"))),
        test_loss=float(test_metrics.get("test_loss", float("nan"))),
        baseline_test_auc=float(baseline_auc),
        num_train=int(split.train_idx.size),
        num_val=int(split.val_idx.size),
        num_test=int(split.test_idx.size),
        seed=seed,
        epochs_run=int(trainer.current_epoch),
        data_source=data_source,
        split_kind=split_kind,
        holdout_cohort=holdout_cohort,
        rf_test_auc=float(rf_auc),
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(result.to_json(), indent=2), encoding="utf-8"
    )
    log.info("wrote %s", run_dir / "metrics.json")
    logging.getLogger().removeHandler(fh)
    return result


def _train_baseline(
    *,
    aligned,
    label,
    split,
    seed: int,
    max_epochs: int,
    bs: int,
) -> float:
    """Train the pure-MLP baseline (DoD #3 comparator) and return test AUC.

    Deliberately uses the same features, split, and pos_weight as the GNN —
    any gap is attributable to the graph structure, not feature choice.
    """
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset

    from bovin_demo.model import BaselineMLP

    torch.manual_seed(seed)
    np.random.seed(seed)

    X = aligned.fillna(0.0).to_numpy(dtype=np.float32)
    y = label.to_numpy(dtype=np.float32)

    X_train, y_train = X[split.train_idx], y[split.train_idx]
    X_val, y_val = X[split.val_idx], y[split.val_idx]
    X_test, y_test = X[split.test_idx], y[split.test_idx]

    mu, sd = X_train.mean(axis=0), X_train.std(axis=0) + 1e-6
    X_train = (X_train - mu) / sd
    X_val = (X_val - mu) / sd
    X_test = (X_test - mu) / sd

    pos_w = _compute_pos_weight(y_train)
    mlp = BaselineMLP(in_features=X.shape[1], hidden_dim=64, dropout=0.25)
    opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]))

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=bs, shuffle=True,
    )

    best_val = -1.0
    best_state = None
    for _ in range(max_epochs):
        mlp.train()
        for xb, yb in loader:
            logit = mlp(xb).view(-1)
            loss = loss_fn(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            probs = torch.sigmoid(mlp(torch.from_numpy(X_val)).view(-1)).numpy()
        val_auc = float(roc_auc_score(y_val, probs)) if len(set(y_val.tolist())) > 1 else 0.5
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}

    if best_state is not None:
        mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        probs_t = torch.sigmoid(mlp(torch.from_numpy(X_test)).view(-1)).numpy()
    return (
        float(roc_auc_score(y_test, probs_t))
        if len(set(y_test.tolist())) > 1
        else 0.5
    )


def _train_baseline_rf(
    *,
    aligned,
    label,
    split,
    seed: int,
) -> float:
    """Second flat-feature baseline — RandomForest on the same 72-feature vector.

    Purpose (plan §7 diagnostic): if RF ≈ MLP ≈ GNN on the same split, the
    256-patient real-RECIST pool has a signal *ceiling* that neither
    capacity nor graph structure can cross. Untuned defaults are intentional —
    this is a reference point, not a tuned competitor.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    X = aligned.fillna(0.0).to_numpy()
    y = label.to_numpy()

    X_train, y_train = X[split.train_idx], y[split.train_idx]
    X_test,  y_test  = X[split.test_idx],  y[split.test_idx]

    mu, sd = X_train.mean(axis=0), X_train.std(axis=0) + 1e-6
    X_train = (X_train - mu) / sd
    X_test  = (X_test  - mu) / sd

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    if len(set(y_test.tolist())) < 2:
        return 0.5
    return float(roc_auc_score(y_test, probs))
