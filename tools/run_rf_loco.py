"""A2-M4.1 · standalone RandomForest LOCO baseline (no GNN retraining).

Mirrors the per-seed × per-fold structure of :mod:`tools.run_ici_loco` but
drops all Lightning / HeteroGNN infrastructure — just sklearn RF on the
pooled 72-feature matrix, same splits, same seeds.

Why standalone: after adding RF to ``run_training``, retraining the full
15-run sweep would duplicate ~1h of GNN compute per seed. The GNN +
BaselineMLP numbers in ``loco_3seed_merged.json`` are already authoritative;
this script only produces the missing RF column.

Output: ``outputs/<ts>_rf_loco_summary.json`` with per-seed × per-fold RF
test AUC, plus a per-cohort mean.

Usage
-----
    python tools/run_rf_loco.py --seeds 42 1337 2024
    python tools/run_rf_loco.py --seeds 42 --config configs/ici_pool.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ici_pool.yaml")
    p.add_argument("--seeds",  nargs="+", type=int, default=[42])
    p.add_argument("--folds",  nargs="+", default=None,
                   help="cohort IDs to iterate; default = eligible labeled cohorts")
    p.add_argument("--output-root", default="outputs")
    args = p.parse_args()

    # Import after warnings filter to keep sklearn import silent.
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    from bovin_demo.data import TIER_A_COHORTS, load_ici_pool
    from bovin_demo.data.build import build_data_and_split
    from bovin_demo.train.loop import _load_config

    config_path = ROOT / args.config
    cfg = _load_config(config_path)

    # Which cohorts survive label filtering (== eligible LOCO folds)?
    ici_cfg = cfg.data.get("ici", {}) if "ici" in cfg.data else {}
    pool = load_ici_pool(
        cohorts=list(ici_cfg.get("cohorts", TIER_A_COHORTS)),
        raw_dir=ici_cfg.get("raw_dir", "data/raw_ici"),
        aliases_csv=ici_cfg.get("aliases_csv",
                                "bovin_demo/data/static/bovin_gene_aliases.csv"),
        filter_timepoint=ici_cfg.get("filter_timepoint", "pre"),
        require_label=True,
    )
    folds = args.folds or pool.cohorts
    print(f"[plan] seeds={args.seeds}  folds={folds}")
    print(f"[plan] {len(args.seeds)} × {len(folds)} = {len(args.seeds) * len(folds)} runs")
    print()

    per_fold_records: list[dict] = []
    for seed in args.seeds:
        for holdout in folds:
            prep = build_data_and_split(
                cfg, seed=seed, holdout_cohort_override=holdout,
            )
            X = prep.aligned.fillna(0.0).to_numpy()
            y = prep.label.to_numpy().astype(float)
            split = prep.split

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
            auc = float(roc_auc_score(y_test, probs)) if len(set(y_test.tolist())) > 1 else 0.5

            rec = {
                "seed": seed,
                "holdout_cohort": holdout,
                "n_train": int(len(y_train)),
                "n_test":  int(len(y_test)),
                "test_pos_rate": float(np.mean(y_test)),
                "rf_test_auc": auc,
            }
            per_fold_records.append(rec)
            print(f"  seed={seed}  holdout={holdout:24s}  "
                  f"n_test={rec['n_test']:>3d}  RF AUC={auc:.4f}")

    # Aggregate per cohort.
    per_cohort: dict[str, list[float]] = {}
    for r in per_fold_records:
        per_cohort.setdefault(r["holdout_cohort"], []).append(r["rf_test_auc"])

    per_cohort_summary = [
        {
            "holdout_cohort": c,
            "n_seeds": len(vs),
            "rf_mean_auc": st.fmean(vs),
            "rf_sd": st.pstdev(vs) if len(vs) > 1 else 0.0,
            "n_test": next(r["n_test"] for r in per_fold_records if r["holdout_cohort"] == c),
        }
        for c, vs in sorted(per_cohort.items(), key=lambda kv: -st.fmean(kv[1]))
    ]

    all_aucs = [r["rf_test_auc"] for r in per_fold_records]
    summary = {
        "seeds": args.seeds,
        "folds": folds,
        "n_runs": len(per_fold_records),
        "per_cohort": per_cohort_summary,
        "global": {
            "rf_mean_auc": st.fmean(all_aucs),
            "rf_sd": st.pstdev(all_aucs) if len(all_aucs) > 1 else 0.0,
        },
        "per_fold_records": per_fold_records,
    }

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_root) / f"{stamp}_rf_loco_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"[done] wrote {out_path}")
    print()
    print("per-cohort:")
    for c in per_cohort_summary:
        print(f"  {c['holdout_cohort']:24s}  N_test={c['n_test']:>3d}  "
              f"RF={c['rf_mean_auc']:.3f}±{c['rf_sd']:.3f}")
    g = summary["global"]
    print()
    print(f"global RF mean AUC = {g['rf_mean_auc']:.3f} ± {g['rf_sd']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
