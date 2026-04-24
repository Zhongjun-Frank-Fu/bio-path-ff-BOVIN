"""A2-M4 · T4.2 — drive the LOCO 6-fold sweep for Aim 2.

Reads ``configs/ici_pool.yaml``, iterates over the labeled cohorts in the
pool, and for each holdout runs ``run_training`` with
``split.kind=loco`` + ``holdout_cohort=<cohort>``. Writes one
``metrics.json`` per fold under ``outputs/<ts>_seed<N>_loco_<cohort>/`` and
an aggregate ``loco_summary.json`` at the end.

Usage
-----
    python tools/run_ici_loco.py              # all labeled folds, seed 42
    python tools/run_ici_loco.py --seeds 42 1337 2024
    python tools/run_ici_loco.py --folds gide_prjeb23709 hugo_gse78220
    python tools/run_ici_loco.py --max-epochs 60     # quick pass
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bovin_demo.data import TIER_A_COHORTS, load_ici_pool  # noqa: E402
from bovin_demo.train.loop import run_training  # noqa: E402


def _eligible_folds(config_path: Path) -> list[str]:
    """Cohorts that survive label-filtering — the ones LOCO can actually use."""
    from bovin_demo.train.loop import _load_config

    cfg = _load_config(config_path)
    ici = cfg.data.get("ici", {})
    cohorts = list(ici.get("cohorts", TIER_A_COHORTS))
    pool = load_ici_pool(
        cohorts=cohorts,
        raw_dir=ici.get("raw_dir", "data/raw_ici"),
        aliases_csv=ici.get("aliases_csv",
                            "bovin_demo/data/static/bovin_gene_aliases.csv"),
        filter_timepoint=ici.get("filter_timepoint", "pre"),
        require_label=True,
    )
    return pool.cohorts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/ici_pool.yaml")
    p.add_argument("--seeds",  nargs="+", type=int, default=[42])
    p.add_argument("--folds",  nargs="+", default=None,
                   help="cohort IDs to iterate; default = eligible labeled cohorts")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="override cfg.train.max_epochs (useful for smoke runs)")
    p.add_argument("--output-root", default=None,
                   help="override cfg.paths.output_root; default = outputs/")
    args = p.parse_args()

    config_path = ROOT / args.config
    folds = args.folds or _eligible_folds(config_path)
    print(f"[plan] {len(args.seeds)} seed(s) × {len(folds)} fold(s) = "
          f"{len(args.seeds) * len(folds)} runs")
    print(f"[plan] seeds: {args.seeds}")
    print(f"[plan] folds: {folds}")

    results: list[dict] = []
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    for seed in args.seeds:
        for holdout in folds:
            print(f"\n{'='*70}\n[fold] seed={seed}  holdout={holdout}\n{'='*70}")
            r = run_training(
                str(config_path),
                seed=seed,
                max_epochs_override=args.max_epochs,
                output_root=args.output_root,
                holdout_cohort_override=holdout,
            )
            rec = r.to_json()
            rec["fold_key"] = f"seed{seed}_{holdout}"
            results.append(rec)
            print(f"[fold OK] seed={seed} holdout={holdout}  "
                  f"test_auc={r.test_auc:.4f}  "
                  f"baseline_auc={r.baseline_test_auc:.4f}  "
                  f"gap={r.test_auc - r.baseline_test_auc:+.4f}")

    # Aggregate.
    out_root = Path(args.output_root or "outputs")
    summary_path = out_root / f"{stamp}_loco_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if results:
        aucs = [r["test_auc"] for r in results]
        baselines = [r["baseline_test_auc"] for r in results]
        summary = {
            "n_runs": len(results),
            "seeds": args.seeds,
            "folds": folds,
            "gnn_mean_auc":       float(sum(aucs) / len(aucs)),
            "gnn_min_auc":        float(min(aucs)),
            "gnn_max_auc":        float(max(aucs)),
            "baseline_mean_auc":  float(sum(baselines) / len(baselines)),
            "gap_mean":           float(sum(a - b for a, b in zip(aucs, baselines)) / len(aucs)),
            "per_fold": results,
        }
    else:
        summary = {"n_runs": 0, "note": "no folds ran"}

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {summary_path}")
    if results:
        print(f"[done] GNN mean test_auc = {summary['gnn_mean_auc']:.4f}  "
              f"(min {summary['gnn_min_auc']:.4f} / max {summary['gnn_max_auc']:.4f})")
        print(f"[done] Baseline mean     = {summary['baseline_mean_auc']:.4f}")
        print(f"[done] Mean gap (GNN-MLP) = {summary['gap_mean']:+.4f}  "
              f"(A2-DoD #3 target ≥ +0.03)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
