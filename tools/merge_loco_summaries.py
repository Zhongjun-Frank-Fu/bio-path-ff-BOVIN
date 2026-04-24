"""A2-M4 · aggregate multiple LOCO summary JSONs into a 3-seed summary.

Given two or more ``*_loco_summary.json`` files (each covering one or more
seeds × some folds), this script merges them into a single summary with
per-fold mean±SD across seeds, plus the global DoD check (plan §3.1):

  - A2-DoD #2 — LOCO mean AUC ≥ 0.60
  - A2-DoD #3 — GNN vs BaselineMLP gap ≥ +0.03
  - A2-DoD #7 — per-fold seed SD < 0.08

Usage
-----
    python tools/merge_loco_summaries.py \
        outputs/20260423_045502_loco_summary.json \
        outputs/<new_summary>.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("summaries", nargs="+", type=Path,
                   help="paths to *_loco_summary.json files to merge")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="output path for merged summary")
    args = p.parse_args()

    all_folds: list[dict] = []
    all_seeds: set[int] = set()
    for path in args.summaries:
        data = json.loads(path.read_text())
        for seed in data.get("seeds", []):
            all_seeds.add(int(seed))
        for rec in data.get("per_fold", []):
            all_folds.append(rec)

    if not all_folds:
        print("[error] no per_fold records found", file=sys.stderr)
        return 2

    # Group by holdout_cohort.
    by_cohort: dict[str, list[dict]] = {}
    for rec in all_folds:
        by_cohort.setdefault(rec["holdout_cohort"], []).append(rec)

    per_cohort_stats: list[dict] = []
    for cohort, recs in by_cohort.items():
        aucs = [r["test_auc"] for r in recs]
        bases = [r["baseline_test_auc"] for r in recs]
        rfs = [r.get("rf_test_auc") for r in recs if r.get("rf_test_auc") is not None]
        row = {
            "holdout_cohort": cohort,
            "n_seeds": len(aucs),
            "gnn_mean_auc": st.fmean(aucs),
            "gnn_sd":       st.pstdev(aucs) if len(aucs) > 1 else 0.0,
            "baseline_mean_auc": st.fmean(bases),
            "baseline_sd":       st.pstdev(bases) if len(bases) > 1 else 0.0,
            "gap_mean":     st.fmean(a - b for a, b in zip(aucs, bases)),
            "n_test_samples": recs[0].get("num_test"),
        }
        if rfs:
            row["rf_mean_auc"] = st.fmean(rfs)
            row["rf_sd"] = st.pstdev(rfs) if len(rfs) > 1 else 0.0
            row["gap_gnn_vs_rf"] = st.fmean(aucs) - st.fmean(rfs)
        per_cohort_stats.append(row)
    # sort by best GNN AUC descending for readability
    per_cohort_stats.sort(key=lambda r: -r["gnn_mean_auc"])

    # Global means across (seed × fold) combinations.
    gnn_all = [r["test_auc"] for r in all_folds]
    base_all = [r["baseline_test_auc"] for r in all_folds]
    rf_all = [r["rf_test_auc"] for r in all_folds if r.get("rf_test_auc") is not None]
    dod = {
        "dod_1_stratified_pooled_geq_065": None,  # not measured here
        "dod_2_loco_mean_geq_060": st.fmean(gnn_all) >= 0.60,
        "dod_3_gap_geq_003": st.fmean(a - b for a, b in zip(gnn_all, base_all)) >= 0.03,
        "dod_7_max_per_cohort_sd_lt_008": max(s["gnn_sd"] for s in per_cohort_stats) < 0.08,
    }

    global_block = {
        "gnn_mean_auc":      st.fmean(gnn_all),
        "gnn_sd":            st.pstdev(gnn_all),
        "baseline_mean_auc": st.fmean(base_all),
        "baseline_sd":       st.pstdev(base_all),
        "gap_mean":          st.fmean(a - b for a, b in zip(gnn_all, base_all)),
    }
    if rf_all and len(rf_all) == len(gnn_all):
        global_block["rf_mean_auc"] = st.fmean(rf_all)
        global_block["rf_sd"] = st.pstdev(rf_all)
        global_block["gap_gnn_vs_rf"] = st.fmean(a - r for a, r in zip(gnn_all, rf_all))
        global_block["gap_rf_vs_mlp"] = st.fmean(r - b for r, b in zip(rf_all, base_all))

    merged = {
        "seeds":   sorted(all_seeds),
        "folds":   sorted(by_cohort.keys()),
        "n_runs":  len(all_folds),
        "per_cohort": per_cohort_stats,
        "global": global_block,
        "dod":        dod,
        "source_files": [str(p) for p in args.summaries],
    }

    out = args.output or args.summaries[0].parent / f"loco_3seed_merged.json"
    out.write_text(json.dumps(merged, indent=2))
    print(f"[done] wrote {out}")
    print()
    print(f"seeds merged: {sorted(all_seeds)}")
    print(f"folds:        {len(by_cohort)}")
    print(f"runs:         {len(all_folds)}")
    print()
    print("per-cohort (mean AUC ± SD across seeds):")
    for s in per_cohort_stats:
        rf_part = (f"  RF={s['rf_mean_auc']:.3f}±{s.get('rf_sd', 0):.3f}"
                   if "rf_mean_auc" in s else "")
        print(f"  {s['holdout_cohort']:24s}  N_test={s['n_test_samples']:>3d}  "
              f"GNN={s['gnn_mean_auc']:.3f}±{s['gnn_sd']:.3f}  "
              f"MLP={s['baseline_mean_auc']:.3f}"
              f"{rf_part}  "
              f"gap(GNN-MLP)={s['gap_mean']:+.3f}")
    print()
    g = merged["global"]
    print(f"global  GNN AUC  = {g['gnn_mean_auc']:.3f} ± {g['gnn_sd']:.3f}")
    print(f"        MLP AUC  = {g['baseline_mean_auc']:.3f} ± {g['baseline_sd']:.3f}")
    print(f"        gap      = {g['gap_mean']:+.3f}")
    print()
    print("DoD check:")
    for k, v in dod.items():
        if v is None:
            print(f"  {k}: skipped")
        else:
            print(f"  {k}: {'✅ PASS' if v else '❌ FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
