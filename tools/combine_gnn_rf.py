"""A2-M4.1 · merge GNN/MLP (loco_3seed_merged.json) + RF (rf_loco_summary.json).

Joins the two summaries by holdout cohort so one file
(``loco_3seed_with_rf.json``) carries every column AIM2_REPORT needs.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gnn-mlp",    required=True, type=Path)
    p.add_argument("--rf-summary", required=True, type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()

    gm  = json.loads(args.gnn_mlp.read_text())
    rf  = json.loads(args.rf_summary.read_text())

    rf_by_cohort = {c["holdout_cohort"]: c for c in rf["per_cohort"]}

    # Inject rf stats into each per_cohort row of the GNN/MLP merged file.
    per_cohort: list[dict] = []
    for row in gm["per_cohort"]:
        h = row["holdout_cohort"]
        out = dict(row)
        if h in rf_by_cohort:
            out["rf_mean_auc"] = rf_by_cohort[h]["rf_mean_auc"]
            out["rf_sd"]       = rf_by_cohort[h]["rf_sd"]
            out["gap_gnn_vs_rf"] = row["gnn_mean_auc"] - rf_by_cohort[h]["rf_mean_auc"]
            out["gap_rf_vs_mlp"] = rf_by_cohort[h]["rf_mean_auc"] - row["baseline_mean_auc"]
        per_cohort.append(out)

    global_block = dict(gm["global"])
    if rf.get("global"):
        global_block["rf_mean_auc"] = rf["global"]["rf_mean_auc"]
        global_block["rf_sd"]       = rf["global"]["rf_sd"]
        global_block["gap_gnn_vs_rf"] = gm["global"]["gnn_mean_auc"] - rf["global"]["rf_mean_auc"]
        global_block["gap_rf_vs_mlp"] = rf["global"]["rf_mean_auc"] - gm["global"]["baseline_mean_auc"]

    combined = {
        **gm,
        "per_cohort": per_cohort,
        "global":     global_block,
        "rf_source":  str(args.rf_summary),
    }

    args.output.write_text(json.dumps(combined, indent=2))
    print(f"[done] wrote {args.output}")
    print()
    print("per-cohort summary (GNN / MLP / RF):")
    print(f"  {'cohort':24s}  {'N':>3s}  {'GNN':>14s}  {'MLP':>7s}  {'RF':>14s}")
    for c in sorted(per_cohort, key=lambda r: -r.get("rf_mean_auc", 0)):
        n = c["n_test_samples"]
        gnn = f"{c['gnn_mean_auc']:.3f}±{c['gnn_sd']:.3f}"
        mlp = f"{c['baseline_mean_auc']:.3f}"
        rf_s = (f"{c['rf_mean_auc']:.3f}±{c['rf_sd']:.3f}" if "rf_mean_auc" in c else "n/a")
        print(f"  {c['holdout_cohort']:24s}  {n:>3d}  {gnn:>14s}  {mlp:>7s}  {rf_s:>14s}")
    g = combined["global"]
    print()
    print(f"global  GNN={g['gnn_mean_auc']:.3f}±{g['gnn_sd']:.3f}  "
          f"MLP={g['baseline_mean_auc']:.3f}±{g['baseline_sd']:.3f}  "
          f"RF={g.get('rf_mean_auc', float('nan')):.3f}±{g.get('rf_sd', 0):.3f}")
    print(f"        gap(GNN-MLP)={g['gap_mean']:+.3f}  "
          f"gap(GNN-RF)={g.get('gap_gnn_vs_rf', 0):+.3f}  "
          f"gap(RF-MLP)={g.get('gap_rf_vs_mlp', 0):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
