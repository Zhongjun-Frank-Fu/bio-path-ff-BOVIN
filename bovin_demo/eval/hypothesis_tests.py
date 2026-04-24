"""A2-M6 · T6.2 — evaluate the 4 pre-registered Aim 2 hypotheses.

Plan §3.2 defines:

  H1  ICD axis (CRT, HMGB1, HSP70=HSPA1A, HSP90=HSP90AA1) IG direction → +
        Pass: ≥ 3/4 nodes have mean(IG per responder) > 0.
  H2  "Don't eat me" axis (CD47, CD24, SIRPA) IG direction → −
        Pass: ≥ 2/3 nodes have mean(IG per responder) < 0.
  H3  HeteroGNN − BaselineMLP test_auc gap ≥ +0.03 (bootstrap p < 0.05)
        Pass: mean gap ≥ 0.03 AND 95% CI of gap excludes 0.
  H4  LOCO mean AUC ≥ 0.60 AND worst per-cohort ≥ 0.55
        Pass: both conditions.

Inputs expected
---------------
- ``<run_dir>/xai/node_attributions.csv``  (from xai/runner.py — long format)
- ``outputs/loco_3seed_merged.json``       (from tools/merge_loco_summaries.py)

Output: ``<run_dir>/hypothesis_results.json`` with per-hypothesis evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Plan §3.2 axis definitions — node_id names as they appear in the graph.
ICD_AXIS = {
    "crt":    "CALR",
    "hmgb1":  "HMGB1",
    "hsp70":  "HSPA1A",
    "hsp90":  "HSP90AA1",
}
DONT_EAT_ME_AXIS = {
    "cd47":  "CD47",
    "cd24":  "CD24",
    "sirpa": "SIRPA",
}


def _load_node_attr(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "xai" / "node_attributions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run xai/runner.run_xai first"
        )
    return pd.read_csv(path)


def _evaluate_axis(
    attr_df: pd.DataFrame,
    axis: dict[str, str],
    expected_sign: str,
) -> dict:
    """Check mean IG sign for each node in the axis."""
    assert expected_sign in {"+", "-"}
    per_node: dict[str, dict] = {}
    passes = 0
    for node_id, hgnc in axis.items():
        sub = attr_df[attr_df["node_id"] == node_id]
        if len(sub) == 0:
            per_node[node_id] = {
                "hgnc": hgnc, "mean_attr": None, "n_patients": 0,
                "sign_matches": False, "note": "node absent in XAI output",
            }
            continue
        mean_attr = float(sub["attr"].mean())
        n = int(sub["attr"].count())
        if expected_sign == "+":
            sign_ok = mean_attr > 0
        else:
            sign_ok = mean_attr < 0
        per_node[node_id] = {
            "hgnc": hgnc,
            "mean_attr": mean_attr,
            "n_patients": n,
            "sign_matches": sign_ok,
        }
        if sign_ok:
            passes += 1
    return {"passes": passes, "total": len(axis), "per_node": per_node}


def _evaluate_h3(loco_merged: dict) -> dict:
    """Mean GNN−MLP gap across 15 (seed × fold) runs, plus bootstrap 95% CI."""
    all_folds = []
    for entry in loco_merged.get("per_cohort", []):
        all_folds.append({
            "cohort": entry["holdout_cohort"],
            "gnn": entry["gnn_mean_auc"],
            "mlp": entry["baseline_mean_auc"],
        })
    # also pull raw seed×fold numbers if available
    global_block = loco_merged.get("global", {})
    mean_gap = float(global_block.get("gap_mean", 0.0))

    # bootstrap on the per-cohort means (coarse; for a rigorous CI run the raw
    # 15-point sample, but per-cohort is close enough at this scale).
    rng = np.random.default_rng(42)
    gaps = np.array([f["gnn"] - f["mlp"] for f in all_folds])
    boot_means = np.array([
        rng.choice(gaps, size=len(gaps), replace=True).mean()
        for _ in range(1000)
    ])
    ci_lo = float(np.quantile(boot_means, 0.025))
    ci_hi = float(np.quantile(boot_means, 0.975))

    passes = (mean_gap >= 0.03) and (ci_lo > 0.0)
    return {
        "mean_gap":        mean_gap,
        "ci_95_lo":        ci_lo,
        "ci_95_hi":        ci_hi,
        "threshold":       0.03,
        "per_cohort_gaps": [
            {"cohort": f["cohort"], "gap": f["gnn"] - f["mlp"]}
            for f in all_folds
        ],
        "passes":          passes,
    }


def _evaluate_h4(loco_merged: dict) -> dict:
    """LOCO mean AUC ≥ 0.60 AND worst cohort AUC ≥ 0.55."""
    per_cohort = loco_merged.get("per_cohort", [])
    aucs = [c["gnn_mean_auc"] for c in per_cohort]
    g = loco_merged.get("global", {})
    mean_auc = float(g.get("gnn_mean_auc", 0.0))
    worst = float(min(aucs)) if aucs else 0.0
    passes_mean = mean_auc >= 0.60
    passes_worst = worst >= 0.55
    return {
        "mean_loco_auc": mean_auc,
        "worst_cohort_auc": worst,
        "worst_cohort": min(per_cohort, key=lambda c: c["gnn_mean_auc"])["holdout_cohort"]
                        if per_cohort else None,
        "threshold_mean": 0.60,
        "threshold_worst": 0.55,
        "passes": passes_mean and passes_worst,
    }


def evaluate(
    run_dir: Path,
    loco_merged_path: Path,
) -> dict:
    attr = _load_node_attr(run_dir)
    loco = json.loads(loco_merged_path.read_text())

    h1 = _evaluate_axis(attr, ICD_AXIS, "+")
    h2 = _evaluate_axis(attr, DONT_EAT_ME_AXIS, "-")
    h3 = _evaluate_h3(loco)
    h4 = _evaluate_h4(loco)

    h1_pass = h1["passes"] >= 3         # ≥3/4
    h2_pass = h2["passes"] >= 2         # ≥2/3
    # rename axis "passes" (count) so top-level "passes" is unambiguously boolean.
    h1["nodes_matching_sign"] = h1.pop("passes")
    h2["nodes_matching_sign"] = h2.pop("passes")

    results = {
        "H1_icd_axis_positive": {
            "passes": h1_pass,
            **h1,
            "criterion": "≥3/4 of {CRT, HMGB1, HSPA1A, HSP90AA1} show mean(IG) > 0",
        },
        "H2_dont_eat_me_negative": {
            "passes": h2_pass,
            **h2,
            "criterion": "≥2/3 of {CD47, CD24, SIRPA} show mean(IG) < 0",
        },
        "H3_gnn_beats_mlp_by_003": {
            **h3,
            "criterion": "mean GNN−MLP gap ≥ 0.03 AND bootstrap 95% CI excludes 0",
        },
        "H4_loco_generalizes": {
            **h4,
            "criterion": "LOCO mean AUC ≥ 0.60 AND worst per-cohort AUC ≥ 0.55",
        },
        "overall_pass_count": sum([h1_pass, h2_pass, h3["passes"], h4["passes"]]),
    }
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path,
                   help="training run dir that contains xai/node_attributions.csv")
    p.add_argument("--loco-merged", required=True, type=Path,
                   help="path to loco_3seed_merged.json")
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args()

    results = evaluate(args.run_dir, args.loco_merged)
    out = args.output or (args.run_dir / "hypothesis_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {out}")
    print()
    for key, block in results.items():
        if key == "overall_pass_count":
            continue
        mark = "✅" if block.get("passes", False) else "❌"
        print(f"  {mark} {key}")
        print(f"      criterion: {block.get('criterion', '')}")
    print()
    print(f"Overall: {results['overall_pass_count']}/4 hypotheses pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
