"""Read the M5 XAI heatmap CSVs and print the matrix as a text grid.

This is the 20×11 table the heatmap PNG is literally coloring in. Useful
for PI walkthroughs when projector rendering is unreliable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    run_dir = Path("outputs/20260420_191607_seed42")
    df = pd.read_csv(run_dir / "xai/node_attributions.csv")
    grid = (
        df.assign(abs_attr=df["attr"].abs())
        .groupby(["sample_id", "module"])["abs_attr"]
        .sum()
        .unstack(fill_value=0.0)
    )
    cols = [f"M{i}" for i in range(1, 12)]
    grid = grid[cols]

    print("HEATMAP matrix — 20 patients × 11 modules (Σ |IG attribution|):")
    print("=" * 100)
    print(f"{'sample_id':<22s} " + " ".join(f"{c:>7s}" for c in cols))
    for sid in grid.index:
        row = grid.loc[sid]
        print(f"{sid:<22s} " + " ".join(f"{v:7.2f}" for v in row))
    print("=" * 100)

    print(f"\nPer-module mean |attr| (col sums ÷ 20) — this is what sanity.json ranks:")
    col_mean = grid.mean(axis=0)
    for mid in cols:
        bar = "█" * int(col_mean[mid] * 30 / max(col_mean.max(), 1e-9))
        print(f"  {mid:3s}  {bar:<30s}  {col_mean[mid]:.3f}")

    top3 = col_mean.nlargest(3).index.tolist()
    print(f"\nTop-3 modules (these are the bright columns): {top3}")

    print("\nPer-patient total |attr| (row sums) — measures how much signal this patient carried:")
    row_sum = grid.sum(axis=1).sort_values(ascending=False)
    for sid in row_sum.index:
        bar = "▓" * int(row_sum[sid] * 30 / row_sum.max())
        print(f"  {sid:<22s}  {bar:<30s}  {row_sum[sid]:.2f}")


if __name__ == "__main__":
    main()
