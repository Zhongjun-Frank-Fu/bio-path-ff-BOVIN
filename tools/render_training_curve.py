"""Quick training-curve render from lightning's metrics.csv."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def render(csv_path: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    per_epoch_train: dict[int, list[float]] = defaultdict(list)
    per_epoch_val: dict[int, float] = {}
    per_epoch_val_loss: dict[int, float] = {}
    test_auc = None

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["epoch"]) if row.get("epoch") else None
            if ep is None:
                continue
            if row.get("train_loss"):
                per_epoch_train[ep].append(float(row["train_loss"]))
            if row.get("val_auc"):
                per_epoch_val[ep] = float(row["val_auc"])
            if row.get("val_loss"):
                per_epoch_val_loss[ep] = float(row["val_loss"])
            if row.get("test_auc"):
                test_auc = float(row["test_auc"])

    epochs = sorted(per_epoch_train.keys())
    train_mean = [sum(per_epoch_train[e]) / len(per_epoch_train[e]) for e in epochs]
    val_ep = sorted(per_epoch_val.keys())
    val_auc = [per_epoch_val[e] for e in val_ep]
    val_loss = [per_epoch_val_loss[e] for e in val_ep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

    ax1.plot(epochs, train_mean, color="#EF5B5B", linewidth=1.8, label="train_loss (mean/epoch)")
    ax1.plot(val_ep, val_loss, color="#5B8DEF", linewidth=1.8, label="val_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (BCE)")
    ax1.set_title("Training & validation loss")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    ax2.plot(val_ep, val_auc, color="#4FB286", linewidth=2.0, marker="o", markersize=3, label="val_auc")
    if test_auc is not None:
        ax2.axhline(test_auc, color="#F2A33A", linestyle="--", linewidth=1.3,
                    label=f"test_auc = {test_auc:.3f}")
    best = max(val_auc)
    best_ep = val_ep[val_auc.index(best)]
    ax2.scatter([best_ep], [best], color="#C0392B", zorder=5, s=60,
                label=f"best val_auc = {best:.3f} @ ep {best_ep}")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("AUC")
    ax2.set_ylim(0.55, 1.0)
    ax2.set_title("Validation AUC (EarlyStopping on val_auc)")
    ax2.legend(frameon=False, loc="lower right")
    ax2.grid(alpha=0.25)

    fig.suptitle("BOVIN-Pathway HeteroGNN · TCGA-COAD · seed=42", fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_training_curve] wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    render(args.csv, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
