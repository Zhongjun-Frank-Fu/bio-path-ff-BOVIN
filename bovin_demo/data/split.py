"""M2 · T2.5 — stratified 60/20/20 train/val/test split with fixed seed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Split:
    """Index arrays for the three folds. Each array contains integer positions
    into the original label series (not TCGA barcodes) so downstream tensors
    can slice directly."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train_idx.size),
            "val": int(self.val_idx.size),
            "test": int(self.test_idx.size),
        }


def stratified_split(
    y: pd.Series | np.ndarray,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Split:
    """Stratified 60/20/20 split, label-balanced per fold.

    Parameters
    ----------
    y : Series or array-like of shape (n,)
        Binary labels.
    seed : int
        Numpy / sklearn seed. Default 42 — matches ``configs/default.yaml``.
    ratios : (train, val, test)
        Must sum to 1.0.

    Returns
    -------
    Split
        integer index arrays summing to ``len(y)``.
    """
    train_r, val_r, test_r = ratios
    if not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    y_arr = np.asarray(y)
    n = y_arr.size
    if n < 5:
        raise ValueError(f"need at least 5 samples, got {n}")

    all_idx = np.arange(n)
    # First cut: train vs (val + test).
    idx_train, idx_hold = train_test_split(
        all_idx,
        test_size=val_r + test_r,
        stratify=y_arr,
        random_state=seed,
    )
    # Second cut inside the holdout: split val/test proportionally.
    rel_test = test_r / (val_r + test_r)
    idx_val, idx_test = train_test_split(
        idx_hold,
        test_size=rel_test,
        stratify=y_arr[idx_hold],
        random_state=seed,
    )

    split = Split(
        train_idx=np.sort(idx_train),
        val_idx=np.sort(idx_val),
        test_idx=np.sort(idx_test),
    )
    # Sanity — strict disjoint union.
    union = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
    assert union.size == n
    assert np.unique(union).size == n
    return split
