"""M2 · T2.5 — stratified 60/20/20 train/val/test split with fixed seed.

A2-M3 · T3.2 — :func:`leave_one_cohort_out` added for Aim 2 LOCO evaluation
(plan §2.4). Existing :func:`stratified_split` is unchanged so the demo
pipeline keeps working.
"""

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


def leave_one_cohort_out(
    cohort_ids: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    *,
    holdout_cohort: str,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Split:
    """LOCO split — one cohort → test, others → train (with carve-out val).

    Parameters
    ----------
    cohort_ids : Series or array of shape (n,)
        Cohort ID per sample (e.g., ``"riaz_gse91061"``). Order must match
        ``labels``.
    labels : Series or array of shape (n,)
        Binary labels. Values can be NaN / pd.NA — those samples are **dropped
        from val stratification** but kept in the fold they belong to
        (unlabeled non-holdout samples go to train; unlabeled holdout samples
        still go to test).
    holdout_cohort : str
        Which ``cohort_ids`` value to use as the test fold. Raises if absent.
    val_frac : float, default 0.15
        Fraction of **labeled** training samples to carve out for early-stop
        validation. Stratified by label.
    seed : int, default 42
        sklearn random_state.

    Returns
    -------
    Split
        ``test_idx`` = all samples with ``cohort_ids == holdout_cohort``.
        ``train_idx + val_idx`` = all other samples. ``val_idx`` is a
        label-stratified ``val_frac`` carve-out of the labeled non-holdout
        samples; unlabeled non-holdout samples go to ``train_idx``.

    Raises
    ------
    ValueError
        If ``holdout_cohort`` is not present in ``cohort_ids``,
        ``len(cohort_ids) != len(labels)``, or fewer than 5 labeled
        non-holdout samples remain for the val carve-out.

    Notes
    -----
    Plan §2.4 — this replaces the demo's stratified_split as the primary
    Aim 2 eval protocol. Each of the 6 Tier A cohorts becomes a fold once,
    giving the 6-fold LOCO AUC reported in `eval/loco_transfer.py`.
    """
    cohort_ids = pd.Series(cohort_ids).reset_index(drop=True)
    labels     = pd.Series(labels).reset_index(drop=True)
    if len(cohort_ids) != len(labels):
        raise ValueError(
            f"cohort_ids ({len(cohort_ids)}) / labels ({len(labels)}) length mismatch"
        )
    if holdout_cohort not in cohort_ids.values:
        raise ValueError(
            f"holdout_cohort {holdout_cohort!r} not in cohort_ids "
            f"(known: {sorted(cohort_ids.unique().tolist())})"
        )

    n = len(cohort_ids)
    all_idx = np.arange(n)
    test_mask = (cohort_ids == holdout_cohort).to_numpy()
    idx_test = all_idx[test_mask]
    idx_rest = all_idx[~test_mask]

    labeled_mask = labels.loc[idx_rest].notna().to_numpy()
    labeled_rest  = idx_rest[labeled_mask]
    unlabeled_rest = idx_rest[~labeled_mask]

    if len(labeled_rest) < 5:
        raise ValueError(
            f"only {len(labeled_rest)} labeled non-holdout samples — "
            "cannot carve a val fold"
        )

    y_for_strat = labels.loc[labeled_rest].astype(float).to_numpy()
    idx_train_labeled, idx_val = train_test_split(
        labeled_rest,
        test_size=val_frac,
        stratify=y_for_strat,
        random_state=seed,
    )
    idx_train = np.sort(np.concatenate([idx_train_labeled, unlabeled_rest]))
    idx_val   = np.sort(idx_val)
    idx_test  = np.sort(idx_test)

    union = np.concatenate([idx_train, idx_val, idx_test])
    assert union.size == n
    assert np.unique(union).size == n
    return Split(train_idx=idx_train, val_idx=idx_val, test_idx=idx_test)
