"""M6 · T6.1 — AUC / ACC / F1 / Brier / ECE + bootstrap CI.

All metrics are binary. ``compute_metrics`` returns a dict that survives a
round-trip through ``json.dumps`` so the report writer can embed it verbatim.

Bootstrap strategy
------------------
Resample with replacement on the (y_true, y_prob) pairs. ECE / Brier are
computed on the resampled pairs; AUC/ACC/F1 use the standard sklearn calls.
Confidence interval = percentile [2.5, 97.5].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricCI:
    mean: float
    ci_lo: float
    ci_hi: float

    def as_dict(self) -> dict:
        return {"mean": self.mean, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi}


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE: mean over bins of |conf - acc| weighted by bin size."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = (y_true[mask] == (y_prob[mask] >= 0.5)).mean()
        ece += mask.mean() * abs(bin_conf - bin_acc)
    return float(ece)


def _one_pass_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    if len(set(y_true.tolist())) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": _expected_calibration_error(y_true, y_prob),
    }


def compute_metrics(
    y_true,
    y_prob,
    *,
    bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """Return a metrics dict ready for ``report.md`` / ``metrics.json``.

    Parameters
    ----------
    y_true, y_prob : 1D arrays of same length (binary label, predicted prob).
    bootstrap : number of resamples for CI. Set 0 to skip CI (point only).
    seed : RNG seed — any plot that re-derives CI must use the same seed.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_prob.shape}")
    if y_true.size == 0:
        raise ValueError("y_true is empty")

    point = _one_pass_metrics(y_true, y_prob)
    if bootstrap <= 0:
        return {k: MetricCI(v, float("nan"), float("nan")).as_dict() for k, v in point.items()}

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in point}
    n = y_true.size
    for _ in range(bootstrap):
        idx = rng.integers(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_prob[idx]
        if len(set(y_t.tolist())) < 2:
            continue
        m = _one_pass_metrics(y_t, y_p)
        for k, v in m.items():
            if np.isfinite(v):
                samples[k].append(v)

    out: dict[str, dict] = {}
    for k, v in point.items():
        arr = np.asarray(samples[k]) if samples[k] else np.asarray([v])
        lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        out[k] = MetricCI(mean=v, ci_lo=lo, ci_hi=hi).as_dict()
    return out
