"""M2 · T2.4 — ICD-readiness surrogate label (z-score signature + median split).

Formula (PLAN §3.2):

    score = z(CALR) + z(HMGB1) + z(HSPA1A) + z(HSP90AA1) − z(CD47) − z(CD24)
    label = 1  if score > median(score)  else 0

This is **not** an ICI-response label. PLAN §3.2 requires that every output
touching this label be annotated with that caveat (DATACARD.md / report.md /
demo_card.md). Aim 2 swaps this for IMvigor210's real response column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


POSITIVE_GENES: tuple[str, ...] = ("CALR", "HMGB1", "HSPA1A", "HSP90AA1")
NEGATIVE_GENES: tuple[str, ...] = ("CD47", "CD24")


@dataclass(frozen=True)
class LabelReport:
    genes_found: list[str]
    genes_missing: list[str]
    pos_rate: float   # fraction of label == 1
    threshold: float  # median of the continuous signature

    def as_dict(self) -> dict:
        return {
            "genes_found": self.genes_found,
            "genes_missing": self.genes_missing,
            "pos_rate": self.pos_rate,
            "threshold": self.threshold,
        }


def _zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sd = series.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=series.index, name=series.name)
    return (series - mu) / sd


def icd_readiness_signature(expr: pd.DataFrame) -> tuple[pd.Series, LabelReport]:
    """Compute the continuous ICD-readiness score.

    Missing genes are silently dropped from the sum rather than failing —
    a COAD matrix with one missing HSP is still diagnostic. ``LabelReport``
    records which genes were actually used so callers can audit.
    """
    present_pos = [g for g in POSITIVE_GENES if g in expr.columns]
    present_neg = [g for g in NEGATIVE_GENES if g in expr.columns]
    missing = [g for g in POSITIVE_GENES + NEGATIVE_GENES if g not in expr.columns]
    if not present_pos and not present_neg:
        raise ValueError(
            "none of the signature genes (CALR/HMGB1/HSPA1A/HSP90AA1/CD47/CD24) "
            "are in expr.columns — cannot build ICD-readiness label"
        )

    score = pd.Series(0.0, index=expr.index)
    for g in present_pos:
        score = score + _zscore(expr[g])
    for g in present_neg:
        score = score - _zscore(expr[g])
    score.name = "icd_readiness_score"

    report = LabelReport(
        genes_found=present_pos + present_neg,
        genes_missing=missing,
        pos_rate=float("nan"),  # filled by icd_readiness_label
        threshold=float("nan"),
    )
    return score, report


def icd_readiness_label(expr: pd.DataFrame) -> tuple[pd.Series, LabelReport]:
    """Compute the binary ICD-readiness label via median split on the score.

    Returns
    -------
    tuple[pd.Series, LabelReport]
        ``(label, report)`` where ``label`` ∈ {0,1} indexed by sample.

    Plan §5 · M2 DoD asks for ``pos_rate ≈ 0.5`` (median split makes this a
    tautology modulo ties; a test in ``tests/test_data.py`` guards the
    0.45–0.55 range anyway).
    """
    score, _report = icd_readiness_signature(expr)
    threshold = float(score.median())
    label = (score > threshold).astype(np.int64).rename("label")
    pos_rate = float(label.mean())
    report = LabelReport(
        genes_found=_report.genes_found,
        genes_missing=_report.genes_missing,
        pos_rate=pos_rate,
        threshold=threshold,
    )
    return label, report
