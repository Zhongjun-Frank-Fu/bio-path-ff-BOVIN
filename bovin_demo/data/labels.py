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


def crt_quartile_label(
    expr: pd.DataFrame, gene: str = "CALR"
) -> tuple[pd.Series, LabelReport]:
    """Binary label = top-quartile vs bottom-quartile of a single gene.

    Motivation (vs. icd_readiness_label):
        The 6-gene composite signature doesn't faithfully replicate Peng 2025
        Fig.1C (CRT-high → longer PFI) — middle-50% noise + negative terms
        dilute the signal. GEPIA2's default quartile split on CRT *alone*
        does replicate Peng. This label returns:

            1   if   expr[gene] >= q75
            0   if   expr[gene] <= q25
            NaN otherwise   (caller filters these out)

    Returns (label_series, LabelReport).
    """
    if gene not in expr.columns:
        raise ValueError(f"gene {gene!r} not in expression matrix")
    s = expr[gene].astype(float)
    q25, q75 = s.quantile([0.25, 0.75])
    label = pd.Series(np.nan, index=s.index, name="label", dtype="float64")
    label[s >= q75] = 1.0
    label[s <= q25] = 0.0
    kept = label.dropna()
    pos_rate = float(kept.mean()) if len(kept) else float("nan")
    report = LabelReport(
        genes_found=[gene],
        genes_missing=[],
        pos_rate=pos_rate,
        threshold=float(q75),
    )
    return label, report


# ---------------------------------------------------------------------------
# Real-ICI-response labels — A2-M2 · T2.4 — RECIST binary mapping.
# ---------------------------------------------------------------------------

_DEFAULT_RECIST_MAPPING: dict[str, float] = {
    # Abbreviated RECIST codes (BMS/GEO common)
    "CR": 1.0, "PR": 1.0, "SD": 0.0, "PD": 0.0,
    # Full RECIST names (cBioPortal / Gide)
    "Complete Response": 1.0, "Partial Response": 1.0,
    "Stable Disease": 0.0, "Progressive Disease": 0.0,
    # Responder/non-responder shortcut (Seo, Sade-Feldman)
    "responder": 1.0, "non-responder": 0.0, "R": 1.0, "NR": 0.0,
    # Not-evaluable → explicit NaN so callers drop the row.
    "NE": float("nan"), "NA": float("nan"), "Not Evaluable": float("nan"),
}


def recist_binary_label(
    clinical: pd.DataFrame,
    response_col: str = "response_raw",
    mapping: dict[str, float] | None = None,
) -> tuple[pd.Series, LabelReport]:
    """Aim-2 primary label — binary RECIST → {1, 0, NaN}.

    Parameters
    ----------
    clinical : pd.DataFrame
        Must carry ``response_col`` (string / categorical). Index is the
        sample ID the label series will adopt.
    response_col : str
        Column with the raw response codes. Default ``"response_raw"`` —
        the canonical column the ``ici_loader`` adapters populate.
    mapping : dict[str, float] | None
        Override the default RECIST mapping. Keys are matched **case-sensitive**
        against the raw values; unmapped values become NaN and are reported
        in ``LabelReport.genes_missing`` (repurposed as "mapping misses").

    Returns
    -------
    tuple[pd.Series, LabelReport]
        * ``label`` — samples × 1, ∈ {0, 1, NaN}. Named ``ici_response``.
        * ``report`` — aggregate stats mirroring :class:`LabelReport` shape
          (re-using the dataclass so downstream code stays uniform).

    Notes
    -----
    Plan §2.2 defines this as the Aim 2 primary label. The default mapping is
    *CR/PR → 1, SD/PD → 0, NE → NaN*; override via ``mapping`` if a config
    wants to (e.g.) count SD as response (rare).
    """
    if response_col not in clinical.columns:
        raise ValueError(f"clinical frame has no {response_col!r} column")

    resolved = dict(_DEFAULT_RECIST_MAPPING)
    if mapping:
        resolved.update(mapping)

    raw = clinical[response_col].astype("object")
    label = raw.map(resolved).astype("Float64")
    label.name = "ici_response"

    observed = set(raw.dropna().unique())
    mapped = set(resolved).intersection(observed)
    misses = sorted(observed - set(resolved))

    non_na = label.dropna()
    pos_rate = float(non_na.mean()) if len(non_na) else float("nan")
    report = LabelReport(
        genes_found=sorted(mapped),     # re-purposed: codes that mapped cleanly
        genes_missing=misses,            # re-purposed: codes seen but not mapped
        pos_rate=pos_rate,
        threshold=float("nan"),          # not applicable — no continuous score here
    )
    return label, report


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
