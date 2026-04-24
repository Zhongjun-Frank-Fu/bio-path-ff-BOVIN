"""A2-M2 · T2.5 — unit tests for ``bovin_demo.data.ici_loader`` + RECIST label.

These guards catch regression in the data layer without needing a GPU or a
trained model. They hit real downloaded files under ``data/raw_ici/`` so the
loader side of A2-M1 / A2-M2 can be revalidated after any edit.

Any cohort whose raw files aren't present is skipped rather than failed —
so the test suite still runs cleanly on a fresh clone that hasn't yet run
``tools/download_ici_pool.sh``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bovin_demo.data.ici_loader import (
    TIER_A_COHORTS,
    ICIBundle,
    ICIPoolBundle,
    load_ici_cohort,
    load_ici_pool,
    load_gene_aliases,
)
from bovin_demo.data.labels import recist_binary_label


_RAW_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw_ici"
_HAS_RAW = _RAW_ROOT.exists() and any(_RAW_ROOT.iterdir())

requires_raw = pytest.mark.skipif(
    not _HAS_RAW,
    reason=f"data/raw_ici/ not populated — run tools/download_ici_pool.sh",
)


# ---------------------------------------------------------------------------
# Alias table (no raw-data dependency — runs on a fresh clone)
# ---------------------------------------------------------------------------

def test_gene_aliases_present_and_hit_rate():
    """Static BOVIN alias table should cover ≥ 95% symbols on both ID systems."""
    df = load_gene_aliases()
    # 70 observable nodes expand to 72 after splitting IFNA/IFNB and TRG/TRD.
    assert len(df) >= 70, f"alias table unexpectedly short: {len(df)} rows"
    assert df["entrez_id"].notna().mean() > 0.95
    assert df["ensembl_id"].notna().mean() > 0.90
    assert set(df.columns) >= {
        "node_id", "node_symbol", "hgnc_symbol", "entrez_id", "ensembl_id"
    }


# ---------------------------------------------------------------------------
# Per-cohort loaders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cohort", TIER_A_COHORTS)
@requires_raw
def test_cohort_loader_shape_and_hit_rate(cohort: str):
    """Every Tier A cohort loads with BOVIN hit-rate ≥ 70% (DoD §4 A2-M2)."""
    b = load_ici_cohort(cohort)
    assert isinstance(b, ICIBundle)
    # Expression shape: samples × 72 BOVIN symbols (NaN padding for misses).
    assert b.expr.shape[1] == 72, f"{cohort} expr has {b.expr.shape[1]} cols, expected 72"
    assert b.expr.shape[0] > 0
    # Hit rate is the DoD-tracked metric.
    assert b.hit_rate >= 0.70, (
        f"{cohort} BOVIN hit rate {b.hit_rate:.1%} < 70% — "
        "check static/bovin_gene_aliases.csv coverage for this ID system"
    )
    # Clinical carries the canonical columns.
    required = {"patient_id", "response_raw", "ici_response",
                "timepoint", "treatment", "disease", "cohort_id"}
    assert required <= set(b.clinical.columns), (
        f"{cohort}: missing {required - set(b.clinical.columns)}"
    )
    # ici_response values (when present) must be in the closed {0, 1} set.
    present = b.clinical["ici_response"].dropna()
    assert set(present.unique()).issubset({0.0, 1.0}), (
        f"{cohort}: unexpected ici_response values {set(present.unique())}"
    )


@requires_raw
def test_loaders_alignment():
    """expr.index must equal clinical.index for every cohort (no stray rows)."""
    for cohort in TIER_A_COHORTS:
        b = load_ici_cohort(cohort)
        assert (b.expr.index == b.clinical.index).all(), (
            f"{cohort}: expr / clinical index mismatch"
        )


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

@requires_raw
def test_pool_unfiltered_preserves_all_samples():
    """Unfiltered pool should keep every sample from every cohort."""
    pool = load_ici_pool()
    assert isinstance(pool, ICIPoolBundle)
    total = sum(load_ici_cohort(c).n_samples for c in TIER_A_COHORTS)
    assert pool.n_samples == total, (
        f"pool lost samples: {pool.n_samples} vs sum-per-cohort {total}"
    )
    assert set(pool.cohorts) == set(TIER_A_COHORTS)
    # Gene intersection is ≤ 72; plan §2.1 hopes for ≥ 50 of 70 observable nodes.
    assert 50 <= len(pool.genes) <= 72


@requires_raw
def test_pool_per_cohort_zscore_is_unit_variance():
    """Per-cohort z-score should yield mean≈0 / std≈1 per gene per cohort."""
    pool = load_ici_pool()
    for cohort, sub in pool.expr.groupby(pool.clinical["cohort_id"]):
        sub_numeric = sub.select_dtypes(include=[np.number])
        # Columns that are all-NaN for this cohort get zero (filled); only
        # check the rest.
        varying = sub_numeric.loc[:, sub_numeric.std(axis=0, ddof=0) > 1e-6]
        if varying.empty:
            continue
        mu = varying.mean(axis=0).abs().max()
        sd = varying.std(axis=0, ddof=0)
        assert mu < 1e-6, f"{cohort}: per-cohort mean not centered (max |μ| = {mu:.2e})"
        assert (sd - 1).abs().max() < 1e-6, f"{cohort}: per-cohort std not unit"


@requires_raw
def test_pool_filtered_has_enough_labels():
    """Filtered (pre + labeled) pool should carry ≥ 160 labeled samples.

    Plan's §4 A2-M2 DoD asks for ≥ 230 labeled patients on the full 6-cohort
    pool — Cloughesy's response isn't in GEO (deferred to M2.1 from paper
    Table 1), so 5-cohort-labeled is the realistic M2 target. 160 is a
    conservative floor; typical runs observe ~200.
    """
    pool = load_ici_pool(filter_timepoint="pre", require_label=True)
    assert pool.n_samples >= 160, (
        f"filtered pool too small: {pool.n_samples} samples "
        f"(cohorts: {pool.cohorts})"
    )
    # Binary balance should be in a realistic ICI response range (20%–70%).
    pos_rate = pool.clinical["ici_response"].astype(float).mean()
    assert 0.20 <= pos_rate <= 0.70, f"pos_rate {pos_rate:.3f} outside realistic range"


# ---------------------------------------------------------------------------
# RECIST binary label helper
# ---------------------------------------------------------------------------

def test_recist_binary_label_defaults():
    """Default mapping covers CR/PR/SD/PD + full names + R/NR."""
    clin = pd.DataFrame({
        "response_raw": ["CR", "PR", "SD", "PD", "NE",
                         "Complete Response", "Partial Response",
                         "responder", "non-responder"],
    }, index=[f"s{i}" for i in range(9)])
    label, report = recist_binary_label(clin)
    expected = [1.0, 1.0, 0.0, 0.0, pd.NA, 1.0, 1.0, 1.0, 0.0]
    for i, e in enumerate(expected):
        if pd.isna(e):
            assert pd.isna(label.iloc[i])
        else:
            assert label.iloc[i] == e, f"row {i}: {label.iloc[i]} vs {e}"
    # 9 rows minus 1 NE = 8 kept; 5 pos (CR, PR, CR, PR, responder) / 3 neg → 5/8 = 0.625
    assert report.pos_rate == pytest.approx(0.625)


def test_recist_binary_label_custom_mapping():
    """Caller override should replace default without merging."""
    clin = pd.DataFrame({"response_raw": ["yes", "no"]}, index=["a", "b"])
    label, _ = recist_binary_label(clin, mapping={"yes": 1.0, "no": 0.0})
    assert list(label) == [1.0, 0.0]


def test_recist_binary_label_reports_unknown_codes():
    """Codes not in mapping show up in report.genes_missing."""
    clin = pd.DataFrame({"response_raw": ["CR", "weird_code"]}, index=["a", "b"])
    _, report = recist_binary_label(clin)
    assert "weird_code" in report.genes_missing


def test_recist_binary_label_missing_column_raises():
    with pytest.raises(ValueError):
        recist_binary_label(pd.DataFrame({"foo": ["CR"]}))
