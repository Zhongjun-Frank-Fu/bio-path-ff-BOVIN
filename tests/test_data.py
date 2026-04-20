"""M2 tests — loader round-trip, alignment hit-rate, label balance, split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bovin_demo.data import (
    icd_readiness_label,
    icd_readiness_signature,
    load_coad,
    map_to_pathway_nodes,
    stratified_split,
)


# ------------------------------ T2.2 --------------------------------------
def test_load_coad_returns_aligned_samples(xena_like_raw_dir):
    bundle = load_coad(xena_like_raw_dir)
    assert bundle.n_samples == 80
    # Gene count must at least cover the pathway symbols we seeded.
    assert bundle.n_genes >= 60
    assert bundle.expr.index.name == "sample"
    # Clinical + expr share the same sample index.
    assert set(bundle.clinical.index) == set(bundle.expr.index)
    # Survival fixture is optional but present in this test.
    assert bundle.survival is not None
    assert "OS" in bundle.survival.columns


def test_load_coad_raises_on_missing_files(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_coad(empty)


# ------------------------------ T2.3 --------------------------------------
def test_alignment_hit_rate_exceeds_plan_dod(xena_like_raw_dir, graph_json):
    """Plan §5 · M2 DoD: hit_rate ≥ 0.70 for observable genes."""
    bundle = load_coad(xena_like_raw_dir)
    aligned, report = map_to_pathway_nodes(bundle.expr, graph_json)
    assert report.hit_rate >= 0.70, report.as_dict()
    # Aligned frame has one column per observable node (including misses).
    n_observable = sum(1 for n in graph_json["nodes"] if n["observable"])
    assert aligned.shape == (bundle.n_samples, n_observable)


def test_alignment_resolves_aggregate_symbols(xena_like_raw_dir, graph_json):
    bundle = load_coad(xena_like_raw_dir)
    _aligned, report = map_to_pathway_nodes(bundle.expr, graph_json)
    # The synthetic fixture seeds IFNA (→ IFNA/IFNB) and TRG (→ TRG/TRD).
    assert "type1_ifn" in report.aggregate_resolved
    assert "trg_d" in report.aggregate_resolved


def test_alignment_misses_are_nan(xena_like_raw_dir, graph_json):
    """Observable nodes without an expression column must surface as NaN
    columns — silent zero-imputation would corrupt downstream z-scoring."""
    bundle = load_coad(xena_like_raw_dir)
    aligned, report = map_to_pathway_nodes(bundle.expr, graph_json)
    if report.misses:
        miss = report.misses[0]
        assert aligned[miss].isna().all()


# ------------------------------ T2.4 --------------------------------------
def test_icd_readiness_label_median_split_balanced(xena_like_raw_dir):
    """Median split → pos_rate ≈ 0.5 (Plan §5 DoD: ~50/50)."""
    bundle = load_coad(xena_like_raw_dir)
    label, report = icd_readiness_label(bundle.expr)
    assert set(label.unique()) <= {0, 1}
    assert 0.45 <= report.pos_rate <= 0.55, report.as_dict()
    assert set(report.genes_found) >= {"CALR", "HMGB1", "CD47", "CD24"}


def test_icd_readiness_signature_is_deterministic(xena_like_raw_dir):
    bundle = load_coad(xena_like_raw_dir)
    score_a, _ = icd_readiness_signature(bundle.expr)
    score_b, _ = icd_readiness_signature(bundle.expr)
    pd.testing.assert_series_equal(score_a, score_b)


def test_icd_readiness_rejects_empty_signature():
    expr = pd.DataFrame(
        np.random.default_rng(0).normal(size=(10, 3)),
        columns=["FOO", "BAR", "BAZ"],
    )
    with pytest.raises(ValueError, match="signature genes"):
        icd_readiness_label(expr)


# ------------------------------ T2.5 --------------------------------------
def test_stratified_split_ratios_and_disjoint(xena_like_raw_dir):
    bundle = load_coad(xena_like_raw_dir)
    label, _ = icd_readiness_label(bundle.expr)
    split = stratified_split(label, seed=42)

    sizes = split.sizes()
    total = sum(sizes.values())
    assert total == bundle.n_samples
    # 60/20/20 with 80 samples → 48/16/16.
    assert sizes == {"train": 48, "val": 16, "test": 16}

    # Disjoint
    all_idx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
    assert np.unique(all_idx).size == total


def test_stratified_split_preserves_class_balance(xena_like_raw_dir):
    bundle = load_coad(xena_like_raw_dir)
    label, _ = icd_readiness_label(bundle.expr)
    split = stratified_split(label, seed=42)

    y = label.to_numpy()
    overall = y.mean()
    for idx in (split.train_idx, split.val_idx, split.test_idx):
        fold = y[idx].mean()
        assert abs(fold - overall) <= 0.08, (overall, fold)


def test_stratified_split_is_seed_stable(xena_like_raw_dir):
    bundle = load_coad(xena_like_raw_dir)
    label, _ = icd_readiness_label(bundle.expr)
    a = stratified_split(label, seed=42)
    b = stratified_split(label, seed=42)
    np.testing.assert_array_equal(a.train_idx, b.train_idx)
    np.testing.assert_array_equal(a.val_idx, b.val_idx)
    np.testing.assert_array_equal(a.test_idx, b.test_idx)


def test_stratified_split_rejects_bad_ratios():
    y = pd.Series([0, 1] * 20)
    with pytest.raises(ValueError, match="sum to 1.0"):
        stratified_split(y, ratios=(0.5, 0.3, 0.3))
