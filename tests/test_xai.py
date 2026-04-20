"""M5 smoke tests — IG end-to-end + heatmap artifact + DoD sanity keys present.

As with M4, the synthetic Xena fixture doesn't carry biological signal, so
we test *plumbing* rather than DoD thresholds (PLAN §5 DoD #4 / #5 are
verified by hand on real TCGA-COAD). These tests guard:

* compute_node_attributions returns shape-correct arrays
* aggregate_by_module + rank_nodes give deterministic orderings
* plot_heatmap writes a non-empty PNG
* run_xai wires train → xai against a real ckpt and produces sanity.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("captum")
pytest.importorskip("matplotlib")
pytest.importorskip("pytorch_lightning")


@pytest.fixture(scope="module")
def tiny_xai_config(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("cfg") / "tcga_xai_tiny.yaml"
    out.write_text(
        """
paths:
  raw_dir: data/raw
  output_root: outputs
graph:
  add_self_loops: true
model:
  hidden_dim: 16
  num_intra_layers: 1
  num_inter_layers: 1
  attention_heads: 2
  dropout: 0.1
train:
  optimizer: adamw
  lr: 5.0e-3
  weight_decay: 1.0e-4
  batch_size: 16
  max_epochs: 2
  patience: 2
  scheduler: cosine
  loss: bce_with_logits
  class_weight: balanced
""",
        encoding="utf-8",
    )
    return out


def test_compute_node_attributions_shapes(xena_like_raw_dir, graph_json):
    """IG runs on a handful of patients and returns aligned arrays."""
    import torch

    from bovin_demo.data import (
        icd_readiness_label,
        load_coad,
        map_to_pathway_nodes,
    )
    from bovin_demo.data.dataset import build_patient_dataset
    from bovin_demo.model import build_classifier
    from bovin_demo.xai import compute_node_attributions

    bundle = load_coad(xena_like_raw_dir)
    aligned, _ = map_to_pathway_nodes(bundle.expr, graph_json)
    label, _ = icd_readiness_label(bundle.expr)
    common = aligned.index.intersection(label.index)
    aligned, label = aligned.loc[common], label.loc[common]

    ds = build_patient_dataset(graph_json, aligned, label)
    samples = [ds[i] for i in range(5)]
    from torch_geometric.loader import DataLoader as PyGDataLoader

    probe = next(iter(PyGDataLoader(samples, batch_size=5)))
    clf = build_classifier(probe, hidden_dim=16, num_intra_layers=1, num_inter_layers=1)
    with torch.no_grad():
        clf(probe)

    attr = compute_node_attributions(clf, samples, n_steps=4)
    assert attr.attributions.shape == (5, 82)
    assert len(attr.node_ids) == 82
    assert len(attr.modules) == 82
    assert attr.probs.shape == (5,)
    assert attr.labels.shape == (5,)


def test_module_rollup_and_node_rank_are_deterministic(xena_like_raw_dir, graph_json):
    import numpy as np
    import torch

    from bovin_demo.data import (
        icd_readiness_label,
        load_coad,
        map_to_pathway_nodes,
    )
    from bovin_demo.data.dataset import build_patient_dataset
    from bovin_demo.model import build_classifier
    from bovin_demo.xai import (
        aggregate_by_module,
        compute_node_attributions,
        rank_nodes,
    )

    bundle = load_coad(xena_like_raw_dir)
    aligned, _ = map_to_pathway_nodes(bundle.expr, graph_json)
    label, _ = icd_readiness_label(bundle.expr)
    common = aligned.index.intersection(label.index)
    aligned, label = aligned.loc[common], label.loc[common]

    ds = build_patient_dataset(graph_json, aligned, label)
    samples = [ds[i] for i in range(4)]
    from torch_geometric.loader import DataLoader as PyGDataLoader

    probe = next(iter(PyGDataLoader(samples, batch_size=4)))
    torch.manual_seed(0)
    clf = build_classifier(probe, hidden_dim=16, num_intra_layers=1, num_inter_layers=1)
    with torch.no_grad():
        clf(probe)

    attr = compute_node_attributions(clf, samples, n_steps=4)
    roll = aggregate_by_module(attr)
    rank = rank_nodes(attr)

    assert roll.matrix.shape == (4, 11)
    assert roll.mean_per_module.shape == (11,)
    assert len(roll.top_modules) == 11
    assert len(rank.node_ids) == 82
    # Rank is monotone non-increasing in mean_abs_attr.
    diffs = np.diff(rank.mean_abs_attr)
    assert (diffs <= 1e-5).all()


def test_plot_heatmap_writes_png(tmp_path):
    import numpy as np

    from bovin_demo.xai import plot_heatmap

    matrix = np.random.default_rng(0).random((5, 11))
    out = plot_heatmap(
        matrix,
        patient_ids=[f"p{i}" for i in range(5)],
        module_ids=[f"M{i}" for i in range(1, 12)],
        out_path=tmp_path / "heatmap.png",
    )
    assert out.exists()
    assert out.stat().st_size > 1000  # real PNG, not a 0-byte placeholder


def test_run_xai_end_to_end(xena_like_raw_dir, tmp_path, tiny_xai_config):
    """Train a tiny model, then run XAI against its ckpt; verify sanity.json."""
    from bovin_demo.train import run_training
    from bovin_demo.xai import run_xai

    result = run_training(
        tiny_xai_config,
        seed=3,
        output_root=tmp_path,
        raw_dir_override=xena_like_raw_dir,
    )
    sanity = run_xai(
        result.run_dir,
        config_path=tiny_xai_config,
        raw_dir_override=xena_like_raw_dir,
        seed=3,
        top_n_patients=5,
        n_steps=4,
    )

    # Artifacts
    xai_dir = result.run_dir / "xai"
    assert (xai_dir / "xai_heatmap.png").exists()
    assert (xai_dir / "node_attributions.csv").exists()
    assert (xai_dir / "module_summary.csv").exists()
    assert (xai_dir / "sanity.json").exists()

    # Sanity schema contract (pass/fail is not asserted — signal-free fixture)
    data = json.loads((xai_dir / "sanity.json").read_text())
    for key in ("top3_modules", "top5_nodes", "dod_4_m4_damp_in_top3_modules",
                "dod_5_landmark_in_top5_nodes", "landmarks_found", "n_patients"):
        assert key in data, f"missing sanity key: {key}"
    assert data["n_patients"] > 0
    assert sanity == data
