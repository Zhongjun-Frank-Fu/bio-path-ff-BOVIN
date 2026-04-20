"""M6 smoke tests — metrics CIs, report markdown assembly, LUAD transfer.

Real cross-cohort verification happens in Docker; these tests guard schema
+ wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("pytorch_lightning")


# ------------------------------ T6.1a -------------------------------------
def test_compute_metrics_returns_finite_numbers():
    from bovin_demo.eval import compute_metrics

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=60)
    # Correlate probs with labels so AUC > 0.5.
    y_prob = y_true * 0.6 + rng.normal(0.2, 0.1, size=60)
    y_prob = np.clip(y_prob, 0, 1)

    out = compute_metrics(y_true, y_prob, bootstrap=50, seed=42)
    assert set(out.keys()) == {"auc", "accuracy", "f1", "brier", "ece"}
    for key, stats in out.items():
        assert set(stats.keys()) == {"mean", "ci_lo", "ci_hi"}
        assert np.isfinite(stats["mean"]), (key, stats)
        assert stats["ci_lo"] <= stats["mean"] <= stats["ci_hi"] + 1e-9


def test_compute_metrics_skips_bootstrap_when_requested():
    from bovin_demo.eval import compute_metrics

    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    out = compute_metrics(y, p, bootstrap=0)
    for stats in out.values():
        assert np.isnan(stats["ci_lo"])
        assert np.isnan(stats["ci_hi"])


def test_compute_metrics_rejects_empty():
    from bovin_demo.eval import compute_metrics

    with pytest.raises(ValueError):
        compute_metrics(np.array([]), np.array([]))


# ------------------------------ T6.1b -------------------------------------
@pytest.fixture(scope="module")
def tiny_eval_config(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("cfg") / "tcga_eval_tiny.yaml"
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


def test_build_report_writes_markdown(xena_like_raw_dir, tmp_path, tiny_eval_config):
    from bovin_demo.eval import build_report
    from bovin_demo.train import run_training

    result = run_training(
        tiny_eval_config,
        seed=11,
        output_root=tmp_path,
        raw_dir_override=xena_like_raw_dir,
    )
    report_path = build_report(
        result.run_dir,
        config_path=tiny_eval_config,
        raw_dir_override=xena_like_raw_dir,
        bootstrap=20,
    )
    md = report_path.read_text(encoding="utf-8")
    assert "DoD checklist" in md
    assert "Run summary" in md
    assert "Test-fold metrics" in md
    assert "Surrogate label" in md


def test_build_report_embeds_xai_when_present(xena_like_raw_dir, tmp_path, tiny_eval_config):
    from bovin_demo.eval import build_report
    from bovin_demo.train import run_training
    from bovin_demo.xai import run_xai

    result = run_training(
        tiny_eval_config,
        seed=5,
        output_root=tmp_path,
        raw_dir_override=xena_like_raw_dir,
    )
    run_xai(
        result.run_dir,
        config_path=tiny_eval_config,
        raw_dir_override=xena_like_raw_dir,
        seed=5,
        top_n_patients=5,
        n_steps=4,
    )
    report = build_report(
        result.run_dir,
        config_path=tiny_eval_config,
        raw_dir_override=xena_like_raw_dir,
        bootstrap=0,
        recompute_test=False,
    )
    md = report.read_text(encoding="utf-8")
    assert "XAI sanity" in md
    assert "xai/xai_heatmap.png" in md


# ------------------------------ T6.4 --------------------------------------
def test_luad_transfer_uses_fixture_as_luad(xena_like_raw_dir, tmp_path, tiny_eval_config):
    """T6.4 runs an already-trained model against a LUAD raw_dir. We cheat
    here by pointing ``luad_raw_dir`` at the COAD fixture — good enough to
    verify the inference path (alignment, score, AUC computation) works."""
    from bovin_demo.eval import run_luad_zero_shot
    from bovin_demo.train import run_training

    result = run_training(
        tiny_eval_config,
        seed=99,
        output_root=tmp_path,
        raw_dir_override=xena_like_raw_dir,
    )
    out = run_luad_zero_shot(
        result.run_dir,
        config_path=tiny_eval_config,
        luad_raw_dir=xena_like_raw_dir,
    )
    assert out["n_samples"] > 0
    assert 0.0 <= out["auc"] <= 1.0
    assert 0.0 <= out["alignment_hit_rate"] <= 1.0


# ------------------------------ CLI eval ----------------------------------
def test_cli_eval_produces_report(xena_like_raw_dir, tmp_path, tiny_eval_config, capsys):
    from bovin_demo.cli import main
    from bovin_demo.train import run_training

    result = run_training(
        tiny_eval_config,
        seed=21,
        output_root=tmp_path,
        raw_dir_override=xena_like_raw_dir,
    )
    code = main(
        [
            "eval",
            "--run-dir", str(result.run_dir),
            "--config", str(tiny_eval_config),
            "--raw-dir", str(xena_like_raw_dir),
            "--bootstrap", "10",
        ]
    )
    assert code == 0
    assert (result.run_dir / "report.md").exists()
